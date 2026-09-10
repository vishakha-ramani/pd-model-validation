import json, math, os, sys, tempfile
import pytest
from calibration.admission.analysis import (
    load_meta, parse_admission_trajectory, parse_enqueue_events, request_traces,
    build_context, block_accounting_diag,
    RunningMean, deployable_rem_steps_est, enqueue_bucket, replay, admission_report,
    admission_censored_rows, write_admission_report,
)
from calibration.admission.estimators import ESTIMATORS
from calibration.modeb.analysis import predict_step, load_coeffs

STEPS = [
    {"step": 0, "t_start": 1.00, "t_end": 1.01, "total_scheduled": 4, "num_running": 0,
     "reqs": [{"id": "A", "kappa": 4, "computed": 0, "prompt_len": 4}],
     "waiting_count": 1, "waiting_ids": ["B"], "waiting_truncated": False, "free_kv_blocks": 100},
    {"step": 1, "t_start": 1.02, "t_end": 1.03, "total_scheduled": 5, "num_running": 1,
     "reqs": [{"id": "A", "kappa": 1, "computed": 4, "prompt_len": 4},
              {"id": "B", "kappa": 4, "computed": 0, "prompt_len": 4}],
     "waiting_count": 0, "waiting_ids": [], "waiting_truncated": False, "free_kv_blocks": 90},
    {"step": 2, "t_start": 1.04, "t_end": 1.05, "total_scheduled": 2, "num_running": 2,
     "reqs": [{"id": "A", "kappa": 1, "computed": 5, "prompt_len": 4},
              {"id": "B", "kappa": 1, "computed": 4, "prompt_len": 4}],
     "waiting_count": 0, "waiting_ids": [], "waiting_truncated": False, "free_kv_blocks": 80},
]

META = {"max_num_seqs": 256, "block_size": 16, "num_gpu_blocks": 1000}
COEFFS = load_coeffs(os.path.join(os.path.dirname(__file__), "..", "..", "..", "coeffs.json"))


def _write_jsonl(rows):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def test_parse_sorts_and_adds_t_iter():
    steps = parse_admission_trajectory(_write_jsonl(STEPS))
    assert [s["step"] for s in steps] == [0, 1]  # last dropped (no successor)
    assert abs(steps[0]["t_iter"] - 0.02) < 1e-9


def test_parse_adds_t_end_next_and_keeps_queue_kv_fields():
    steps = parse_admission_trajectory(_write_jsonl(STEPS))
    assert steps[0]["t_end_next"] == 1.02
    assert steps[1]["t_end_next"] == 1.04
    assert steps[0]["waiting_ids"] == ["B"]
    assert steps[0]["waiting_count"] == 1
    assert steps[0]["free_kv_blocks"] == 100


def test_parse_sorts_out_of_order_input():
    shuffled = [STEPS[2], STEPS[0], STEPS[1]]
    steps = parse_admission_trajectory(_write_jsonl(shuffled))
    assert [s["step"] for s in steps] == [0, 1]


def test_load_meta_reads_json():
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(META, f)
    assert load_meta(path) == META


def test_parse_enqueue_events_last_write_wins():
    rows = [
        {"req_id": "A", "t_enq": 0.5, "prompt_len": 4},
        {"req_id": "B", "t_enq": 1.02, "prompt_len": 4},
        {"req_id": "A", "t_enq": 0.9, "prompt_len": 999},  # duplicate: last wins
    ]
    events = parse_enqueue_events(_write_jsonl(rows))
    assert events["A"] == {"t_enq": 0.9, "prompt_len": 999}
    assert events["B"] == {"t_enq": 1.02, "prompt_len": 4}


def test_parse_enqueue_events_preserves_upstream_arrival_fields():
    rows = [{"req_id": "A", "t_engine_arrive": 0.99, "t_enq": 1.01,
             "engine_input_wait": 0.02, "prompt_len": 4}]
    events = parse_enqueue_events(_write_jsonl(rows))
    assert events["A"]["t_engine_arrive"] == 0.99
    assert events["A"]["engine_input_wait"] == 0.02


def test_request_traces_t_sched_and_censor():
    traces = request_traces(STEPS)
    assert traces["A"]["t_sched"] == 1.00
    assert traces["B"]["t_sched"] == 1.02
    assert traces["A"]["censored"] is True  # A appears in final step
    assert traces["B"]["censored"] is True  # B appears in final step


def test_request_traces_first_last_idx_and_oracle_output_len():
    traces = request_traces(STEPS)
    # A: steps 0,1,2 -> first=0, last=2; decode (computed>=prompt_len) at steps 1,2 -> 2
    assert traces["A"]["first_step_idx"] == 0
    assert traces["A"]["last_step_idx"] == 2
    assert traces["A"]["oracle_output_len"] == 2
    # B: steps 1,2 -> first=1, last=2; decode only at step 2 -> 1
    assert traces["B"]["first_step_idx"] == 1
    assert traces["B"]["last_step_idx"] == 2
    assert traces["B"]["oracle_output_len"] == 1


def test_build_context_for_waiting_request_B_at_step0():
    traces = request_traces(STEPS)
    enq_b = {"t_enq": 1.0, "prompt_len": 4}
    expected_t_iter = predict_step(STEPS[0]["reqs"], COEFFS)
    assert abs(expected_t_iter - 0.016231522791139326) < 1e-15  # exact hand-checked value

    ctx = build_context(STEPS[0], 0, "B", enq_b, META, COEFFS, traces,
                         use_oracle=True, nout_est=5.0)

    assert ctx["batch_size"] == 1                 # only A is a total occupied slot at step 0
    assert ctx["max_batch_size"] == 256
    assert ctx["free_kv_blocks"] == 100            # direct field, no reconstruction needed
    assert "free_kv_reconstructed" not in ctx
    assert ctx["req_kv_need"] == math.ceil(4 / 16) == 1
    assert abs(ctx["t_iter"] - expected_t_iter) < 1e-15
    assert ctx["queue_depth"] == 0                 # B is at index 0 of waiting_ids=["B"]
    assert "queue_pos_from_count" not in ctx
    assert ctx["remaining_steps_est"] == 5.0
    assert ctx["running"] == [{
        "id": "A", "prompt_len": 4, "computed": 0, "scheduled_tokens": 4,
        "steps_done": 0, "kv_blocks": 1, "true_remaining": 3,
        "remaining_est": 6.0,
    }]
    assert ctx["current_iter_elapsed"] == 0.0
    assert ctx["current_iter_remaining"] == expected_t_iter


def test_build_context_use_oracle_false_sets_minus_one():
    traces = request_traces(STEPS)
    enq_b = {"t_enq": 1.0, "prompt_len": 4}
    ctx = build_context(STEPS[0], 0, "B", enq_b, META, COEFFS, traces,
                         use_oracle=False, nout_est=5.0)
    assert ctx["running"] == [{
        "id": "A", "prompt_len": 4, "computed": 0, "scheduled_tokens": 4,
        "steps_done": 0, "kv_blocks": 1, "true_remaining": -1,
        "remaining_est": 6.0,
    }]
    assert ctx["remaining_steps_est"] == 5.0       # unaffected by use_oracle


def test_build_context_does_not_subtract_decode_progress_twice():
    step = {
        "step": 0, "t_start": 1.0, "t_end": 1.01,
        "reqs": [{"id": "A", "kappa": 1, "computed": 404, "prompt_len": 4}],
        "running_reqs": [{"id": "A", "prompt_len": 4, "computed": 404,
                          "scheduled_tokens": 1, "kv_blocks_est": 26}],
        "waiting_count": 0, "waiting_ids": [], "waiting_reqs": [],
        "waiting_work": {"count": 0, "remaining_prefill_tokens": 0},
        "free_kv_blocks": 100,
    }
    ctx = build_context(
        step, 0, "target", {"t_enq": 1.0, "prompt_len": 4},
        META, COEFFS, {}, use_oracle=False, nout_est=112.0,
        output_len_est=512.0)
    assert ctx["running"][0]["steps_done"] == 400
    assert ctx["running"][0]["remaining_est"] == 112.0
    assert ctx["output_steps_est"] == 512.0


def test_build_context_running_has_one_entry_per_total_slot_at_step1():
    # step 1: reqs = [A (decode, computed=4), B (prefill, computed=0)] -> 2 total slots
    traces = request_traces(STEPS)
    enq_b = {"t_enq": 1.0, "prompt_len": 4}
    ctx = build_context(STEPS[1], 1, "B", enq_b, META, COEFFS, traces,
                         use_oracle=True, nout_est=5.0)
    assert ctx["batch_size"] == 2
    assert ctx["running"] == [
        {"id": "A", "prompt_len": 4, "computed": 4, "scheduled_tokens": 1,
         "steps_done": 0, "kv_blocks": 1, "true_remaining": 2,
         "remaining_est": 5.0},
        {"id": "B", "prompt_len": 4, "computed": 0, "scheduled_tokens": 4,
         "steps_done": 0, "kv_blocks": 1, "true_remaining": 2,
         "remaining_est": 6.0},
    ]
    # B is no longer in a waiting_ids list at step 1 (empty) and waiting_truncated is False
    # -> fallback branch -> queue_depth = step 1's waiting_count (0 in this fixture),
    # flagged as count-derived same as the truncated branch.
    assert ctx["queue_depth"] == STEPS[1]["waiting_count"] == 0
    assert ctx["queue_pos_from_count"] is True


def test_build_context_free_kv_blocks_none_reconstructs_and_flags():
    traces = request_traces(STEPS)
    step = dict(STEPS[1])
    step["free_kv_blocks"] = None
    enq_b = {"t_enq": 1.0, "prompt_len": 4}
    ctx = build_context(step, 1, "B", enq_b, META, COEFFS, traces,
                         use_oracle=True, nout_est=5.0)
    # Free-KV is sampled after scheduling, so include this step's grants:
    # ceil((4+1)/16) + ceil((0+4)/16) = 2 blocks.
    assert ctx["free_kv_blocks"] == 998
    assert ctx["free_kv_reconstructed"] is True


def test_build_context_queue_pos_from_count_when_truncated():
    traces = request_traces(STEPS)
    step = dict(STEPS[0])
    step["waiting_ids"] = []            # "C" was truncated out of the capped list
    step["waiting_truncated"] = True
    step["waiting_count"] = 37
    enq_c = {"t_enq": 1.0, "prompt_len": 4}
    traces_with_c = dict(traces)
    traces_with_c["C"] = {"first_step_idx": 0, "last_step_idx": 2, "t_sched": 1.0,
                           "oracle_output_len": 0, "censored": True}
    ctx = build_context(step, 0, "C", enq_c, META, COEFFS, traces_with_c,
                         use_oracle=True, nout_est=5.0)
    assert ctx["queue_depth"] == 37
    assert ctx["queue_pos_from_count"] is True


def test_build_context_not_waiting_and_not_truncated_uses_waiting_count():
    # Request "A" itself: not in waiting_ids (["B"]), waiting_truncated False ->
    # fallback to step 0's observed waiting_count (1 in this fixture) as the
    # back-of-queue proxy, same as the truncated branch.
    traces = request_traces(STEPS)
    enq_a = {"t_enq": 1.0, "prompt_len": 4}
    ctx = build_context(STEPS[0], 0, "A", enq_a, META, COEFFS, traces,
                         use_oracle=True, nout_est=5.0)
    assert ctx["queue_depth"] == STEPS[0]["waiting_count"] == 1
    assert ctx["queue_pos_from_count"] is True


def test_build_context_not_in_snapshot_regression_uses_waiting_count():
    # Regression lock for Fix 2: a step with waiting_count=5, not truncated,
    # and a waiting_ids list that does NOT contain the request being
    # contextualized. The request's true position is unobserved, so it must
    # inherit the step's observed backlog (5), not the old front-of-queue 0.
    traces = request_traces(STEPS)
    step = dict(STEPS[0])
    step["waiting_ids"] = ["Q"]       # some other request occupies the snapshot
    step["waiting_count"] = 5
    step["waiting_truncated"] = False
    enq_a = {"t_enq": 1.0, "prompt_len": 4}
    ctx = build_context(step, 0, "A", enq_a, META, COEFFS, traces,
                         use_oracle=True, nout_est=5.0)
    assert ctx["queue_depth"] == 5
    assert ctx["queue_pos_from_count"] is True


def test_build_context_uses_upstream_arrival_phase_and_queue_work():
    traces = request_traces(STEPS)
    step = dict(STEPS[0])
    step["waiting_reqs"] = [{"id": "Q", "prompt_len": 100, "computed": 20,
                              "scheduled_tokens": 0, "cached_tokens": 16,
                              "remaining_prefill_tokens": 80, "kv_blocks_est": 2}]
    step["waiting_work"] = {"count": 1, "remaining_prefill_tokens": 80}
    enq = {"t_engine_arrive": 1.005, "t_enq": 1.019, "prompt_len": 4}
    c = build_context(step, 0, "B", enq, {**META, "max_num_batched_tokens": 8192},
                      COEFFS, traces, use_oracle=False, nout_est=5.0)
    assert c["current_iter_elapsed"] == pytest.approx(0.005)
    assert c["current_iter_remaining"] == pytest.approx(max(c["t_iter"] - 0.005, 0))
    assert c["waiting"][0]["remaining_prefill_tokens"] == 80
    assert c["waiting_work"]["remaining_prefill_tokens"] == 80


def test_build_context_adds_same_iteration_input_arrivals_ahead_of_target():
    traces = request_traces(STEPS)
    step = dict(STEPS[0])
    step["running_reqs"] = [{
        "id": "A", "prompt_len": 4, "computed": 0,
        "scheduled_tokens": 4, "kv_blocks_est": 1,
    }]
    step["waiting_reqs"] = [{
        "id": "B", "prompt_len": 4, "computed": 0,
        "scheduled_tokens": 0, "cached_tokens": 0,
        "remaining_prefill_tokens": 4, "kv_blocks_est": 0,
    }]
    step["waiting_work"] = {"count": 1, "prompt_tokens": 4,
                             "computed_tokens": 0,
                             "cached_tokens_observed": 0,
                             "cached_tokens_missing": 0,
                             "remaining_prefill_tokens": 4,
                             "full_prompt_kv_blocks": 1}
    events = {
        "B": {"t_engine_arrive": 0.99, "t_enq": 1.0, "prompt_len": 4},
        "X": {"t_engine_arrive": 1.006, "t_enq": 1.02, "prompt_len": 10},
        "C": {"t_engine_arrive": 1.010, "t_enq": 1.02, "prompt_len": 6},
    }
    c = build_context(
        step, 0, "C", events["C"],
        {**META, "max_num_batched_tokens": 8}, COEFFS, traces,
        use_oracle=False, nout_est=5.0, enq_events=events)
    assert c["queue_depth"] == 2  # captured B plus pending-input X
    assert c["pending_input_count"] == 1
    assert [r["id"] for r in c["waiting"]] == ["B", "X"]
    assert c["waiting_work"]["count"] == 2
    assert c["waiting_work"]["remaining_prefill_tokens"] == 14


def test_block_accounting_diag_exact():
    diag = block_accounting_diag(STEPS[1], META)
    # reconstructed after scheduling = 1000 - 2 allocated blocks.
    assert diag["captured_free"] == 90
    assert diag["reconstructed_free"] == 998
    assert diag["delta"] == 90 - 998 == -908


def test_block_accounting_diag_none_safe_when_captured_missing():
    step = dict(STEPS[1])
    step["free_kv_blocks"] = None
    diag = block_accounting_diag(step, META)
    assert diag["captured_free"] is None
    assert diag["reconstructed_free"] == 998
    assert diag["delta"] is None


# ---------------------------------------------------------------------------
# Task 5: teacher-forced replay + report
# ---------------------------------------------------------------------------

STEPS_PARSED = parse_admission_trajectory(_write_jsonl(STEPS))
# STEPS_PARSED brackets (from t_start / t_end_next):
#   step 0: [1.00, 1.02)
#   step 1: [1.02, 1.04)


def test_enqueue_bucket_drops_before_first_start():
    enq = {
        "P0": {"t_enq": 0.99, "prompt_len": 4},
        "P1": {"t_enq": 1.005, "prompt_len": 4},
        "P2": {"t_enq": 1.03, "prompt_len": 4},
    }
    bucket, dropped = enqueue_bucket(STEPS_PARSED, enq)
    assert dropped == 1
    assert "P0" not in bucket
    assert bucket["P1"]["step"] == 0
    assert bucket["P2"]["step"] == 1


def test_enqueue_bucket_drops_at_or_after_last_edge():
    enq = {"P3": {"t_enq": 1.04, "prompt_len": 4}}  # == last bracket edge -> dropped
    bucket, dropped = enqueue_bucket(STEPS_PARSED, enq)
    assert dropped == 1
    assert "P3" not in bucket


def test_enqueue_bucket_prefers_engine_arrival_over_late_scheduler_enqueue():
    enq = {"P": {"t_engine_arrive": 1.005, "t_enq": 1.03, "prompt_len": 4}}
    bucket, dropped = enqueue_bucket(STEPS_PARSED, enq)
    assert dropped == 0
    assert bucket["P"]["step"] == 0


def test_running_mean_empty_is_one():
    rm = RunningMean()
    assert rm.value() == 1.0


def test_running_mean_average_above_floor():
    rm = RunningMean()
    rm.add(4)
    rm.add(6)
    assert rm.value() == 5.0


def test_running_mean_floors_at_one():
    rm = RunningMean()
    rm.add(0)
    assert rm.value() == 1.0


def test_deployable_rem_steps_est_two_decode_occupants():
    # computed - prompt_len = 2 and 4; nhat_out_mean = 3
    step = {"reqs": [
        {"id": "X", "computed": 6, "prompt_len": 4},
        {"id": "Y", "computed": 8, "prompt_len": 4},
    ]}
    # max_steps = 4; nhat_eff = max(3, 4) = 4
    # per-occupant: max(4-2,1)=2, max(4-4,1)=1 -> mean = 1.5
    assert abs(deployable_rem_steps_est(step, 3.0) - 1.5) < 1e-9


def test_deployable_rem_steps_est_no_decode_returns_one():
    step = {"reqs": [{"id": "X", "computed": 0, "prompt_len": 4}]}
    assert deployable_rem_steps_est(step, 3.0) == 1.0


def _write_jsonl_ext(rows):
    return _write_jsonl(rows)


def test_replay_oracle_rollforward_free_slot_gets_floored_t_iter():
    # Extend the fixture with a 4th raw step so B departs (last appears at
    # index 2) and is NOT censored, while A is still running at the end.
    step3 = {"step": 3, "t_start": 1.06, "t_end": 1.07, "total_scheduled": 1, "num_running": 1,
             "reqs": [{"id": "A", "kappa": 1, "computed": 6, "prompt_len": 4}],
             "waiting_count": 0, "waiting_ids": [], "waiting_truncated": False, "free_kv_blocks": 70}
    steps_ext_raw = STEPS + [step3]
    traces_ext = request_traces(steps_ext_raw)
    assert traces_ext["B"]["censored"] is False  # departed before the trace ends
    assert traces_ext["A"]["censored"] is True   # still running at the end

    steps_parsed = parse_admission_trajectory(_write_jsonl_ext(steps_ext_raw))
    # steps_parsed == [step0, step1, step2] (step3 dropped, no successor)

    enq = {"B": {"t_enq": 1.01, "prompt_len": 4}}  # buckets to step 0
    rows, never_scheduled = replay(steps_parsed, enq, traces_ext, META, COEFFS,
                                    estimator_name="rollforward", use_oracle=True,
                                    load_of=lambda rid: "sub_capacity")

    assert never_scheduled == 0
    assert len(rows) == 1
    row = rows[0]
    assert row["req_id"] == "B"
    assert abs(row["realized"] - (1.02 - 1.01)) < 1e-12  # t_sched(1.02) - t_enq
    # slot+KV fit at step 0 -> floored(0) -> ctx["t_iter"], i.e. the estimator's
    # OWN predict_step(reqs, coeffs) value for step 0's reqs (NOT the raw
    # captured t_iter=0.02 delta -- build_context always recomputes via
    # predict_step; this fixture's coeffs don't reproduce that raw delta).
    expected_t_iter = predict_step(steps_parsed[0]["reqs"], COEFFS)
    assert abs(expected_t_iter - 0.016231522791139326) < 1e-15  # exact hand-checked value
    assert abs(row["predicted"] - expected_t_iter) < 1e-15
    assert row["load_bin"] == "sub_capacity"
    assert row["regime_at_enq"] == "pure_prefill"  # step0 has only A prefilling


def test_replay_never_scheduled_request_is_skipped_not_raised():
    # "Z" is enqueued (so it has an enq_events entry and buckets into a
    # step's bracket) but NEVER appears in any step's `reqs` -- it sits in
    # waiting_ids for the whole capture window and is never scheduled. This
    # is the defining symptom of the overload/backlog regime; traces has no
    # entry for Z (request_traces only visits requests that appear in
    # step["reqs"]), so replay must not raise KeyError and must exclude Z
    # from rows while counting it as never-scheduled.
    # Reuse the 4-step extension (B departs at step 2, not censored) from the
    # test above, and additionally park "Z" in waiting_ids at steps 0-1
    # without ever putting it in any step's `reqs`.
    step0 = dict(STEPS[0]); step0["waiting_ids"] = ["Z"]; step0["waiting_count"] = 1
    step1 = dict(STEPS[1]); step1["waiting_ids"] = ["Z", *STEPS[1]["waiting_ids"]]
    step1["waiting_count"] = len(step1["waiting_ids"])
    step3 = {"step": 3, "t_start": 1.06, "t_end": 1.07, "total_scheduled": 1, "num_running": 1,
             "reqs": [{"id": "A", "kappa": 1, "computed": 6, "prompt_len": 4}],
             "waiting_count": 1, "waiting_ids": ["Z"], "waiting_truncated": False, "free_kv_blocks": 70}
    steps_raw = [step0, step1, STEPS[2], step3]
    traces = request_traces(steps_raw)
    assert "Z" not in traces  # never scheduled -> never visited by request_traces
    assert traces["B"]["censored"] is False  # B still departs before capture ends

    steps_parsed = parse_admission_trajectory(_write_jsonl(steps_raw))
    enq = {
        "B": {"t_enq": 1.01, "prompt_len": 4},   # scheduled, buckets to step 0
        "Z": {"t_enq": 1.005, "prompt_len": 4},  # never scheduled, buckets to step 0
    }

    rows, never_scheduled = replay(steps_parsed, enq, traces, META, COEFFS,
                                    estimator_name="rollforward", use_oracle=True,
                                    load_of=lambda rid: "sub_capacity")

    assert never_scheduled == 1
    assert {r["req_id"] for r in rows} == {"B"}  # Z excluded, no KeyError raised

    censored = admission_censored_rows(
        steps_parsed, enq, traces, load_of=lambda rid: "sub_capacity")
    assert [r["req_id"] for r in censored] == ["Z"]
    assert censored[0]["lower_bound"] == pytest.approx(1.06 - 1.005)


def test_replay_can_predict_never_scheduled_without_treating_bound_as_realized():
    step0 = dict(STEPS[0]); step0["waiting_ids"] = ["Z"]; step0["waiting_count"] = 1
    step3 = {"step": 3, "t_start": 1.06, "t_end": 1.07,
             "total_scheduled": 1, "num_running": 1,
             "reqs": [{"id": "A", "kappa": 1, "computed": 6, "prompt_len": 4}],
             "waiting_count": 1, "waiting_ids": ["Z"],
             "waiting_truncated": False, "free_kv_blocks": 70}
    steps_raw = [step0, STEPS[1], STEPS[2], step3]
    steps_parsed = parse_admission_trajectory(_write_jsonl(steps_raw))
    traces = request_traces(steps_raw)
    enq = {
        "B": {"t_enq": 1.01, "prompt_len": 4},
        "Z": {"t_engine_arrive": 1.005, "t_enq": 1.02, "prompt_len": 4},
    }
    rows, never_scheduled = replay(
        steps_parsed, enq, traces, META, COEFFS, "rollforward", False,
        load_of=lambda rid: "sub_capacity", include_censored=True)
    assert never_scheduled == 1
    z = next(r for r in rows if r["req_id"] == "Z")
    assert z["right_censored"] is True
    assert z["realized"] is None
    assert z["ratio"] is None
    assert z["lower_bound"] == pytest.approx(1.06 - 1.005)
    assert z["known_underprediction_lower_bound"] >= 0


def test_admission_report_buckets_into_sub_capacity_and_overload():
    row_sub = {"req_id": "r1", "predicted": 2.0, "realized": 4.0, "ratio": 2.0,
               "load_bin": "sub_capacity", "regime_at_enq": "pure_decode"}
    row_over = {"req_id": "r2", "predicted": 10.0, "realized": 5.0, "ratio": 0.5,
                "load_bin": "overload_r5", "regime_at_enq": "mixed"}
    rows_by_key = {("rollforward", "deployable"): [row_sub, row_over]}
    report = admission_report(rows_by_key)

    sub_stats = report["sub_capacity"]["rollforward"]["deployable"]["sub_capacity"]
    assert sub_stats["n"] == 1
    assert sub_stats["median_ratio"] == 2.0
    assert abs(sub_stats["mape"] - 50.0) < 1e-9  # |2-4|/4*100
    assert abs(sub_stats["bias_pct"] - 100.0) < 1e-9  # (2-1)*100

    over_stats = report["overload"]["rollforward"]["deployable"]["overload_r5"]
    assert over_stats["n"] == 1
    assert over_stats["median_ratio"] == 0.5
    assert abs(over_stats["mape"] - 100.0) < 1e-9  # |10-5|/5*100
    assert abs(over_stats["bias_pct"] - (-50.0)) < 1e-9  # (0.5-1)*100


def test_admission_report_includes_block_accounting_default():
    report = admission_report({})
    assert report["block_accounting"] == {"n": 0, "mean_abs_delta": None, "max_abs_delta": None}
    assert report["right_censored"]["n"] == 0


def test_admission_report_keeps_censored_bounds_out_of_mape():
    row = {"req_id": "r1", "predicted": 2.0, "realized": 4.0, "ratio": 2.0,
           "load_bin": "sub_capacity", "regime_at_enq": "pure_decode"}
    censored = [
        {"req_id": "z1", "lower_bound": 5.0, "load_bin": "overload",
         "predicted": 4.0, "prediction_below_lower_bound": True,
         "known_underprediction_lower_bound": 1.0},
        {"req_id": "z2", "lower_bound": 9.0, "load_bin": "overload",
         "predicted": 10.0, "prediction_below_lower_bound": False,
         "known_underprediction_lower_bound": 0.0},
    ]
    report = admission_report(
        {("token_rollforward", "deployable"): [row]}, censored_rows=censored)
    stats = report["sub_capacity"]["token_rollforward"]["deployable"]["sub_capacity"]
    assert stats["mape"] == pytest.approx(50.0)
    assert report["right_censored"]["n"] == 2
    assert report["right_censored"]["lower_bound_s_p50"] == pytest.approx(7.0)
    assert report["right_censored"]["by_load_bin"] == {"overload": 2}
    assert report["right_censored"]["prediction_below_lower_bound_frac"] == 0.5


def test_admission_report_block_accounting_from_diag_rows():
    diag_rows = [{"captured_free": 90, "reconstructed_free": 998, "delta": -908},
                 {"captured_free": None, "reconstructed_free": 5, "delta": None},
                 {"captured_free": 10, "reconstructed_free": 8, "delta": 2}]
    report = admission_report({}, diag_rows=diag_rows)
    assert report["block_accounting"]["n"] == 2
    assert abs(report["block_accounting"]["mean_abs_delta"] - (908 + 2) / 2) < 1e-9
    assert report["block_accounting"]["max_abs_delta"] == 908


def test_write_admission_report_writes_json_and_png(tmp_path=None):
    out_dir = tempfile.mkdtemp()
    row = {"req_id": "r1", "predicted": 0.02, "realized": 0.03, "ratio": 1.5,
           "load_bin": "sub_capacity", "regime_at_enq": "pure_decode"}
    rows_by_key = {("rollforward", "oracle"): [row]}
    report = admission_report(rows_by_key)
    paths = write_admission_report(report, rows_by_key, out_dir)

    json_path = os.path.join(out_dir, "admission_report.json")
    assert os.path.exists(json_path)
    with open(json_path) as f:
        written = json.load(f)
    assert written["sub_capacity"]["rollforward"]["oracle"]["sub_capacity"]["n"] == 1
    assert paths["json_path"] == json_path
    # matplotlib is present in this environment -> png should also be written
    assert paths["png_path"] is not None
    assert os.path.exists(paths["png_path"])


def test_write_admission_report_json_only_when_matplotlib_absent():
    out_dir = tempfile.mkdtemp()
    row = {"req_id": "r1", "predicted": 0.02, "realized": 0.03, "ratio": 1.5,
           "load_bin": "sub_capacity", "regime_at_enq": "pure_decode"}
    rows_by_key = {("rollforward", "oracle"): [row]}
    report = admission_report(rows_by_key)

    saved = {}
    for name in list(sys.modules):
        if name == "matplotlib" or name.startswith("matplotlib."):
            saved[name] = sys.modules.pop(name)
    sys.modules["matplotlib"] = None  # forces ImportError on `import matplotlib`
    try:
        paths = write_admission_report(report, rows_by_key, out_dir)
    finally:
        del sys.modules["matplotlib"]
        sys.modules.update(saved)

    assert os.path.exists(os.path.join(out_dir, "admission_report.json"))
    assert paths["png_path"] is None
