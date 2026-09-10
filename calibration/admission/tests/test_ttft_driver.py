"""Tests for the TTFT reconstruction driver.

The driver's real input is a multi-gigabyte capture that lives only on the
cluster PVC, so its correctness is pinned here on a synthetic trajectory small
enough that every expected value is computed by hand in the comments below.
Two properties matter most and are asserted directly:

  1. compose_segment reproduces the (E4) composition term for term.
  2. bisect_enqueue_bucket agrees with the frozen analysis.enqueue_bucket it
     replaces for speed.
"""
import json
import math

import pytest

from calibration.admission import ttft_driver as D

D.ensure_matplotlib_importable()
from calibration.admission import analysis as A  # noqa: E402


# Round coefficients chosen so the hand arithmetic below is exact in binary.
COEFFS = {"c_base": 1.0, "c_dec": 0.1, "c_kv": 0.01,
          "c_pf": 0.001, "c_attn": 0.0001}

# kappa is deliberately small so a 150-token prompt spans two chunks.
META = {"max_num_batched_tokens": 100, "block_size": 16,
        "num_gpu_blocks": 1000, "max_num_seqs": 4}

KAPPA = META["max_num_batched_tokens"]


def _step(step, t_start, reqs, waiting=0, free_kv=1000):
    return {"step": step, "t_start": t_start, "t_end": t_start + 0.1,
            "total_scheduled": len(reqs), "num_running": len(reqs), "reqs": reqs,
            "waiting_count": waiting, "waiting_ids": [], "waiting_truncated": False,
            "free_kv_blocks": free_kv}


def _z(computed):
    """Filler decode occupant, prompt fully computed."""
    return {"id": "Z", "kappa": 1, "computed": computed, "prompt_len": 5}


def _a(computed, kappa):
    return {"id": "A", "kappa": kappa, "computed": computed, "prompt_len": 150}


@pytest.fixture
def fixture_segment():
    """Request A arrives mid-step-0, prefills over steps 1-2, first token at step 3.

    Timeline, one second per step:
        step 0  t=0  [Z decode]                      A enqueued at t=0.5
        step 1  t=1  [A prefill computed=0   k=100]  t_sched = 1.0
        step 2  t=2  [A prefill computed=100 k=50 ]
        step 3  t=3  [A decode  computed=150      ]  t_first = 3.0
        step 4  t=4  [A decode  computed=151      ]
        step 5  t=5  [Z decode]                      A has departed
        step 6  t=6  [Z decode]                      dropped by add_step_deltas

    add_step_deltas keeps steps 0-5 and gives every one t_iter = 1.0, so A's
    last_step_idx is 4 against n-1 = 5 and A is therefore uncensored, which is
    what makes it eligible for the deployable replay.
    """
    steps = [
        _step(0, 0.0, [_z(5)]),
        _step(1, 1.0, [_a(0, 100), _z(6)]),
        _step(2, 2.0, [_a(100, 50), _z(7)]),
        _step(3, 3.0, [_a(150, 1), _z(8)]),
        _step(4, 4.0, [_a(151, 1), _z(9)]),
        _step(5, 5.0, [_z(10)]),
        _step(6, 6.0, [_z(11)]),
    ]
    events = {"A": {"t_enq": 0.5, "prompt_len": 150}}
    return steps, events


# ---------------------------------------------------------------------------
# (E2) and (E3) closed forms.
# ---------------------------------------------------------------------------
def test_n_chunks_boundaries():
    assert D.n_chunks(1, 100) == 1
    assert D.n_chunks(100, 100) == 1          # exact fit stays one chunk
    assert D.n_chunks(101, 100) == 2
    assert D.n_chunks(150, 100) == 2
    assert D.n_chunks(8192, 8192) == 1
    assert D.n_chunks(16000, 8192) == 2


def test_n_chunks_rejects_nonsense():
    with pytest.raises(ValueError):
        D.n_chunks(0, 100)
    with pytest.raises(ValueError):
        D.n_chunks(150, 0)


def test_chunk_tokens_partition_the_prompt():
    assert D.chunk_tokens(150, 100) == [100, 50]
    assert D.chunk_tokens(100, 100) == [100]
    assert D.chunk_tokens(16000, 8192) == [8192, 7808]
    for plen in (1, 99, 100, 101, 8191, 8192, 8193, 16000):
        assert sum(D.chunk_tokens(plen, 8192)) == plen


def test_prefill_work_matches_hand_computation():
    # prompt_len 150, kappa 100 -> chunks (t_0=100, P_0=0), (t_1=50, P_1=100)
    #   chunk 0: c_pf*100 + c_attn*100*(0   + 50)  = 0.1  + 0.5   = 0.6
    #   chunk 1: c_pf*50  + c_attn*50 *(100 + 25)  = 0.05 + 0.625 = 0.675
    #   W_p = 1.275
    assert D.prefill_work(150, KAPPA, COEFFS) == pytest.approx(1.275)


def test_prefill_work_single_chunk_is_the_k_zero_term():
    # One chunk: t_0 = prompt_len, P_0 = 0, so W_p = c_pf*n + c_attn*n*n/2.
    n = 80
    expected = COEFFS["c_pf"] * n + COEFFS["c_attn"] * n * (n / 2.0)
    assert D.prefill_work(n, KAPPA, COEFFS) == pytest.approx(expected)


def test_prefill_work_grows_superlinearly_in_prompt_length():
    """The c_attn term is quadratic, so doubling the prompt more than doubles W_p."""
    w1 = D.prefill_work(4000, 8192, COEFFS)
    w2 = D.prefill_work(8000, 8192, COEFFS)
    assert w2 > 2 * w1


# ---------------------------------------------------------------------------
# Process-restart segmentation.
# ---------------------------------------------------------------------------
def test_split_on_reset_detects_step_counter_reset():
    recs = [{"step": 0, "t_start": 100.0}, {"step": 1, "t_start": 101.0},
            {"step": 0, "t_start": 5.0}, {"step": 1, "t_start": 6.0}]
    segs = D.split_on_reset(recs, "t_start", step_key="step")
    assert [len(s) for s in segs] == [2, 2]
    assert segs[1][0]["t_start"] == 5.0


def test_split_on_reset_detects_clock_reset_without_step_key():
    recs = [{"t_enq": 100.0}, {"t_enq": 101.0}, {"t_enq": 3.0}, {"t_enq": 4.0}]
    segs = D.split_on_reset(recs, "t_enq")
    assert [len(s) for s in segs] == [2, 2]


def test_split_on_reset_tolerates_small_backward_jitter():
    """Writes can land slightly out of order; that is not a restart."""
    recs = [{"t_enq": 100.0}, {"t_enq": 99.9}, {"t_enq": 100.5}]
    assert len(D.split_on_reset(recs, "t_enq", backward_tolerance=1.0)) == 1


def test_split_on_reset_handles_three_segments():
    recs = [{"step": 0, "t_start": 50.0}, {"step": 0, "t_start": 10.0},
            {"step": 0, "t_start": 2.0}]
    assert len(D.split_on_reset(recs, "t_start", step_key="step")) == 3


def test_split_on_reset_single_segment_when_monotone():
    recs = [{"step": i, "t_start": float(i)} for i in range(20)]
    assert len(D.split_on_reset(recs, "t_start", step_key="step")) == 1


def test_events_by_id_preserves_schema_v2_arrival_fields():
    events = D.events_by_id([{
        "req_id": "A", "t_engine_arrive": 0.25, "t_enq": 0.5,
        "engine_input_wait": 0.25, "prompt_len": 150,
    }])
    assert events["A"]["t_engine_arrive"] == pytest.approx(0.25)
    assert events["A"]["engine_input_wait"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# add_step_deltas parity with the frozen parser.
# ---------------------------------------------------------------------------
def test_add_step_deltas_matches_frozen_parser(fixture_segment, tmp_path):
    steps, _ = fixture_segment
    path = tmp_path / "trajectory.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in steps))
    assert D.add_step_deltas(steps) == A.parse_admission_trajectory(str(path))


def test_add_step_deltas_drops_the_final_step(fixture_segment):
    steps, _ = fixture_segment
    out = D.add_step_deltas(steps)
    assert len(out) == len(steps) - 1
    assert all(s["t_iter"] == pytest.approx(1.0) for s in out)
    assert out[0]["t_end_next"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The bisect bucket must match the frozen linear-scan bucket.
# ---------------------------------------------------------------------------
def test_bisect_bucket_matches_frozen_bucket(fixture_segment):
    steps, events = fixture_segment
    steps = D.add_step_deltas(steps)
    assert D.bisect_enqueue_bucket(steps, events) == A.enqueue_bucket(steps, events)


def test_bisect_bucket_matches_frozen_bucket_over_many_offsets(fixture_segment):
    """Sweep t_enq across and beyond the whole capture window, including the
    exact step boundaries where a half-open bracket is easiest to get wrong."""
    steps, _ = fixture_segment
    steps = D.add_step_deltas(steps)
    offsets = [-1.0, -0.001, 0.0, 0.5, 0.999, 1.0, 2.5, 4.999, 5.0, 5.5,
               5.999, 6.0, 7.0]
    events = {f"r{i}": {"t_enq": t, "prompt_len": 10} for i, t in enumerate(offsets)}
    mine = D.bisect_enqueue_bucket(steps, events)
    theirs = A.enqueue_bucket(steps, events)
    assert mine[1] == theirs[1]                      # same dropped count
    assert set(mine[0]) == set(theirs[0])
    for rid in mine[0]:
        assert mine[0][rid] is theirs[0][rid]        # same step object identity


def test_bucket_is_half_open_at_the_upper_edge(fixture_segment):
    """t_enq exactly at t_end_next belongs to the NEXT step, not this one."""
    steps, _ = fixture_segment
    steps = D.add_step_deltas(steps)
    bucket, dropped = D.bisect_enqueue_bucket(
        steps, {"x": {"t_enq": 1.0, "prompt_len": 10}})
    assert dropped == 0
    assert bucket["x"]["step"] == 1


def test_bisect_bucket_prefers_upstream_engine_arrival(fixture_segment):
    steps, _ = fixture_segment
    steps = D.add_step_deltas(steps)
    bucket, dropped = D.bisect_enqueue_bucket(steps, {
        "x": {"t_engine_arrive": 0.75, "t_enq": 1.25, "prompt_len": 10},
    })
    assert dropped == 0
    assert bucket["x"]["step"] == 0


def test_bisect_bucket_rejects_a_non_ascending_segment():
    """The frozen linear scan tolerates an unordered step list and the binary
    search cannot, so an unordered segment must fail loudly. add_step_deltas
    orders by step number, and if a capture ever broke the step-to-time
    correspondence the bisect would otherwise return a silently wrong step."""
    steps = [
        {"step": 0, "t_start": 0.0, "t_end_next": 1.0, "reqs": []},
        {"step": 1, "t_start": 1.0, "t_end_next": 0.5, "reqs": []},
        {"step": 2, "t_start": 0.5, "t_end_next": 2.0, "reqs": []},
    ]
    with pytest.raises(ValueError, match="not ascending"):
        D.bisect_enqueue_bucket(steps, {"x": {"t_enq": 0.7, "prompt_len": 10}})


def test_arrival_batch_excludes_the_arriving_request(fixture_segment):
    """A request whose t_enq coincides with the t_start of the step that admits it
    lands in its own bracket. W_p already charges that request's prefill work, so
    letting it into T_iter would double-count its first chunk.

    Setting t_enq to 1.0 puts A in step 1, which is also A's own first step. That
    step holds A mid-prefill plus Z decoding with computed = 6. The resident batch
    is Z alone:
        T_iter = c_base + c_dec*1 + c_kv*6 = 1.0 + 0.1 + 0.06 = 1.16
        compute = 2 * 1.16 + 1.275 = 3.595
    Had A stayed in the batch it would have added its own chunk-0 prefill term,
    c_pf*100 + c_attn*100*50 = 0.6, giving T_iter 1.76 and compute 4.795, a 33%
    inflation of the very term this driver exists to measure.
    """
    steps, _ = fixture_segment
    tied, diag = D.compose_segment(steps, {"A": {"t_enq": 1.0, "prompt_len": 150}},
                                   META, COEFFS, "rollforward", 1)
    assert diag["self_in_arrival_batch"] == 1, "the tie should be detected"
    assert tied[0]["r_tadm"] == pytest.approx(0.0)
    assert tied[0]["compute"] == pytest.approx(3.595)
    assert tied[0]["compute"] != pytest.approx(4.795), "A's own chunk leaked in"


def test_arrival_batch_untied_case_reports_no_self_inclusion(fixture_segment):
    steps, events = fixture_segment
    _, diag = D.compose_segment(steps, events, META, COEFFS, "rollforward", 1)
    assert diag["self_in_arrival_batch"] == 0


def test_bucket_drops_arrivals_outside_the_capture_window(fixture_segment):
    steps, _ = fixture_segment
    steps = D.add_step_deltas(steps)
    events = {"early": {"t_enq": -5.0, "prompt_len": 10},
              "late": {"t_enq": 99.0, "prompt_len": 10}}
    bucket, dropped = D.bisect_enqueue_bucket(steps, events)
    assert bucket == {} and dropped == 2


# ---------------------------------------------------------------------------
# Realized first-token reconstruction (E5).
# ---------------------------------------------------------------------------
def test_first_token_is_the_first_step_with_prefill_complete(fixture_segment):
    steps, _ = fixture_segment
    first_token, skipped = D.first_token_times(D.add_step_deltas(steps))
    assert first_token["A"] == pytest.approx(3.0)
    assert "A" not in skipped


def test_request_already_decoding_at_capture_start_is_skipped():
    """Z is decoding from step 0, so its prefill was never observed."""
    steps = D.add_step_deltas([_step(0, 0.0, [_z(5)]), _step(1, 1.0, [_z(6)]),
                               _step(2, 2.0, [_z(7)])])
    first_token, skipped = D.first_token_times(steps)
    assert "Z" in skipped
    assert "Z" not in first_token


def test_request_still_prefilling_at_capture_end_has_no_first_token():
    steps = D.add_step_deltas([
        _step(0, 0.0, [_a(0, 100)]),
        _step(1, 1.0, [_a(100, 50)]),
        _step(2, 2.0, [_a(100, 50)]),
    ])
    first_token, skipped = D.first_token_times(steps)
    assert first_token == {} and skipped == set()


# ---------------------------------------------------------------------------
# (E4) composition, end to end.
# ---------------------------------------------------------------------------
def test_compose_segment_reproduces_the_composition_by_hand(fixture_segment):
    steps, events = fixture_segment
    rows, diag = D.compose_segment(steps, events, META, COEFFS, "rollforward", 1)

    assert len(rows) == 1, "only A has an enqueue event, so only A is scored"
    row = rows[0]

    # Arrival batch is step 0, holding Z alone with computed = 5 and prefill done:
    #   T_iter = c_base + c_dec*1 + c_kv*5 = 1.0 + 0.1 + 0.05 = 1.15        (E1)
    t_iter = 1.15
    # n_c = ceil(150/100) = 2, W_p = 1.275 (see test_prefill_work above)   (E2,E3)
    compute = 2 * t_iter + 1.275                                          # = 3.575

    assert row["nc"] == 2
    assert row["prompt_len"] == 150
    assert row["compute"] == pytest.approx(compute)

    # Realized, per (E5): t_enq 0.5, t_sched 1.0, t_first 3.0
    assert row["r_tadm"] == pytest.approx(0.5)
    assert row["r_prefill"] == pytest.approx(2.0)
    assert row["r_ttft"] == pytest.approx(2.5)

    # A arrives to a free slot (batch 1 of 4) with KV headroom (1000 blocks free
    # against ceil(150/16) = 10 needed), so rollforward takes the _slot_and_kv_fit
    # early return and floored_t_adm raises 0 to one iteration.
    assert row["t_adm_deploy"] == pytest.approx(t_iter)

    assert row["p_oracle"] == pytest.approx(0.5 + compute)                # 4.075
    assert row["p_deploy"] == pytest.approx(t_iter + compute)             # 4.725

    assert diag["kappa"] == 100
    assert diag["n_rows"] == 1
    assert diag["estimator"] == "rollforward"


def test_oracle_composition_is_exactly_realized_admission_plus_compute(fixture_segment):
    """The identity the offline decomposition relies on: p_oracle - r_tadm is the
    compute term, so the compute term is recoverable from the emitted rows."""
    steps, events = fixture_segment
    rows, _ = D.compose_segment(steps, events, META, COEFFS, "rollforward", 1)
    for row in rows:
        assert row["p_oracle"] - row["r_tadm"] == pytest.approx(row["compute"])
        assert row["p_deploy"] - row["compute"] == pytest.approx(row["t_adm_deploy"])


def test_compose_segment_measures_from_upstream_engine_arrival(fixture_segment):
    steps, events = fixture_segment
    events["A"]["t_engine_arrive"] = 0.25
    events["A"]["engine_input_wait"] = 0.25
    rows, _ = D.compose_segment(steps, events, META, COEFFS, "rollforward", 1)
    assert rows[0]["r_tadm"] == pytest.approx(0.75)
    assert rows[0]["r_ttft"] == pytest.approx(2.75)
    assert rows[0]["engine_input_wait"] == pytest.approx(0.25)


def test_compose_segment_uses_full_scheduler_rollout_for_schema_v2_ttft(
        fixture_segment):
    steps, events = fixture_segment
    # The arrival bucket is step 0. Schema-v2 snapshots describe its already
    # scheduled Z decode and the exact empty queue after scheduling.
    steps[0]["running_reqs"] = [{
        "id": "Z", "prompt_len": 5, "computed": 5,
        "scheduled_tokens": 1, "cached_tokens": 0, "kv_blocks_est": 1,
    }]
    steps[0]["waiting_reqs"] = []
    steps[0]["waiting_work"] = {"count": 0, "remaining_prefill_tokens": 0}
    rows, _ = D.compose_segment(
        steps, events, META, COEFFS, "token_rollforward", 1)
    row = rows[0]
    # Arrival at 0.5 during a predicted 1.15s iteration leaves 0.65s. The
    # target then runs 100- and 50-token chunks costing 1.6s and 1.675s.
    assert row["t_adm_deploy"] == pytest.approx(0.65)
    assert row["p_deploy_rollout"] == pytest.approx(3.925)
    assert row["p_deploy"] == pytest.approx(row["p_deploy_rollout"])
    assert row["p_deploy_composed"] == pytest.approx(4.225)


def test_compose_segment_tags_the_segment_index(fixture_segment):
    steps, events = fixture_segment
    rows, diag = D.compose_segment(steps, events, META, COEFFS, "rollforward", 7)
    assert diag["segment"] == 7
    assert all(r["seg"] == 7 for r in rows)


def test_compose_segment_skips_a_request_with_no_first_token():
    """A departs during the capture without ever completing prefill, so it is
    uncensored and reaches the composition loop, but has no first-token instant."""
    steps = [
        _step(0, 0.0, [_z(5)]),
        _step(1, 1.0, [_a(0, 100), _z(6)]),
        _step(2, 2.0, [_a(100, 50), _z(7)]),
        _step(3, 3.0, [_z(8)]),                 # A gone, prefill never finished
        _step(4, 4.0, [_z(9)]),
        _step(5, 5.0, [_z(10)]),
        _step(6, 6.0, [_z(11)]),
    ]
    events = {"A": {"t_enq": 0.5, "prompt_len": 150}}
    rows, diag = D.compose_segment(steps, events, META, COEFFS, "rollforward", 1)
    assert rows == []
    assert diag["skipped_no_first_token"] == 1
    assert diag["censored_excluded"] == 0


def test_compose_segment_scores_first_token_before_later_departure():
    """A is still running when capture ends, but admission and first token are
    observed and the deployable estimate does not need A's eventual output."""
    steps = [
        _step(0, 0.0, [_z(5)]),
        _step(1, 1.0, [_a(0, 100), _z(6)]),
        _step(2, 2.0, [_a(100, 50), _z(7)]),
        _step(3, 3.0, [_a(150, 1), _z(8)]),     # first token exists at t=3.0
        _step(4, 4.0, [_a(151, 1), _z(9)]),     # still running at capture end
    ]
    events = {"A": {"t_enq": 0.5, "prompt_len": 150}}
    rows, diag = D.compose_segment(steps, events, META, COEFFS, "rollforward", 1)
    assert [row["req_id"] for row in rows] == ["A"]
    assert rows[0]["r_ttft"] == pytest.approx(2.5)
    assert diag["censored_excluded"] == 0
    assert diag["skipped_no_first_token"] == 0


def test_compose_segment_emits_never_scheduled_as_right_censored(fixture_segment):
    steps, _ = fixture_segment
    events = {
        "Q": {"t_engine_arrive": 0.25, "t_enq": 0.5, "prompt_len": 150},
    }
    censored = []
    rows, diag = D.compose_segment(
        steps, events, META, COEFFS, "rollforward", 1,
        censored_out=censored)
    assert rows == []
    assert diag["never_scheduled"] == 1
    assert diag["right_censored"]["n"] == 1
    assert censored == [{
        "seg": 1,
        "req_id": "Q",
        "arrival": 0.25,
        "capture_end": 6.0,
        "r_tadm_lower_bound": 5.75,
        "r_ttft_lower_bound": 5.75,
        "t_adm_deploy": pytest.approx(1.15),
        "compute": pytest.approx(3.575),
        "p_deploy": pytest.approx(4.725),
        "p_deploy_composed": pytest.approx(4.725),
        "p_deploy_rollout": None,
        "prediction_below_lower_bound": True,
        "known_underprediction_lower_bound": pytest.approx(1.025),
        "censoring": "right",
        "reason": "enqueued_but_never_scheduled",
    }]


def test_compose_segment_handles_an_empty_segment():
    rows, diag = D.compose_segment([], {}, META, COEFFS, "rollforward", 1)
    assert rows == [] and diag["n_steps"] == 0


def test_compose_segment_restores_the_frozen_bucket_afterwards(fixture_segment):
    """The bisect swap must not leak into the frozen module for other callers."""
    original = A.enqueue_bucket
    steps, events = fixture_segment
    D.compose_segment(steps, events, META, COEFFS, "rollforward", 1)
    assert A.enqueue_bucket is original


# ---------------------------------------------------------------------------
# Reporting conventions.
# ---------------------------------------------------------------------------
def test_view_stats_ratio_is_realized_over_predicted():
    """A predicted value twice the realized one gives median_ratio 0.5, matching
    the committed pins where median_ratio below 1 means the estimate runs high."""
    stats = D.view_stats([(1.0, 2.0), (2.0, 4.0), (4.0, 8.0)])
    assert stats["median_ratio"] == pytest.approx(0.5)
    assert stats["bias_pct"] == pytest.approx(-50.0)
    assert stats["mape_pct"] == pytest.approx(100.0)
    assert stats["over_pred_frac"] == pytest.approx(1.0)


def test_view_stats_under_prediction_reports_ratio_above_one():
    stats = D.view_stats([(10.0, 1.0)])
    assert stats["median_ratio"] == pytest.approx(10.0)
    assert stats["over_pred_frac"] == pytest.approx(0.0)
    assert stats["mape_pct"] == pytest.approx(90.0)


def test_view_stats_perfect_prediction():
    stats = D.view_stats([(3.0, 3.0), (5.0, 5.0)])
    assert stats["median_ratio"] == pytest.approx(1.0)
    assert stats["bias_pct"] == pytest.approx(0.0)
    assert stats["mape_pct"] == pytest.approx(0.0)


def test_view_stats_excludes_nonpositive_realized():
    stats = D.view_stats([(0.0, 1.0), (-1.0, 1.0), (2.0, 2.0)])
    assert stats["n"] == 1


def test_view_stats_empty():
    stats = D.view_stats([])
    assert stats["n"] == 0 and stats["mape_pct"] is None


def test_view_stats_reports_milliseconds():
    stats = D.view_stats([(0.040, 0.080)])
    assert stats["realized_ms_p50"] == pytest.approx(40.0)
    assert stats["pred_ms_p50"] == pytest.approx(80.0)


def test_summarize_splits_on_realized_admission_delay(fixture_segment):
    rows = [
        {"seg": 1, "nc": 1, "r_tadm": 0.001, "r_ttft": 0.04, "r_prefill": 0.039,
         "compute": 0.04, "t_adm_deploy": 0.02, "p_oracle": 0.041, "p_deploy": 0.06},
        {"seg": 1, "nc": 2, "r_tadm": 10.0, "r_ttft": 10.5, "r_prefill": 0.5,
         "compute": 0.5, "t_adm_deploy": 0.55, "p_oracle": 10.5, "p_deploy": 1.05},
    ]
    diag = {"kappa": 8192, "estimator": "rollforward",
            "right_censored": {"n": 3, "lower_bound_s_p50": 2.0,
                               "lower_bound_s_p90": 5.0,
                               "note": "lower bounds are excluded from MAPE"}}
    rep = D.summarize(rows, diag)
    assert rep["n_c_histogram"] == {"1": 1, "2": 1}
    assert rep["ttft_deployable.not_queued"]["n"] == 1
    assert rep["ttft_deployable.queued_gt500ms"]["n"] == 1
    # The queued row's oracle prediction is exact by construction.
    assert rep["ttft_oracle_tadm"]["n"] == 2
    # prefill_only ignores the admission term, so the queued row scores well there.
    assert rep["prefill_only.queued_gt500ms"]["mape_pct"] == pytest.approx(0.0)
    assert rep["right_censored"]["n"] == 3


def test_summarize_view_keys_match_the_committed_pins():
    """Guard against renaming a view the pins and the README refer to."""
    rep = D.summarize([], {"kappa": 8192, "estimator": "rollforward"})
    for key in ("kappa", "n_c_histogram", "prefill_only",
                "prefill_only.not_queued", "prefill_only.queued_gt500ms",
                "ttft_oracle_tadm", "ttft_deployable",
                "ttft_deployable.not_queued", "ttft_deployable.queued_gt500ms"):
        assert key in rep, key


# ---------------------------------------------------------------------------
# The matplotlib placeholder must not shadow a real installation.
# ---------------------------------------------------------------------------
def test_ensure_matplotlib_importable_is_idempotent():
    D.ensure_matplotlib_importable()
    assert D.ensure_matplotlib_importable() is False
    import matplotlib  # noqa: F401
