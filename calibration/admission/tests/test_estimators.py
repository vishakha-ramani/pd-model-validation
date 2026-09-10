import math
from calibration.admission.estimators import (
    floored_t_adm, estimate_fluid, estimate_rollforward,
    estimate_token_rollforward, estimate_token_rollforward_times,
    estimate_waiting,
)


def ctx(**kw):
    base = dict(batch_size=0, max_batch_size=256, free_kv_blocks=0, req_kv_need=0,
                t_iter=0.0, queue_depth=0, remaining_steps_est=0.0, running=[])
    base.update(kw)
    return base


def test_floor_raises_to_t_iter():
    assert floored_t_adm(0.0, ctx(t_iter=0.016)) == 0.016
    assert floored_t_adm(0.05, ctx(t_iter=0.016)) == 0.05


def test_free_slot_and_kv_returns_floor():
    # slot free and KV fits -> floored(0) == t_iter
    c = ctx(batch_size=10, max_batch_size=256, free_kv_blocks=100, req_kv_need=5, t_iter=0.016)
    assert estimate_fluid(c) == 0.016
    assert estimate_rollforward(c) == 0.016


def test_fluid_wave_form():
    # full batch, need to wait: waves = ceil((queue_depth+1)/batch_size)
    c = ctx(batch_size=4, max_batch_size=4, free_kv_blocks=0, req_kv_need=1,
            t_iter=0.02, queue_depth=7, remaining_steps_est=10.0)
    waves = math.ceil((7 + 1) / 4)  # = 2
    assert abs(estimate_fluid(c) - waves * 10.0 * 0.02) < 1e-12


def test_rollforward_departure_lookahead():
    # batch full (4/4), KV full; three running depart at steps 3,5,8 freeing kv 2 each.
    # need_slots = queue_depth+1 = 1; first departure (step 3) frees a slot and 2 kv >= req_kv_need 2.
    c = ctx(batch_size=4, max_batch_size=4, free_kv_blocks=0, req_kv_need=2,
            t_iter=0.02, queue_depth=0, remaining_steps_est=10.0,
            running=[{"steps_done": 0, "kv_blocks": 2, "true_remaining": 8},
                     {"steps_done": 0, "kv_blocks": 2, "true_remaining": 3},
                     {"steps_done": 0, "kv_blocks": 2, "true_remaining": 5}])
    # sorted departures: 3,5,8; first satisfies -> 3 * 0.02
    assert abs(estimate_rollforward(c) - 3 * 0.02) < 1e-12


def test_rollforward_falls_back_to_fluid_when_queue_deep():
    # queue deeper than the running set can drain -> fluid wave fallback
    c = ctx(batch_size=2, max_batch_size=2, free_kv_blocks=0, req_kv_need=1,
            t_iter=0.02, queue_depth=20, remaining_steps_est=10.0,
            running=[{"steps_done": 0, "kv_blocks": 1, "true_remaining": 3},
                     {"steps_done": 0, "kv_blocks": 1, "true_remaining": 4}])
    waves = math.ceil((20 + 1) / 2)  # = 11
    assert abs(estimate_rollforward(c) - waves * 10.0 * 0.02) < 1e-12


def test_rollforward_uses_estimate_when_true_remaining_negative():
    # true_remaining < 0 -> use max(int(remaining_steps_est),1)
    c = ctx(batch_size=1, max_batch_size=1, free_kv_blocks=0, req_kv_need=1,
            t_iter=0.02, queue_depth=0, remaining_steps_est=6.0,
            running=[{"steps_done": 0, "kv_blocks": 1, "true_remaining": -1}])
    assert abs(estimate_rollforward(c) - 6 * 0.02) < 1e-12


def test_waiting_estimator():
    # waiting uses QWork/Mu; ctx here carries them as extra keys
    c = ctx()
    c["qwork"] = 100.0
    c["mu"] = 4.0
    assert estimate_waiting(c) == 25.0
    c["mu"] = 0.0
    assert estimate_waiting(c) == 0.0


ROLLOUT_COEFFS = {"c_base": 0.010, "c_dec": 0.001, "c_kv": 0.0,
                  "c_pf": 0.001, "c_attn": 0.0}


def rollout_ctx(**kw):
    base = ctx(
        batch_size=0, max_batch_size=16, free_kv_blocks=1000,
        req_kv_need=1, t_iter=0.010, queue_depth=0,
        remaining_steps_est=4.0, running=[], waiting=[],
        waiting_work={"remaining_prefill_tokens": 0},
        max_num_batched_tokens=4, block_size=16, coeffs=ROLLOUT_COEFFS,
        current_iter_remaining=0.006, target_id="target", target_prompt_len=1,
        target_cached_tokens=0, queue_work_observed=True,
    )
    base.update(kw)
    return base


def test_token_rollforward_free_path_uses_residual_iteration_not_full_floor():
    assert estimate_token_rollforward(rollout_ctx()) == 0.006


def test_token_rollforward_predicts_first_token_at_end_of_target_step():
    admission, first_token = estimate_token_rollforward_times(rollout_ctx())
    assert admission == 0.006
    # Residual 6ms, then target's 1-token prefill: 10ms base + 1ms token.
    assert first_token == 0.017


def test_token_rollforward_ttft_recomputes_until_all_target_chunks_finish():
    admission, first_token = estimate_token_rollforward_times(
        rollout_ctx(target_prompt_len=8))
    assert admission == 0.006
    # Two target chunks, each 10ms base + 4ms prefill.
    assert first_token == 0.006 + 2 * 0.014


def test_token_rollforward_does_not_skip_nonempty_queue_when_slot_and_kv_fit():
    c = rollout_ctx(
        queue_depth=1,
        waiting=[{"id": "ahead", "prompt_len": 8, "computed": 0,
                  "scheduled_tokens": 0, "kv_blocks_est": 0}],
        waiting_work={"remaining_prefill_tokens": 8},
    )
    # Two 4-token prefill steps run before target admission. Each costs
    # c_base + c_pf*4 = 14 ms. The old estimator incorrectly returned 10 ms.
    assert estimate_token_rollforward(c) == 0.006 + 2 * 0.014
    admission, first_token = estimate_token_rollforward_times(c)
    assert admission == 0.006 + 2 * 0.014
    # Once the target joins, the ahead request also consumes one decode token.
    assert math.isclose(first_token, admission + 0.010 + 0.001 + 0.001)


def test_token_rollforward_running_prefill_consumes_budget_before_waiting():
    c = rollout_ctx(
        running=[{"id": "running", "prompt_len": 12, "computed": 4,
                  "scheduled_tokens": 4, "kv_blocks_est": 1,
                  "true_remaining": -1, "remaining_est": 4}],
        batch_size=1,
    )
    # The in-flight grant advances computed 4->8. At the next boundary the
    # remaining four prompt tokens consume the entire token budget, so target
    # waits one additional 14-ms iteration after the 6-ms residual.
    assert estimate_token_rollforward(c) == 0.006 + 0.014


def test_token_rollforward_recomputes_attention_time_per_chunk():
    coeffs = dict(ROLLOUT_COEFFS, c_attn=0.0001)
    c = rollout_ctx(
        coeffs=coeffs, queue_depth=1,
        waiting=[{"id": "ahead", "prompt_len": 4, "computed": 0,
                  "scheduled_tokens": 0, "kv_blocks_est": 0}],
        waiting_work={"remaining_prefill_tokens": 4},
    )
    # 10ms base + 4ms linear + 0.1ms * 4 * (0 + 2) = 14.8ms.
    assert estimate_token_rollforward(c) == 0.006 + 0.0148


def test_token_rollforward_expands_capped_queue_from_exact_work_summary():
    c = rollout_ctx(
        queue_depth=2, waiting=[],
        waiting_work={"remaining_prefill_tokens": 8},
    )
    # Two synthesized 4-token prompts fit together only one per 4-token step.
    assert estimate_token_rollforward(c) == 0.006 + 2 * 0.014


def test_token_rollforward_gives_waiting_request_full_output_lifetime():
    c = rollout_ctx(
        max_batch_size=1, queue_depth=1, remaining_steps_est=1.0,
        output_steps_est=3.0,
        waiting=[{"id": "ahead", "prompt_len": 1, "computed": 0,
                  "scheduled_tokens": 0, "kv_blocks_est": 0}],
        waiting_work={"remaining_prefill_tokens": 1},
    )
    # Ahead uses one prefill plus three decode steps before releasing the only
    # slot. The old replay incorrectly used the current batch's one remaining
    # step as this newly admitted request's whole lifetime.
    assert math.isclose(estimate_token_rollforward(c), 0.006 + 4 * 0.011)


def test_token_rollforward_replays_fcfs_kv_preemption_instead_of_falling_back():
    c = rollout_ctx(
        max_batch_size=2, free_kv_blocks=0, queue_depth=1,
        output_steps_est=2.0,
        running=[
            {"id": "r1", "prompt_len": 1, "computed": 16,
             "scheduled_tokens": 0, "kv_blocks_est": 1,
             "true_remaining": -1, "remaining_est": 2},
            {"id": "r2", "prompt_len": 1, "computed": 16,
             "scheduled_tokens": 0, "kv_blocks_est": 1,
             "true_remaining": -1, "remaining_est": 2},
        ],
        waiting=[{"id": "ahead", "prompt_len": 1, "computed": 0,
                  "scheduled_tokens": 0, "kv_blocks_est": 0}],
        waiting_work={"remaining_prefill_tokens": 1},
    )
    admission, first_token = estimate_token_rollforward_times(c)
    assert admission > c["current_iter_remaining"]
    assert first_token is not None
    assert first_token > admission


def test_token_rollforward_legacy_context_falls_back_bit_exactly():
    c = ctx(batch_size=1, max_batch_size=1, free_kv_blocks=0, req_kv_need=1,
            t_iter=0.02, queue_depth=0, remaining_steps_est=6.0,
            running=[{"steps_done": 0, "kv_blocks": 1, "true_remaining": -1}])
    assert estimate_token_rollforward(c) == estimate_rollforward(c)
    assert estimate_token_rollforward_times(c) == (estimate_rollforward(c), None)
