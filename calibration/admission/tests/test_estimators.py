import math
from calibration.admission.estimators import (
    floored_t_adm, estimate_fluid, estimate_rollforward, estimate_waiting,
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
