from calibration.admission.admission_capture import (
    build_enqueue_record, cap_waiting, build_admission_step_record,
)


def test_build_enqueue_record():
    r = build_enqueue_record("req-7", 12.5, 128)
    assert r == {"req_id": "req-7", "t_enq": 12.5, "prompt_len": 128}


def test_cap_waiting_under_cap():
    count, ids, trunc = cap_waiting(["a", "b", "c"], 512)
    assert count == 3
    assert ids == ["a", "b", "c"]
    assert trunc is False


def test_cap_waiting_over_cap():
    count, ids, trunc = cap_waiting([str(i) for i in range(600)], 512)
    assert count == 600
    assert ids == [str(i) for i in range(512)]
    assert trunc is True


def test_cap_waiting_empty():
    count, ids, trunc = cap_waiting([], 512)
    assert count == 0 and ids == [] and trunc is False


def test_build_admission_step_record_shape():
    reqs = [{"id": "a", "kappa": 1, "computed": 10, "prompt_len": 5}]
    rec = build_admission_step_record(
        step=3, t_start=1.0, t_end=1.01, total_scheduled=1, num_running=1,
        reqs=reqs, waiting_count=2, waiting_ids=["w1", "w2"],
        waiting_truncated=False, free_kv_blocks=1000)
    assert rec == {
        "step": 3, "t_start": 1.0, "t_end": 1.01, "total_scheduled": 1,
        "num_running": 1, "reqs": reqs, "waiting_count": 2,
        "waiting_ids": ["w1", "w2"], "waiting_truncated": False,
        "free_kv_blocks": 1000,
    }
