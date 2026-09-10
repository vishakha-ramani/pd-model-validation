from calibration.admission.admission_capture import (
    build_enqueue_record, build_request_work_record, cap_waiting,
    summarize_waiting, build_admission_step_record,
)
from calibration.admission.sitecustomize import JsonlWriter
import json
from types import SimpleNamespace


def test_build_enqueue_record():
    r = build_enqueue_record("req-7", 12.5, 128)
    assert r == {"req_id": "req-7", "t_enq": 12.5, "prompt_len": 128}


def test_build_enqueue_record_preserves_engine_input_wait():
    r = build_enqueue_record("req-7", 12.5, 128, t_engine_arrive=12.48)
    assert r["t_engine_arrive"] == 12.48
    assert abs(r["engine_input_wait"] - 0.02) < 1e-12


def test_single_event_can_be_persisted_immediately(tmp_path):
    path = tmp_path / "admission_events.jsonl"
    writer = JsonlWriter(str(path), flush_every=1)
    writer.write({"req_id": "req-7", "t_enq": 12.5})
    assert json.loads(path.read_text()) == {"req_id": "req-7", "t_enq": 12.5}


def test_request_work_record_captures_prefill_and_kv_state():
    req = SimpleNamespace(request_id="q1", num_prompt_tokens=100,
                          num_computed_tokens=24, num_cached_tokens=16)
    r = build_request_work_record(req, scheduled_tokens=20, block_size=16)
    assert r == {
        "id": "q1", "prompt_len": 100, "computed": 24,
        "scheduled_tokens": 20, "cached_tokens": 16,
        "remaining_prefill_tokens": 76, "kv_blocks_est": 3,
    }


def test_request_work_record_marks_unobservable_cached_tokens():
    req = SimpleNamespace(req_id="q1", prompt_token_ids=list(range(33)),
                          num_computed_tokens=0, num_cached_tokens=-1)
    r = build_request_work_record(req, block_size=16)
    assert r["prompt_len"] == 33
    assert r["cached_tokens"] is None
    assert r["kv_blocks_est"] == 0


def test_waiting_summary_covers_uncapped_queue_work():
    records = [
        {"prompt_len": 100, "computed": 20, "cached_tokens": 16,
         "remaining_prefill_tokens": 80},
        {"prompt_len": 35, "computed": 0, "cached_tokens": None,
         "remaining_prefill_tokens": 35},
    ]
    assert summarize_waiting(records, block_size=16) == {
        "count": 2, "prompt_tokens": 135, "computed_tokens": 20,
        "cached_tokens_observed": 16, "cached_tokens_missing": 1,
        "remaining_prefill_tokens": 115, "full_prompt_kv_blocks": 10,
    }


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


def test_build_admission_step_record_adds_work_detail_when_available():
    running = [{"id": "r", "prompt_len": 10, "computed": 10,
                "scheduled_tokens": 1}]
    waiting = [{"id": "w", "prompt_len": 100, "computed": 0,
                "scheduled_tokens": 0}]
    work = {"count": 1, "remaining_prefill_tokens": 100}
    rec = build_admission_step_record(
        step=3, t_start=1.0, t_end=1.01, total_scheduled=1, num_running=1,
        reqs=[], waiting_count=1, waiting_ids=["w"], waiting_truncated=False,
        free_kv_blocks=1000, running_reqs=running, waiting_reqs=waiting,
        waiting_work=work)
    assert rec["running_reqs"] == running
    assert rec["waiting_reqs"] == waiting
    assert rec["waiting_work"] == work
