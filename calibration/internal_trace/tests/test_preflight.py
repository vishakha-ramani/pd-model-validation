import json

import pytest

from calibration.internal_trace import preflight


def _row(regime, B, S):
    return {
        "step": 1,
        "regime": regime,
        "B_decode": B,
        "S_prefill": S,
        "engine_step_s": 0.02,
        "vllm_iteration_details": {
            "generation_requests": B,
            "context_tokens": S,
        },
    }


def test_validate_checks_version_and_scheduler_parity(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "TRACE_DIR", tmp_path)
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "vllm_version": "0.26.0",
                "tensor_parallel_size": 4,
                "prefix_caching_enabled": False,
            }
        )
    )
    result = preflight.validate(
        [_row("pure_prefill", 0, 64), _row("pure_decode", 1, 0)]
    )
    assert result["passed"] is True


def test_validate_fails_closed_on_phase_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "TRACE_DIR", tmp_path)
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "vllm_version": "0.26.0",
                "tensor_parallel_size": 4,
                "prefix_caching_enabled": False,
            }
        )
    )
    rows = [_row("pure_prefill", 0, 64), _row("pure_decode", 1, 0)]
    rows[1]["vllm_iteration_details"]["generation_requests"] = 2
    with pytest.raises(RuntimeError, match="decode classification mismatch"):
        preflight.validate(rows)
