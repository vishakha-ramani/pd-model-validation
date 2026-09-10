import pytest

from calibration.internal_trace.analysis import annotate_steps, build_report
from calibration.internal_trace.capture import build_step_record


TRUE = {
    "alphaD": 0.015,
    "alphaP": 0.016,
    "c0": 0.00002,
    "c1": 0.00000003,
    "cPf": 0.00006,
    "cAttn": 0.000000001,
}


def _decode(seed, B, N):
    requests = [
        {
            "id": f"cmpl-internal-v1:s{seed}:decode:B{B}:N{N}:r{i}-0",
            "phase": "decode",
            "scheduled_tokens": 1,
            "computed_tokens": N + 10,
            "prompt_tokens": N,
            "output_tokens": 10,
            "prefill_tokens": 0,
            "decode_tokens": 1,
        }
        for i in range(B)
    ]
    duration = TRUE["alphaD"] + TRUE["c0"] * B + TRUE["c1"] * B * (N + 10)
    row = build_step_record(0, 0, round(duration * 1e9), requests)
    return row


def _prefill(seed, N, S, computed):
    request = {
        "id": f"cmpl-internal-v1:s{seed}:prefill:N{N}:r0-0",
        "phase": "prefill",
        "scheduled_tokens": S,
        "computed_tokens": computed,
        "prompt_tokens": N,
        "output_tokens": 0,
        "prefill_tokens": S,
        "decode_tokens": 0,
    }
    U = S * (computed + S / 2.0)
    duration = TRUE["alphaP"] + TRUE["cPf"] * S + TRUE["cAttn"] * U
    return build_step_record(0, 0, round(duration * 1e9), [request])


def _mixed(seed, B, S, computed):
    requests = [
        {
            "id": f"cmpl-internal-v1:s{seed}:mixed:B{B}:r0:decode:r{i}-0",
            "phase": "decode",
            "scheduled_tokens": 1,
            "computed_tokens": 4100 + i,
            "prompt_tokens": 4096,
            "output_tokens": 4,
            "prefill_tokens": 0,
            "decode_tokens": 1,
        }
        for i in range(B)
    ]
    requests.append(
        {
            "id": f"cmpl-internal-v1:s{seed}:mixed:B{B}:r0:prefill-0",
            "phase": "prefill",
            "scheduled_tokens": S,
            "computed_tokens": computed,
            "prompt_tokens": 24000,
            "output_tokens": 0,
            "prefill_tokens": S,
            "decode_tokens": 0,
        }
    )
    K = sum(request["computed_tokens"] for request in requests[:-1])
    U = S * (computed + S / 2.0)
    duration = (
        TRUE["alphaD"]
        + TRUE["c0"] * B
        + TRUE["c1"] * K
        + TRUE["cPf"] * S
        + TRUE["cAttn"] * U
    )
    return build_step_record(0, 0, round(duration * 1e9), requests)


def test_report_recovers_coefficients_and_validates_mixed():
    steps = []
    for seed in range(3):
        for B in (1, 4, 16):
            for N in (64, 1024, 4096):
                steps.append(_decode(seed, B, N))
        for N, S, computed in (
            (64, 64, 0),
            (2048, 2048, 0),
            (12000, 8192, 0),
            (12000, 3808, 8192),
        ):
            steps.append(_prefill(seed, N, S, computed))
        for B in (4, 16):
            steps.append(_mixed(seed, B, 8192 - B, 0))

    report = build_report(steps)
    for name, expected in TRUE.items():
        assert report["final_coefficients_seconds"][name] == pytest.approx(
            expected, rel=1e-6
        )
    assert report["coverage"]["decode_cells"] == 27
    assert report["coverage"]["mixed_steps"] == 6
    assert report["final_validation"]["mixed_exact_B_decode"]["mape_pct"] < 1e-5
    assert len(report["leave_one_seed_out"]) == 3
    assert report["reliability_gates"]["three_independent_seeds"] is True
    assert report["reliability_gates"]["all_coefficients_positive"] is True


def test_unlabelled_requests_are_not_silently_used():
    row = _decode(0, 1, 64)
    row["requests"][0]["id"] = "cmpl-random"
    annotated = annotate_steps([row])
    assert annotated[0]["seed"] is None
    assert annotated[0]["workload"] == "unlabelled"
