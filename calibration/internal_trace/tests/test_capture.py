import pytest

from calibration.internal_trace.capture import (
    aggregate_features,
    build_step_record,
    extract_requests,
)


def test_extracts_exact_prefill_and_decode_state():
    cache = {"decode": 4096}
    requests = extract_requests(
        {"decode": 1, "prefill": 8187},
        [("prefill", 24000, 0)],
        ["decode"],
        [4112],
        [16],
        cache,
    )
    assert requests == [
        {
            "id": "decode",
            "phase": "decode",
            "scheduled_tokens": 1,
            "computed_tokens": 4112,
            "prompt_tokens": 4096,
            "output_tokens": 16,
            "prefill_tokens": 0,
            "decode_tokens": 1,
        },
        {
            "id": "prefill",
            "phase": "prefill",
            "scheduled_tokens": 8187,
            "computed_tokens": 0,
            "prompt_tokens": 24000,
            "output_tokens": 0,
            "prefill_tokens": 8187,
            "decode_tokens": 0,
        },
    ]


def test_aggregates_the_iteration_law_features():
    requests = [
        {"phase": "decode", "computed_tokens": 4100, "decode_tokens": 1},
        {"phase": "decode", "computed_tokens": 4200, "decode_tokens": 1},
        {
            "phase": "prefill",
            "computed_tokens": 8192,
            "prefill_tokens": 1000,
        },
    ]
    features = aggregate_features(requests)
    assert features["B_decode"] == 2
    assert features["B_scheduled"] == 3
    assert features["K_decode"] == 8300
    assert features["S_prefill"] == 1000
    assert features["U_prefill"] == 1000 * (8192 + 500)
    assert features["decode_scheduled_tokens"] == 2
    assert features["prefill_requests"] == 1


@pytest.mark.parametrize(
    ("requests", "expected"),
    [
        ([{"phase": "decode", "computed_tokens": 1, "decode_tokens": 1}], "pure_decode"),
        ([{"phase": "prefill", "computed_tokens": 0, "prefill_tokens": 1}], "pure_prefill"),
        (
            [
                {"phase": "decode", "computed_tokens": 1, "decode_tokens": 1},
                {"phase": "prefill", "computed_tokens": 0, "prefill_tokens": 1},
            ],
            "mixed",
        ),
    ],
)
def test_build_step_record_classifies_regime(requests, expected):
    record = build_step_record(7, 1_000_000_000, 1_025_000_000, requests)
    assert record["regime"] == expected
    assert record["engine_step_s"] == pytest.approx(0.025)
