import random
import pytest
from calibration.client import (
    align_injection_window,
    env_int_list,
    per_step_median,
    random_prompt,
    steady_decode_steps,
    streamed_token_count,
    trim_ramp,
)

def test_random_prompt_exact_length_and_range():
    rng = random.Random(0)
    p = random_prompt(128, rng)
    assert len(p) == 128
    assert all(1000 <= t < 100000 for t in p)
    # distinct-ish: two prompts differ
    assert random_prompt(128, random.Random(1)) != p

def test_per_step_median_aligns_and_medians():
    # three streams, token timestamps (seconds). dt between consecutive tokens.
    streams = [
        [0.0, 0.10, 0.21, 0.33],
        [0.0, 0.11, 0.20, 0.34],
        [0.0, 0.09, 0.22, 0.32],
    ]
    steps = per_step_median(streams)
    # step k=0 is prompt->first token (ramp); k>=1 are decode dt medians
    ks = [k for k, _ in steps]
    assert ks == [0, 1, 2, 3]
    # median dt at k=1 across streams of (t1 - t0): [0.10,0.11,0.09] -> 0.10
    dt_k1 = dict(steps)[1]
    assert abs(dt_k1 - 0.10) < 1e-9

def test_trim_ramp_drops_warmup():
    steps = [(0, 0.5), (1, 0.10), (2, 0.10), (3, 0.10)]
    assert trim_ramp(steps, warmup=1) == [(1, 0.10), (2, 0.10), (3, 0.10)]


def test_steady_decode_steps_waits_for_every_stream():
    # Stream 0 began decoding earlier. With warmup=2 the common steady-state
    # barrier is t=2.0, when stream 1 emits its second token. Earlier gaps from
    # stream 0 must not enter the decode-only fit.
    streams = [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ]
    steps = steady_decode_steps(streams, warmup=2)
    assert [k for k, _ in steps] == pytest.approx([2.5, 3.5, 4.5])
    assert [dt for _, dt in steps] == pytest.approx([1.0, 1.0, 1.0])


def test_env_int_list_uses_default_and_parses_override(monkeypatch):
    monkeypatch.delenv("CAL_TEST_VALUES", raising=False)
    assert env_int_list("CAL_TEST_VALUES", [1, 2]) == [1, 2]
    monkeypatch.setenv("CAL_TEST_VALUES", "4, 8,16")
    assert env_int_list("CAL_TEST_VALUES", [1, 2]) == [4, 8, 16]


def test_streamed_token_count_uses_ids_and_rejects_terminal_metadata():
    # The final generated token may also carry finish_reason="length".
    assert streamed_token_count({
        "text": " I", "token_ids": [358], "finish_reason": "length"
    }) == 1
    # A metadata-only terminal frame is not a generated token.
    assert streamed_token_count({
        "text": "", "token_ids": [], "finish_reason": "length"
    }) == 0
    # Compatibility fallback for servers without return_token_ids support.
    assert streamed_token_count({"text": "hello", "finish_reason": None}) == 1
    assert streamed_token_count({"text": "", "finish_reason": "stop"}) == 0

def test_align_injection_window_selects_labels_and_aligns_chunks():
    from calibration.client import align_injection_window
    # window [10.0, 12.0]; two decode streams. chunk_bud=100, n_inject=250 -> 3 chunks.
    # Each stream has exactly 3 gaps overlapping the window (k=2,3,4), dt=1.0 each.
    sA = [9.0, 9.5, 10.5, 11.5, 12.5, 13.5]
    sB = [9.1, 9.6, 10.6, 11.6, 12.6, 13.6]
    rows = align_injection_window([sA, sB], t_fire=10.0, t_first=12.0,
                                  n_inject=250, chunk_bud=100, B=2, n_decode=1000)
    assert len(rows) == 3
    # j=0: kappa=100, P_k=0,   ctx=(1000+2)*2=2004 ; j=1: kappa=100,P_k=100,ctx=(1003)*2=2006
    # j=2: kappa=50 (250-200), P_k=200, ctx=(1004)*2=2008 ; median dt=1.0 throughout
    assert rows[0] == (2, 2004.0, 100, 0, 1.0)
    assert rows[1] == (2, 2006.0, 100, 100, 1.0)
    assert rows[2] == (2, 2008.0, 50, 200, 1.0)


def test_align_injection_window_rejects_short_edge_gaps():
    # Client-side request admission and first-token delivery add ordinary
    # decode gaps at the edges of the observed window. The two long interior
    # gaps are the actual prefill-bearing engine iterations.
    stream = [9.9, 10.1, 10.3, 11.0, 11.8, 12.0, 12.2]
    rows = align_injection_window(
        [stream],
        t_fire=10.0,
        t_first=12.0,
        n_inject=200,
        chunk_bud=100,
        B=1,
        n_decode=1000,
    )
    assert [row[:4] for row in rows] == [
        (1, 1003.0, 100, 0),
        (1, 1004.0, 100, 100),
    ]
    assert [row[4] for row in rows] == pytest.approx([0.7, 0.8])
