import numpy as np
from calibration.fit import fit_decode, fit_prefill, predict_mixed, mape

def test_fit_decode_recovers_known_coeffs():
    c_base, c_dec, c_kv = 0.020, 0.0011, 2.5e-7
    rows = []
    for B in [1, 2, 4, 8, 16, 32, 64]:
        for n in [64, 256, 1024, 4096]:
            for k in range(0, 256, 8):
                t = c_base + B * c_dec + c_kv * B * (n + k)
                rows.append((B, n, k, t))
    out = fit_decode(rows)
    assert abs(out["c_base"] - c_base) < 1e-6
    assert abs(out["c_dec"] - c_dec) < 1e-6
    assert abs(out["c_kv"] - c_kv) < 1e-9
    assert out["r2"] > 1 - 1e-9


def test_fit_prefill_recovers_known_coeffs():
    c_base = 0.020
    c_pf, c_attn = 3.0e-5, 1.0e-8
    chunk_bud = 8192
    rows = []
    for n in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        import math
        n_c = math.ceil(n / chunk_bud)
        ttft = n_c * c_base + c_pf * n + c_attn * (n ** 2 / 2)
        rows.append((n, ttft))
    out = fit_prefill(rows, c_base=c_base, chunk_bud=chunk_bud)
    assert abs(out["c_pf"] - c_pf) < 1e-7
    assert abs(out["c_attn"] - c_attn) < 1e-11
    assert out["r2"] > 1 - 1e-9


def test_predict_mixed_matches_construction():
    coeffs = {"c_base": 0.020, "c_pf": 3.0e-5, "c_attn": 1.0e-8,
              "c_dec": 0.0011, "c_kv": 2.5e-7}
    B, resident_ctx_sum, kappa, P_k = 8, 8 * 1024, 512, 2048
    expected = (coeffs["c_base"] + B * coeffs["c_dec"]
                + coeffs["c_kv"] * resident_ctx_sum
                + coeffs["c_pf"] * kappa
                + coeffs["c_attn"] * kappa * (P_k + kappa / 2))
    got = predict_mixed((B, resident_ctx_sum, kappa, P_k), coeffs)
    assert abs(got - expected) < 1e-12
    assert abs(mape([2.0, 4.0], [2.0, 4.0])) < 1e-12
