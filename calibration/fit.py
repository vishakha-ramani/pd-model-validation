import numpy as np
import math


def _ols(X, y):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return beta, pred, r2


def mape(pred, obs):
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    return float(np.mean(np.abs(pred - obs) / np.abs(obs)) * 100.0)


def fit_decode(rows):
    arr = np.asarray(rows, dtype=float)  # columns B, n, k, t_iter
    # Median-aggregate per (B, n) cell before OLS. Each cell holds one row per k
    # value in the max_tokens context sweep; that within-cell k signal is
    # sub-millisecond (c_kv * B * dk), far below the ~3% per-step timing jitter,
    # so fitting raw rows lets noise dominate the loss (R2 ~= 0.07, c_kv biased
    # ~4x). c_kv is identified by the between-cell n sweep, not within-cell k, so
    # collapsing each cell to its median (t, k) denoises without losing signal.
    cells = {}
    for Bv, nv, kv, tv in arr:
        cells.setdefault((Bv, nv), []).append((kv, tv))
    agg = []
    for (Bv, nv), pts in cells.items():
        ks = np.array([p[0] for p in pts])
        ts = np.array([p[1] for p in pts])
        agg.append((Bv, nv, float(np.median(ks)), float(np.median(ts))))
    agg = np.asarray(agg, dtype=float)
    B, n, k, t = agg[:, 0], agg[:, 1], agg[:, 2], agg[:, 3]
    X = np.column_stack([np.ones_like(B), B, B * (n + k)])
    beta, pred, r2 = _ols(X, t)
    return {
        "c_base": float(beta[0]),
        "c_dec": float(beta[1]),
        "c_kv": float(beta[2]),
        "r2": r2,
        "mape": mape(pred, t),
    }


def fit_prefill(rows, c_base, chunk_bud):
    arr = np.asarray(rows, dtype=float)  # columns n, ttft
    n, ttft = arr[:, 0], arr[:, 1]
    n_c = np.array([math.ceil(v / chunk_bud) for v in n], dtype=float)
    y = ttft - n_c * c_base
    X = np.column_stack([n, n ** 2 / 2.0])
    beta, pred, r2 = _ols(X, y)
    return {
        "c_pf": float(beta[0]),
        "c_attn": float(beta[1]),
        "r2": r2,
        "mape": mape(pred, y),
    }


def predict_mixed(row, coeffs):
    B, resident_ctx_sum, kappa, P_k = row
    return (coeffs["c_base"]
            + B * coeffs["c_dec"]
            + coeffs["c_kv"] * resident_ctx_sum
            + coeffs["c_pf"] * kappa
            + coeffs["c_attn"] * kappa * (P_k + kappa / 2.0))
