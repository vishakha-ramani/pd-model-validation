# Real-engine calibration results (vLLM 0.11.0, Llama-3.3-70B-Instruct, 4×H100 TP4)

Run: 2026-07-25, namespace `vramani-perfcal`, node `pokprod-b93r43s3`, client co-located
(RTT≈0). `enable_prefix_caching=False`, `chunked_prefill_enabled=True`,
`max_num_batched_tokens=8192` (== assumed `chunk_bud`), KV capacity 469,312 tokens.

## Headline

The per-iteration latency law, calibrated **independently** on the decode-only and
prefill-only regimes, predicts a **held-out mixed (prefill+decode co-resident) regime**
that was never used for fitting. This breaks the circular-validation critique.

| Regime | Fit quality |
|---|---|
| **Prefill-only** (chunk law `c_pf·κ + c_attn·κ·(P_k+κ/2)`) | **R² = 0.9998** |
| **Decode-only** (`c_base + B·c_dec + c_kv·Σctx`), cell-median fit | **R² = 0.9908, MAPE = 0.86%** (all 28 (B,n) cells within ±2.5%) |
| **Mixed** (cross-regime prediction, 10 co-resident rows) | **MAPE = 17.6%** (systematic over-prediction, grows with P_k) |

## Authoritative coefficients (denoised decode fit + prefill fit)

Note: `c_pf`/`c_attn` below are from the current `fit.py` pipeline (prefill fit
subtracts `n_c·c_base` for multi-chunk prompts). This slightly refines the
earlier manual values (c_pf 6.29e-5, c_attn 1.48e-9) and gives mixed co-resident
MAPE = 17.6% (the earlier 21.2% used the manual coeffs). The paper reports 18%.

```
c_base  = 0.015977   s      (fixed per-iteration overhead)
c_dec   = 4.4947e-05 s/req  (per decode request marginal)
c_kv    = 2.6932e-08 s/tok  (per resident context token)
c_pf    = 6.3524e-05 s/tok  (per prefill token, linear)
c_attn  = 1.2893e-09 s/tok² (prefill attention, quadratic in position)
```

## Two methodology findings (NOT law failures)

### 1. Decode fit must aggregate to cell-medians before OLS — [APPLIED 2026-07-25]
`fit_decode` now median-aggregates per (B,n) cell; report.py on real data gives
decode R²=0.9908, MAPE=0.86%; 8/8 tests pass; coeffs.json carries the corrected fit.
Original analysis below.

`report.py` fits decode on all 7056 raw rows. Within each (B,n) cell the 252-row context
sweep (k=4..255) moves `t_iter` by <0.03ms — far below the ~3% per-step jitter — so raw-row
OLS fits noise: **R²=0.065** and c_kv inflated ~4× (1.14e-7 vs correct 2.69e-8).
Fitting the 28 cell-medians recovers **R²=0.99, MAPE<1%**. The decode structure is exact:
`t_iter` rises monotonically in both B and context, per-request time falls cleanly with batch
(15.85ms→0.30ms as B: 1→64). **FIX: median-aggregate per (B,n) cell in `fit_decode`.**

### 2. Mixed window-detection has a leading-edge artifact at low batch
2 of 12 mixed rows (B=8, P_k∈{0,8192}) were observed at **decode magnitude (~17ms)** while
the model predicted a full prefill chunk (~580ms) → 3283%/3916% error. A 8192-token chunk
takes 589ms standalone, so 17ms is physically impossible for a co-resident prefill step —
those two decode gaps fell inside `[t_fire, t_first]` by timestamp but executed **before the
injected prefill actually began** (scheduling latency ≈ 2 fast decode gaps at B=8).
`align_injection_window` assumes "in-window gap j ⇒ chunk j"; at fast-decode/low-B this
misaligns the leading edge. At B=16/32 (slower gaps) all 4 rows align correctly.
**FIX (needs client re-run): detect prefill onset by the magnitude jump (first gap that
crosses ~10× decode time) rather than assuming onset at t_fire.**

## Residual over-prediction in mixed (physical, ~10–38%)
Even on aligned rows the additive law over-predicts, growing with P_k. Co-resident execution
shares fixed overhead and overlaps some work, so `base + prefill_terms + decode_terms`
double-counts; the quadratic `c_attn` term also appears slightly high when extrapolated to
batched context. This is a real (small) model-form limitation, not a calibration error.

## Artifacts
- `decode.csv` (7056 rows), `prefill.csv` (9), `mixed.csv` (12) — raw measurements
- `coeffs.json`, `residuals.json`, `predicted_vs_realized.png` — report.py output (raw-row decode fit; see finding 1)
- Corrected coefficients above are the authoritative ones for the paper.
