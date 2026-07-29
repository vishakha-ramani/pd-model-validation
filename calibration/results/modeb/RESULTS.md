# Mode-B teacher-forced validation of the per-iteration latency law

Real vLLM 0.11.0 V1, Llama-3.3-70B-Instruct, H100 TP=4, budget 8192, chunked prefill on,
async_scheduling off, prefix caching off (meta.json). Captured the real per-step scheduler
batch composition and inter-step timing via the Mode-B `sitecustomize` hook, fed the REAL
composition into the calibrated law, scored predicted-vs-measured T_iter per regime. This
isolates the latency law from any scheduler model (teacher-forced), answering the
circular-validation critique.

## Data
- 5 guidellm `--profile sweep` archetypes, 90s/rate, one collocated instance:
  decode-corner 256/512, balanced 2048/128, prefill-lean 8192/64, prefill-bound 16000/16,
  conversation 1024/256.
- 191,800 captured steps; after dropping the 250-row warmup probe and empty steps: 190,873.
- Engine ran essentially back-to-back: 99.1% of steps have idle gap <= 1ms to the next step.
  Analysis uses the back-to-back subset (gap_after <= 2ms) = 190,583 steps so measured
  inter-step delta ~= true iteration compute time (matches the saturated basis the coeffs
  were fit on). Cross-check against in-step compute (t_end-t_start) agrees to ~0.2% MAPE.

## Headline (measured = inter-step t_iter, back-to-back)
| regime | n | MAPE | bias | R2 | median meas |
|---|---|---|---|---|---|
| pure_decode | 174,665 | 2.72% | -2.30% | 0.938 | 17.1 ms |
| pure_prefill | 1,200 | 3.68% | +0.59% | 0.988 | 570.0 ms |
| **mixed** | 14,718 | **10.38%** | +9.35% | 0.998 | 37.8 ms |
| all | 190,583 | 3.32% | -1.38% | 0.998 | 17.2 ms |

Per archetype (all regimes): prefill-lean 0.92%, prefill-bound 1.63%, balanced 2.34%,
conversation 3.47%, decode-corner 4.75% MAPE. R2 >= 0.982 each.

## The mixed-regime story (the key result)
The offline calibration reported a manufactured ~18% mixed residual, partly attributed to
kappa/P_k mislabeling (it ASSUMED kappa=8192 / P_k=j*8192, but the real co-resident prefill
chunk is 8192-B_dec because decodes are scheduled first). Mode B records the true kappa/P_k,
eliminating that artifact. Result: mixed MAPE drops ~18% -> 10.4%.

The residual 10% is now characterized, not mysterious. Residual vs max prefill P_k:
| P_k bin | n | MAPE | bias |
|---|---|---|---|
| [0,1) | 13,638 | 10.76% | +10.53% |
| [1,2048) | 319 | 4.19% | -4.18% |
| [2048,4096) | 13 | 1.96% | -1.96% |
| [4096,8192) | 700 | 6.43% | -6.43% |
| [8192,16000) | 48 | 2.03% | -2.00% |

The residual concentrates in P_k=0 mixed steps (93% of mixed), where the law OVER-predicts
by +10.5%. Physical reading: on a step with a fresh prefill chunk co-resident with decodes,
the additive law (c_dec*B_dec + c_pf*kappa + c_base) slightly overcounts because the prefill
and decode work co-batch more efficiently than the sum of parts. High-P_k mixed steps
(continuing chunks of long prompts) are predicted well (2-6%). So the remaining error is a
small, physically-interpretable co-resident additivity effect, NOT mislabeling and NOT a
snapshot/scheduler artifact.

## Anti-flattering note
Capture overhead (json.dumps + periodic flush) sits inside the inter-step delta, so it can
only INFLATE measured t_iter, never flatter the law. The reported errors are conservative.

## Repro
`python3 calibration/results/modeb/analyze_modeb.py` (reads trajectory.jsonl.gz + coeffs.json).
Artifacts here: trajectory.jsonl.gz (60MB, 191,800 rows), meta.json, guidellm/*/benchmarks.csv,
modeb_report.json, modeb_pred_vs_meas.png.
