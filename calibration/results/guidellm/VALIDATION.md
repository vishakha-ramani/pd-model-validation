# guidellm end-to-end validation (Mode A) — vLLM 0.11.0 / Llama-3.3-70B / H100 TP4

Run 2026-07-26, ns `vramani-perfcal`, `--profile sweep`, workload pinned to
1024-token prompt / 256-token output (deterministic). 10 benchmarks:
synchronous + throughput (saturating) + 8 constant rates. Raw: benchmarks.csv/.json;
tidy: curves.csv.

## Sweep curves (median unless noted)
| arm | req/s achieved | mean conc | TTFT med (ms) | ITL med (ms) | out tok/s |
|---|---|---|---|---|---|
| synchronous | 0.23 | 1.0 | 81.4 | 16.3 | 60 |
| constant 1.0 | 1.01 | 4.7 | 91.9 | 17.8 | 260 |
| constant 1.9 | 1.78 | 9.0 | 92.7 | 19.5 | 458 |
| constant 2.7 | 2.55 | 13.9 | 94.0 | 21.0 | 655 |
| constant 3.5 | 3.31 | 20.0 | 93.5 | 23.2 | 849 |
| constant 4.3 | 4.06 | 27.2 | 95.5 | 25.9 | 1040 |
| constant 5.1 | 4.79 | 36.1 | 98.1 | 29.2 | 1228 |
| constant 5.9 | 5.50 | 46.2 | 100.2 | 32.6 | 1411 |
| constant 6.7 | 6.21 | 57.9 | 101.6 | 36.3 | 1591 |
| throughput (sat.) | 6.72 | 406.5 | 12015 | 165.0 | 1659 |

Capacity ~6–7 req/s for this workload; the throughput arm drives full saturation
(TTFT 12s, ITL 165ms, concurrency 406).

## Instrumentation-free cross-check (single-stream, B=1, batch composition known)
The calibrated per-iteration law predicts the synchronous arm directly, no rollforward:
- TTFT: predicted 81.7 ms vs measured 81.4 ms (**0.4%**) — prefill law on a 1024-tok prompt.
- ITL:  predicted 16.05 ms vs measured 16.3 ms (**1.5%**) — decode law at B=1, avg ctx 1152.

This is independent third-party corroboration of the coefficients from RESULTS.md.

## Loaded arms need the rollforward (Mode A) or instrumentation (Mode B)
A naive pure-decode T_iter(B=concurrency) underpredicts measured ITL
(16.3→20.4 ms predicted vs 17.8→36.3 ms measured across conc 4.7→57.9). The gap
grows with load and is the prefill co-residency + queueing that a single snapshot
omits. Closing it needs the real per-iteration batch trajectory: feed guidellm's
arrival process into the rollforward (Mode A), or capture per-step num_scheduled_tokens
from an instrumented vLLM (Mode B). guidellm is client-side only and does not expose it.
