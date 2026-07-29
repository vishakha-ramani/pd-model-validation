# Admission-delay estimator validation — real vLLM results

Validation of the EDPP admission-delay (`T_adm = t_sched - t_enq`) estimators
against real vLLM 0.11.0 V1, Llama-3.3-70B-Instruct, TP4, H100. Run
2026-07-27 on `vramani-perfcal`. This is the admission-delay counterpart to
the Mode-B per-iteration latency-law validation. It closes the "circular
validation" gap by measuring the quantity the routing rule actually
minimizes against a real engine, teacher-forced on the engine's own captured
scheduler trajectory.

Frozen coeffs (`coeffs.json`, from the base calibration): `c_base=0.01598`,
`c_pf=6.352e-05`, `c_attn=1.289e-09`, `c_dec=4.495e-05`, `c_kv=2.693e-08`.
Scheduler config (meta.json): `async_scheduling=false`, `block_size=16`,
`num_gpu_blocks=29334`, `max_num_seqs=1024`, chunked prefill on.

## What was run

Five guidellm sweep archetypes (decode-corner 256/512, balanced,
prefill-lean, prefill-bound, conversation) plus one fixed-rate overload job,
all against a single instrumented `vllm-cal`. A `sitecustomize.py` hook
captured, per scheduler step, batch composition + real iteration time +
waiting-queue snapshot + free-KV, plus per-request enqueue events. Offline,
`analysis.py` reconstructs each request's enqueue-time context and replays the
Python ports of the Go `rollforward` and `fluid` estimators (oracle and
deployable variants), comparing predicted vs realized `T_adm`.

The gpu-reaper scaled `vllm-cal` to 0 once mid-sweep (90-min idle), restarting
the vLLM process. Capture files are append-mode, so no data was lost, but the
restart resets the perf_counter epoch, step counter, and request-id
namespace. Offline analysis therefore splits the capture at the restart
boundary and treats each process lifetime as its own segment.

- Segment 1 (pre-restart, one lifetime): decode-corner + balanced +
  prefill-lean. trajectory 197,950 steps, 20,600 enqueue events.
- Segment 2 (post-restart): prefill-bound + conversation + overload job.
  trajectory 92,350 steps, 14,200 enqueue events.

## Headline finding: realized `T_adm` is bimodal, and the estimator is
## harmless-high on the common mode and dangerously-low on the tail

The result is consistent across both segments and all four estimator/variant
combinations. The two occupancy-aware estimators (`rollforward`, `fluid`)
collapse onto each other here because the dominant behavior is their shared
`_slot_and_kv_fit` early-return floor path, which is estimator-agnostic.

**Free-slot mode (the common case, ~80-90% of requests).** When a request
arrives to a free sequence slot with KV headroom, vLLM admits it on the next
iteration and realized `T_adm` is microseconds (segment medians 0.011-0.15
ms). The estimator floors to one iteration (~19-21 ms predicted). It
over-predicts by roughly 1000-2000x (median ratio ~0.0005), but the absolute
prediction is ~20 ms against a sub-millisecond truth. In TTFT terms this
over-prediction is negligible and in the safe direction.

  | segment | subset | n | realized p50 | pred p50 | over-pred frac |
  |---------|--------|---|--------------|----------|----------------|
  | seg1 | pure_decode | 14949 | 0.011 ms | 19.5 ms | 1.00 |
  | seg2 | pure_decode | 6603 | 0.010 ms | 18.1 ms | 1.00 |

**Queued tail (the requests that actually wait, ~11-20%).** When a request
arrives during a prefill-heavy or slot-saturated batch it waits seconds to
tens of seconds. Here the estimator systematically UNDER-predicts and over_pred
fraction collapses to ~0 — it essentially never over-predicts a genuinely
queued request.

  | segment | subset | n | realized p50 | pred p50 | median ratio | over-pred frac |
  |---------|--------|---|--------------|----------|--------------|----------------|
  | seg1 | realized > 500 ms | 2028 | 32.9 s | 0.55 s | 59x | 0.00 |
  | seg1 | pure_prefill | 260 | 3.9 s | 0.28 s | 11.8x | 0.29 |
  | seg2 | realized > 500 ms | 2355 | 10.2 s | 0.55 s | 18.7x | 0.00 |
  | seg2 | pure_prefill | 127 | 57.8 s | 0.58 s | 91.5x | 0.09 |

The under-prediction is 1-2 orders of magnitude and is the snapshot-blind
limitation the harness design already documented. The estimator sees the
waiting-queue snapshot nearest the enqueue instant and cannot see the depth of
work already committed ahead of the request in a chunked-prefill batch, so it
cannot anticipate a multi-second stall.

**Standing backlog.** 1608 (seg1) and 1475 (seg2) requests were enqueued but
never entered the running batch within the capture window — the never-scheduled
symptom of a real backlog.

## Estimator ranking

On this workload `rollforward` and `fluid` are indistinguishable in the bulk
because the floor path dominates. A mild, expected ordering appears only in
the queued tail: `rollforward.oracle` has the lowest tail MAPE (seg1 87.2%,
seg2 85.4%) and `fluid.deployable` the highest (seg1 209.8%). Oracle beats
deployable by a small margin in the tail. None of this changes the qualitative
conclusion — every combination under-predicts the tail by 1-2 orders.

## Honest weakness: the dedicated overload job did not overload

The fixed-rate overload job (`--rate 110`, decode-corner 256/512) produced
only ~7.5 req/s of actual arrivals. guidellm's constant profile is
concurrency-bound, and 512-token/~7 s requests cap the achievable open-loop
rate far below the target. Its rows (seg2 "overload" bin, n=3591) show a
realized p50 of 0.147 ms — it did not build a standing queue. The genuine
saturation signal comes instead from the sweep's own throughput phases,
captured in the `queued_gt500ms` slice above. The decode-corner throughput
phase alone reached 75.2 req/s at 1004 ms TTFT in segment 1. The `N_BOUNDARY`
sub/overload split is therefore not the meaningful axis here; realized-magnitude
bucketing (`queued_gt500ms`) is.

## Bearing on the paper

This validates, against a real engine, the admission-delay estimator the
routing rule minimizes, on the engine's own scheduler trajectory rather than
on the simulator that trained the rule. The estimator is accurate-to-safe in
the free-slot regime that dominates sub-capacity operation and systematically
optimistic in the queued tail. A routing rule that trusts it will correctly
treat admission as near-free when slots are open and will underestimate the
cost of routing into a prefill-saturated pool — the same directional bias the
snapshot-granularity limit predicts.

## End-to-end TTFT composition

The admission delay is one term of the TTFT estimate the router scores
(`main.tex` eq:ttft-local), `TTFT(d,d) = Tadm + n_c * Titer(B) + Wp`. The other
term is the prefill the request runs on the chosen instance. We reconstruct
realized TTFT from the same capture and compare it to the composed estimate,
which lets us attribute end-to-end error to a term. Realized first-token time
is the first step at which a request's computed tokens reach its prompt length.
Realized TTFT is that instant minus the enqueue time. The chunk count `n_c` is
deterministic, `ceil(prompt_len / 8192)` at the captured token budget, and
prefix caching was off, so the uncached suffix is the full prompt. `Titer(B)`
uses the frozen coeffs on the batch resident at arrival, `Wp` is the chunk-sum
of the request's own prefill marginals, and both reuse the Mode-B `predict_step`
term-for-term. Reports: `ttft_seg1.json`, `ttft_seg2.json`.

The prefill-compute term validates well on its own, across single-chunk and
multi-chunk prompts (seg2 carries 848 two-chunk requests) and across queued and
unqueued requests.

  | segment | view | n | median ratio | bias | MAPE |
  |---------|------|---|--------------|------|------|
  | seg1 | prefill compute only | 18879 | 0.955 | -4.4% | 10.1% |
  | seg2 | prefill compute only | 12365 | 0.937 | -6.3% | 9.5% |
  | seg1 | full TTFT, oracle admission | 18879 | 0.957 | -4.3% | 8.4% |
  | seg2 | full TTFT, oracle admission | 12365 | 0.992 | -0.8% | 7.2% |

With a perfect admission delay the full TTFT estimate is accurate to under 10%
MAPE. Every larger end-to-end error traces to the admission term.

The deployable end-to-end TTFT inherits the bimodal admission signature. Below
capacity it is accurate and slightly high, because the deployable admission
floor adds about 20 ms that shows as over-prediction on a 40 ms request. In the
queued tail it under-predicts by roughly an order of magnitude, because that is
where the admission estimate is blind.

  | segment | view | n | median ratio | bias | MAPE | realized p50 | pred p50 |
  |---------|------|---|--------------|------|------|--------------|----------|
  | seg1 | deployable, not queued | 16960 | 0.579 | -42.1% | 56.5% | 43 ms | 72 ms |
  | seg1 | deployable, queued >500ms | 1919 | 21.9 | +2093% | 99.3% | 33.4 s | 1.23 s |
  | seg2 | deployable, not queued | 9660 | 0.752 | -24.8% | 46.7% | 81 ms | 107 ms |
  | seg2 | deployable, queued >500ms | 2193 | 9.31 | +831% | 76.2% | 10.8 s | 1.16 s |

The paper claim this supports is narrow and strong. The TTFT model's compute
term is validated end-to-end against a real engine to within about 5% median
bias and 10% MAPE, and the full estimate reaches the same accuracy whenever the
queue term is right. The one regime where end-to-end TTFT degrades is the
prefill-saturated tail, and the degradation is exactly the documented admission
blind spot rather than a separate modeling error. Only the collocated path is
tested here (single instance), so the disaggregated transfer term is unvalidated.

## Reproduce

Runbook: `calibration/admission/run.md`. The offline segmented replay used a
bisect-based enqueue bucket (identical `[t_start, t_end_next)` bracketing to
`analysis.enqueue_bucket`, O(E log S) instead of O(E*S)) so it finishes on the
full capture; the bracketing semantics and every downstream metric are
unchanged. Per-segment compact reports: `report_seg1.json`, `report_seg2.json`.
