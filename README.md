# Real-engine validation of a prefill/decode latency model

This repository reproduces the measurements behind two figures, `titer_modeb.png`
and `ttft_parity.png`. Both test a closed-form model of vLLM iteration timing
against a live engine, so that a routing rule built on that model rests on
measured behavior rather than on the simulator that trained it.

Everything here ran against vLLM 0.11.0 serving Llama-3.3-70B-Instruct on four
H100 GPUs with tensor parallelism 4. The model is stated in full below as
equations (E1) through (E5). No claim in this README points at a paper for its
content. Where a number appears, the script that produces it is named.

Two reproduction paths exist. The offline path needs only Python and reproduces
both figures bit-for-bit from data vendored here, in about a minute. The cluster
path regenerates that data from scratch and needs an OpenShift namespace with
four H100s. Start with the offline path. It confirms the toolchain works before
you spend GPU time.

---

## 1. What the model says

### (E1) Per-iteration latency

A vLLM engine step runs one batch to completion. Let `B` be the set of requests
the scheduler placed in that step. For each request `r`, `prompt_len_r` is its
prompt length, `computed_r` is how many of its tokens the engine had already
processed when the step began, and `kappa_r` is how many tokens this step was
granted for it. A request is prefilling while `computed_r < prompt_len_r` and
decoding once `computed_r >= prompt_len_r`.

The predicted wall-clock duration of the step is

```
T_iter(B) = c_base
          + c_dec * B_dec
          + c_kv  * SUM over r in decode(B) of computed_r
          + SUM over r in prefill(B) of [ c_pf   * kappa_r
                                        + c_attn * kappa_r * (computed_r + kappa_r / 2) ]
```

where `decode(B)` is the decoding requests, `prefill(B)` the prefilling ones, and
`B_dec = |decode(B)|`.

Each term has a physical reading. `c_base` is the fixed cost of running a step at
all, dominated by streaming the 70B weights through the GPUs. `c_dec` is the
marginal cost of carrying one more decoding request. `c_kv` prices reading one
resident context token out of the KV cache, which is why it multiplies the summed
context length. `c_pf` prices the projection work for one prefill token. `c_attn`
prices prefill attention, which compares each of the `kappa_r` new tokens against
every earlier token, so the count of pairs is `kappa_r * (computed_r + kappa_r/2)`
and the cost grows quadratically in prompt length.

The implementation is `predict_step` in `calibration/modeb/analysis.py`.

### The coefficients

`coeffs.json` at the repository root holds the frozen fit.

```
c_base = 0.015977418092905253   s        fixed per-iteration cost
c_dec  = 4.494688023467866e-05  s/req    per decoding request
c_kv   = 2.6932114735579514e-08 s/token  per resident context token
c_pf   = 6.352359596486496e-05  s/token  per prefill token, linear
c_attn = 1.289296826584455e-09  s/token² prefill attention, quadratic
```

Section 4 regenerates these from the raw measurements, bit-exactly.

### (E2) Chunk count

Chunked prefill splits a long prompt across steps, at most `kappa` tokens per
step, where `kappa` is the engine's `max_num_batched_tokens`. Every run here used
`kappa = 8192`. Prefix caching was disabled, so no part of a prompt is ever
already cached and the uncached suffix is the whole prompt. The chunk count is
therefore fixed by the prompt length alone:

```
n_c = ceil(prompt_len / kappa)
```

### (E3) A request's own prefill work

Chunk `k`, counting from zero, carries `t_k` tokens against an already-computed
causal prefix of `P_k` tokens:

```
t_k = min(kappa, prompt_len - k * kappa)
P_k = k * kappa

W_p = SUM over k = 0 .. n_c-1 of [ c_pf * t_k + c_attn * t_k * (P_k + t_k / 2) ]
```

`W_p` collects only the prefill terms of (E1) for this one request. It excludes
`c_base` and the decode terms on purpose, because those are charged once per
iteration by the `n_c * T_iter` term of (E4) rather than once per chunk.

The implementation is `prefill_work` in `calibration/admission/ttft_driver.py`.

### (E4) Composed time to first token

A router choosing where to send a request wants its time to first token. That
splits into the wait before the engine starts work and the prefill itself:

```
TTFT_hat = T_adm + n_c * T_iter(B_arrival) + W_p
```

`T_adm` is the admission delay, the wait from arrival until the scheduler first
runs the request. `B_arrival` is the batch resident at the moment the request
arrives. The request runs `n_c` iterations to finish prefilling, each costing
about `T_iter(B_arrival)`, and `W_p` adds its own prefill work, which
`T_iter(B_arrival)` excludes because the request was not yet in that batch.

Two variants share that compute term and differ only in `T_adm`.

- **oracle** substitutes each request's realized admission delay from (E5). Any
  error left is then attributable to the closed-form terms.
- **deployable** uses the roll-forward estimator a router actually runs online,
  which sees a queue snapshot and a censored mean output length. It is
  implemented in `calibration/admission/estimators.py` and replayed by
  `calibration/admission/analysis.py`.

### (E5) Realized quantities

Three timestamps come out of the capture. `t_enq` is when the request was
enqueued. `t_sched` is the start of the first step in which it appears. `t_first`
is the start of the first step in which `computed_r >= prompt_len_r`, which is the
step where its prefill has finished and the first token exists. Then

```
T_adm_realized   = t_sched - t_enq
prefill_realized = t_first - t_sched
TTFT_realized    = t_first - t_enq
```

Section 8 explains a limitation in how `t_enq` is measured. Read it before
quoting any deployable number.

---

## 2. What you need

**For the offline path.** Python 3.9 or newer and the packages in
`requirements.txt`, which are numpy, matplotlib, httpx and pytest. Nothing else.

**For the cluster path.**

- An OpenShift namespace you own, plus the `oc` CLI, logged in. Every manifest
  here targets `vramani-perfcal`. Change it throughout if yours differs.
- A schedulable node with four H100 GPUs, tolerating `nvidia.com/gpu`.
- A Hugging Face token with access to `meta-llama/Llama-3.3-70B-Instruct`, as a
  secret named `hf-token-secret`. The first run downloads about 140 GB.
- Two `ReadWriteMany` PVCs, one for results and one for the model cache.
- Roughly four hours of GPU time for the full sequence.

**Cluster etiquette, which matters on a shared cluster.** Use the
`nvidia.com/gpu` toleration only. Never add an `llm-d-benchmark-harness`
toleration or taint. Never uncordon a node. Never evict another tenant's GPU job
to make room. Scale `vllm-cal` to zero the moment a stage finishes, because idle
H100s block other people.

**One scheduler behavior to expect.** A reaper scales idle GPU deployments to
zero after about 90 minutes, which restarts the vLLM process mid-run. Capture
files are opened in append mode so no rows are lost. A restart does reset the
`perf_counter` epoch, the step counter and the request-id namespace, so
timestamps only compare within one process lifetime. The analysis handles this by
splitting at every reset boundary. Section 7 covers it.

---

## 3. Layout

```
coeffs.json                     frozen coefficients, the fit of (E1)
requirements.txt

calibration/                    stage 0, fitting the coefficients
  client.py                     in-cluster load generator for the fit sweeps
  fit.py                        the regressions
  report.py                     CLI wrapper, writes coeffs.json
  decode.csv prefill.csv mixed.csv    raw measurements, vendored
  run.md                        original stage-0 runbook

calibration/modeb/              stage 1, checking (E1) against real batches
  sitecustomize.py              the capture hook
  modeb_capture.py              pure record extraction, unit-testable
  analysis.py                   predict_step, regime classification, reporting
  run.md

calibration/admission/          stage 2, admission delay and composed TTFT
  sitecustomize.py              the capture hook, adds enqueue events
  admission_capture.py          pure record extraction
  estimators.py                 ports of the online estimators
  analysis.py                   context reconstruction and replay
  ttft_driver.py                composes (E4), writes the row file
  run.md
  ttft_seg1.json ttft_seg2.json committed reference outputs
  report_seg1.json report_seg2.json

calibration/deploy/             every manifest, 14 files
calibration/results/            vendored captured data and reports

figures/
  plot_titer_modeb.py           builds titer_modeb.png
  plot_ttft_full.py             builds ttft_parity.png
  ttft_rows.json                per-request rows, vendored
  verify_compute_term.py        checks (E1)-(E3) on the real capture
  decompose_ttft_error.py       attributes the composed error to a term
  *.reference.png               the committed figures, to diff against
```

---

## 4. Offline path, no cluster

Run these from the repository root. Total runtime is about a minute.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest calibration -q
```

Expect 105 tests to pass.

The first line says `python3` because many systems ship no bare `python`. Every
later command says `python`, which the activated virtualenv always provides. If
you skip the virtualenv, substitute `python3` throughout.

### 4a. Refit the coefficients

```sh
python -m calibration.report \
  --decode calibration/decode.csv \
  --prefill calibration/prefill.csv \
  --mixed calibration/mixed.csv \
  --chunk-bud 8192 --out-dir /tmp/stage0

python -c "import json,sys
a=json.load(open('/tmp/stage0/coeffs.json')); b=json.load(open('coeffs.json'))
print('coefficients reproduce bit-exactly' if a==b else 'MISMATCH')
sys.exit(0 if a==b else 1)"
```

Expect `decode_r2` 0.9908, `decode_mape` 0.8647, `prefill_r2` 0.9998, and all
five coefficients identical to `coeffs.json`.

The same output also prints `prefill_mape` 13.997, which looks alarming beside an
R² of 0.9998 and is not. `fit_prefill` regresses on the residual
`ttft - n_c * c_base` rather than on first-token time itself, so its error is
relative to a residualized target. The nine prompt lengths span two orders of
magnitude, and the smallest residuals sit near zero, where any absolute error
becomes a large percentage. R² measures the same fit against the spread of the
data and is the number to read here.

Expect `mixed_mape` to print **606.8**, and read section 9a before drawing any
conclusion from it. The honest mixed-regime number is 17.6%, and the difference
is two rows the fit pipeline cannot detect on its own.

### 4b. Build both figures

```bash
python figures/plot_titer_modeb.py
python figures/plot_ttft_full.py
```

Each script prints the statistics it plots. Those numbers are the check that
matters, and section 4d verifies them automatically.

The committed references are also byte-comparable, at the pinned matplotlib
version and nowhere else.

```bash
for p in titer_modeb ttft_parity; do
  if cmp -s figures/$p.png figures/$p.reference.png;
    then echo "$p identical";
    else echo "$p differs, check your matplotlib version"; fi
done
```

A difference here is almost always the library rather than the data. Rasterized
glyph bytes depend on the freetype build inside the matplotlib wheel, and
matplotlib 3.10.8 ships freetype 2.6.1 while 3.11.1 ships 2.14.3, which moves
every text pixel and changes the file by a few hundred bytes. `requirements.txt`
pins 3.10.8 for that reason. Under any other version every printed number stays
identical and only the pixels move, so trust section 4d over `cmp`.

### 4c. Check the model on real captured batches

```bash
python figures/verify_compute_term.py
python figures/decompose_ttft_error.py
```

The first recomputes (E2), (E3) and the compute term of (E4) for every request in
the vendored capture whose prefill was observed from its first chunk, then scores
them against realized prefill from (E5). The second attributes the composed TTFT
error to a term. Section 8 discusses what it shows.

### 4d. Verify every number in this README

```bash
python figures/check_reconciliation.py
```

This re-derives all 41 headline figures from the raw vendored data, using its own
implementation of (E1) and of the documented filters rather than importing the
plotting scripts, so it cross-checks them instead of trusting them. It compares
against the values written in this README and exits non-zero on any mismatch.
Expect `41/41 checks passed`. Nothing in it renders pixels, so it holds across
matplotlib versions.

---

## 5. Stage 0, fitting the coefficients

**What we did and why.** (E1) has five coefficients. Fitting them all at once on
mixed traffic would let errors in one term hide behind another, and it would also
mean the same data both trained and tested the model. We instead drove the engine
into two pure regimes, fit disjoint subsets of the coefficients in each, and then
predicted a third regime that no fit had seen.

Decode-only traffic makes every prefill term vanish, so (E1) collapses to
`c_base + c_dec * B_dec + c_kv * SUM computed_r`, linear in three unknowns.
Sweeping batch size and context length identifies all three. Prefill-only traffic
with a known prompt length isolates `c_pf` and `c_attn`, with `c_base` already
known and subtracted as `n_c * c_base`. Co-resident prefill-and-decode traffic
then tests whether the two halves add, which is (E1)'s central assumption.

**One methodology point worth stating,** because it changes the answer by an
order of magnitude. `fit_decode` in `calibration/fit.py` aggregates to the median
of each batch-size and context cell before the regression. Within a cell the
context sweep moves iteration time by under 0.03 ms, far below the roughly 3%
per-step timing jitter. Regressing raw rows therefore fits noise and gives
R² 0.065 with `c_kv` inflated about fourfold. Regressing the 28 cell medians
gives R² 0.9908. The comment at `calibration/fit.py:26` records this.

**Cluster steps.** The full runbook is `calibration/run.md`. In outline:

```bash
oc apply -f calibration/deploy/results-pvc.yaml
oc apply -f calibration/deploy/model-cache-pvc.yaml
oc apply -f calibration/deploy/vllm-70b-tp4.yaml
oc logs -f deploy/vllm-cal -n vramani-perfcal | grep -m1 "Application startup complete"

oc create configmap calibrate-code --from-file=client.py=./calibration/client.py \
  -n vramani-perfcal --dry-run=client -o yaml | oc apply -f -
oc apply -f calibration/deploy/calibrate-job.yaml
oc wait --for=condition=complete job/calibrate -n vramani-perfcal --timeout=3600s
```

`calibrate-job.yaml` pins the client to the vLLM node by pod affinity, so
round-trip time is negligible and client-side first-token time approximates
server compute directly. The prefill fit depends on that.

**One gate not to skip.** Before trusting `decode.csv`, inspect the raw
server-sent-event frames. `client.py` timestamps every streamed chunk carrying
text. If vLLM emits a terminal empty-text chunk, that adds one spurious timestamp
per stream and corrupts the last decode gap in every cell.

```bash
oc port-forward deploy/vllm-cal 8000:8000 -n vramani-perfcal &
PF_PID=$!; sleep 3
curl -sN http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"meta-llama/Llama-3.3-70B-Instruct","prompt":"Count to five:","max_tokens":5,"stream":true,"ignore_eos":true,"temperature":0}'
kill $PF_PID
```

Expect exactly five frames with non-null text, then `data: [DONE]`. If the last
frame before `[DONE]` has empty text or carries only a `finish_reason`, guard
`_stream_tokens` to append only when `finish_reason` is `None`, then re-ship the
ConfigMap. On vLLM 0.11.0 this was a non-issue.

Pull the CSVs with `calibration/deploy/extractor.yaml`, then fit locally with the
command in section 4a.

---

## 6. Stage 1, checking (E1) against real batches

**What we did and why.** A latency law can look good on the traffic that trained
it and still fail in production, because production batches mix prefill and
decode in proportions no sweep covers. We wanted the strongest available test, so
we stopped predicting the batch and started reading it. The engine tells us
exactly which requests were in each step and how many tokens each was granted. We
feed that real composition into (E1) and compare against the measured duration of
that same step. This is a teacher-forced test. It isolates the fidelity of (E1)
itself from any error in guessing what the batch will contain.

**What was instrumented.** `calibration/modeb/sitecustomize.py`, shipped as a
ConfigMap and placed on `PYTHONPATH`, patches two vLLM internals.

- `Scheduler.schedule` returns the scheduler output. The patch reads
  `num_scheduled_tokens` for `kappa_r`, plus the new and cached request lists for
  `computed_r` and `prompt_len_r`. Prompt length is immutable, so it is cached at
  a request's first appearance and reused, which is what `prompt_len_cache` in
  `modeb_capture.py` does.
- `EngineCore.step` is wrapped with `time.perf_counter()` on both sides, and
  writes one JSONL row per step.

`sitecustomize.py` is chosen deliberately. vLLM V1 runs the engine in a spawned
`EngineCoreProc` child, and a wrapper script around the parent would miss it.
Python imports `sitecustomize` automatically in every interpreter that starts
with it on the path, so the child gets patched too. The hook logs
`[MODEB] hook active` when it succeeds, and the runbook gates on that line.

**Which timestamp is authoritative.** Each row records `t_start` and `t_end`
around the step. The measured iteration time is the difference of consecutive
`t_start` values, not `t_end - t_start`. The former is the period the engine
actually sustains and includes inter-step overhead a request really waits
through. `parse_trajectory` computes it and drops the final step, which has no
successor.

**What was collected.** `trajectory.jsonl`, one row per step:

```json
{"step": 1, "t_start": 2956240.558, "t_end": 2956240.576,
 "total_scheduled": 1, "num_running": 1,
 "reqs": [{"id": "cmpl-...", "kappa": 1, "computed": 5, "prompt_len": 5}]}
```

Plus `meta.json` with the scheduler configuration, which pins the run:
`max_num_batched_tokens` 8192, `chunked_prefill_enabled` true,
`async_scheduling` false, `num_gpu_blocks` 29334, `max_model_len` 131072,
`tensor_parallel_size` 4. `async_scheduling` false matters, because the offline
analysis assumes one scheduling decision per step.

The vendored capture is 191,800 steps in
`calibration/results/modeb/trajectory.jsonl.gz`.

**The workload.** Five guidellm archetypes span the regime space, with token
counts pinned so prompt lengths are known rather than sampled: decode-corner
256 in and 512 out, balanced 2048 and 128, prefill-lean 8192 and 64,
prefill-bound 16000 and 16, conversation 1024 and 256. Prefill-bound at 16000
tokens exceeds `kappa`, which is what produces the two-chunk requests that test
(E2) and (E3) beyond a single chunk.

**Cluster steps.** Full runbook in `calibration/modeb/run.md`.

```bash
oc apply -f calibration/deploy/modeb-configmap.yaml
oc apply -f calibration/deploy/vllm-70b-tp4-modeb.yaml
oc logs deploy/vllm-cal -n vramani-perfcal | grep MODEB          # gate
oc exec deploy/vllm-cal -n vramani-perfcal -- cat /results/modeb/meta.json
bash calibration/modeb/run_sweep.sh    # prints 5 apply lines, review each
```

`run_sweep.sh` prints commands rather than running them, so an operator reviews
each before applying. Pull results with `extractor.yaml`, then plot with section
4b.

**What to expect.** `figures/plot_titer_modeb.py` prints:

```
n 190583 all MAPE 3.32
pure_decode (174665, 2.72)
pure_prefill (1200, 3.68)
mixed (14718, 10.38)
```

The script keeps only back-to-back steps, where the gap between one step ending
and the next beginning is under 2 ms. A larger gap means the engine idled waiting
for work, so the inter-`t_start` delta measures idle time rather than compute. It
also drops the first 250 steps as warmup.

Mixed batches at 10.4% are the weakest regime, roughly four times the decode
error. Co-resident execution shares fixed overhead and overlaps some work, so
adding prefill and decode terms double-counts a little. The effect is real and
bounded, and it is a limitation of (E1)'s additive form rather than a calibration
error.

---

## 7. Stage 2, admission delay and composed TTFT

**What we did and why.** Stage 1 validates one iteration. A router needs the time
to a request's first token, which is (E4). Two things had to be measured that
stage 1 never captured. First, `T_adm`, which needs a timestamp at enqueue, not
just at scheduling. Second, the queue and KV state a deployable estimator sees,
so its online prediction can be replayed faithfully offline.

**What was instrumented.** `calibration/admission/sitecustomize.py` keeps both
stage-1 patches and adds a third, plus two per-step fields.

- `Scheduler.add_request` is patched to stamp `t_enq` with `time.perf_counter()`
  and write one enqueue event carrying `req_id`, `t_enq` and `prompt_len`.
- Each step row gains `waiting_count`, an exact count of the waiting queue, and
  `waiting_ids`, the leading ids capped at 512. The count stays exact even when
  the id list truncates, so queue depth is never wrong. It also gains
  `free_kv_blocks`, read from `kv_cache_manager.block_pool.get_num_free_blocks()`.

A deployable estimator's whole input is a snapshot of occupancy, so the replay is
only as faithful as these fields.

**What was collected.** `admission_events.jsonl`, one row per enqueue:

```json
{"req_id": "cmpl-...", "t_enq": 2956240.113, "prompt_len": 256}
```

and `trajectory.jsonl` as in stage 1 with the three extra fields. The run
produced 290,300 steps and 34,800 enqueue events across two process lifetimes.

**The workload.** The same five archetypes, plus one fixed-rate job intended to
build a standing backlog. Section 9c records that the dedicated overload job
failed at its purpose, and where the genuine saturation signal came from instead.

**Cluster steps.** Full runbook in `calibration/admission/run.md`. Scale the
stage-1 deployment down first, because both reuse the `vllm-cal` name.

```bash
oc scale deploy/vllm-cal --replicas=0 -n vramani-perfcal
oc apply -f calibration/deploy/admission-configmap.yaml
oc apply -f calibration/deploy/vllm-70b-tp4-admission.yaml
oc logs deploy/vllm-cal -n vramani-perfcal | grep ADMISSION       # gate
```

The gate line must name all three patches. Then confirm `meta.json` carries
`async_scheduling` false and a positive `block_size`. The offline reconstruction
divides by `block_size` to get per-request KV blocks, so a missing or zero value
makes every estimate wrong.

Send one request and read five rows of each file by hand before spending GPU
time. Every trajectory row needs `waiting_count` and `free_kv_blocks`. Every
event row needs `t_enq`. A missing field means the live-path assumption is wrong
for your vLLM build, and a full sweep on that assumption wastes hours.

```bash
bash calibration/admission/run_sweep.sh
```

Then pull `trajectory.jsonl`, `admission_events.jsonl` and `meta.json`. The
trajectory is about 1.1 GB, which is why it is not vendored here.

**Reconstructing TTFT.** `calibration/admission/ttft_driver.py` does the offline
work. Run it in a CPU pod with the PVC mounted, using
`calibration/deploy/ttft-analysis-pod.yaml`, because the data is large and local
to the cluster.

```bash
python -m calibration.admission.ttft_driver \
  --trajectory /mnt/pvc/admission/trajectory.jsonl \
  --events     /mnt/pvc/admission/admission_events.jsonl \
  --meta       /mnt/pvc/admission/meta.json \
  --coeffs     coeffs.json \
  --out-rows   figures/ttft_rows.json \
  --out-dir    calibration/admission
```

The driver splits both files at every process restart, then per segment
reconstructs `t_sched`, `t_first` and the realized quantities of (E5), computes
`n_c` by (E2) and `W_p` by (E3), evaluates `T_iter` by (E1) on the batch resident
at arrival, and composes both variants of (E4). It writes per-segment reports and
the pooled row file the figure consumes.

Three implementation notes, each a deliberate choice.

- The analysis pod runs `python:3.11-slim` with no matplotlib and no root to
  install one, yet the frozen analysis module imports matplotlib at module scope.
  The driver inserts a placeholder module so the real frozen code imports
  unchanged. Stubbing the analysis module itself would risk diverging from the
  frozen `predict_step`, which is the one thing that must stay identical.
- Bracketing each arrival into its step by scanning every step is O(events ×
  steps), about four billion comparisons here, and does not finish. The driver
  binary-searches instead. `test_ttft_driver.py` asserts the two agree, including
  at the exact step boundaries where a half-open interval is easy to get wrong.
- A request still running when the capture ended has no observed departure, so
  the deployable replay cannot score it. Those are dropped from both variants
  rather than from one, so the two series in the parity plot cover the same
  requests. The driver reports the count as `censored_excluded`. This is why the
  pooled row file holds 30,734 rows while the oracle-only views in
  `ttft_seg2.json` cover more.

**What to expect.** `figures/plot_ttft_full.py` prints:

```
n 30734
oracle        (30734, 4.45, 7.89)
deploy all    (30734, 33.44, 57.52)
deploy nqueue (26622, 64.49, 52.96)
deploy queued (4112, -89.27, 86.99)
```

reading as count, median bias percent, MAPE percent.

---

## 8. Where the composed error lives

The oracle variant reaches 7.9% MAPE and the deployable variant 57.5%. Both use
the identical compute term, so the entire difference is `T_adm`.
`figures/decompose_ttft_error.py` separates them exactly, with no fitting.
Subtracting the two forms of (E4) recovers both unknowns from the row file:

```
compute   = p_oracle - r_tadm
t_adm_est = p_deploy - compute
```

Swapping only the admission term gives:

| variant | all | un-queued (26,622) | queued (4,112) |
|---|---|---|---|
| deployable as shipped | 57.5% | 53.0% | 87.0% |
| admission term set to 0 | 19.7% | **8.86%** | 89.7% |
| oracle, realized delay | 7.9% | **8.88%** | 1.5% |

Dropping the admission term lands on the oracle's own accuracy for the un-queued
subset, 8.86% against 8.88%, and on the same 5.3% median bias. The two do not
coincide exactly, because the realized delay there is small rather than zero, so
the oracle carries a residual median 11 microseconds the other does not. The
compute term therefore already carries oracle fidelity for 87% of requests, and
the deployable gap on that subset is the admission estimate alone.

**Probe placement.** The realized admission delay on the un-queued subset has
median 11 microseconds, with 85% of requests under 0.1 ms and p99 at 1.44 ms. A
step occupies the GPU for about 20 ms, so a request arriving mid-iteration cannot
physically be admitted in 11 microseconds. If `t_enq` were an arrival instant
independent of the engine's rhythm, the delay would spread across the whole
iteration and the median would sit near 10 ms.

The explanation is that `t_enq` is synchronized to the step boundary.
`calibration/admission/sitecustomize.py:112` patches `Scheduler.add_request`, and
vLLM V1's engine loop drains its input queue and then calls `schedule()`
immediately. So the real wait divides into two parts and we measure only the
second.

- **Part one** waits in the EngineCore input queue for the in-flight iteration to
  drain. Our probe cannot see it. This is what the estimator's floor models,
  since `floored_t_adm` in `estimators.py:9` raises any estimate to one full
  iteration.
- **Part two** waits in the scheduler's waiting queue for a slot or KV space. We
  measure this. It is near zero for un-queued requests and seconds for the tail.

Adding part one back and re-scoring the same predictions bounds what a correctly
placed probe would give:

| correction added to measured TTFT | un-queued MAPE | median bias |
|---|---|---|
| none, as measured | 53.0% | +64.5% |
| plus `T_iter`/2, the mean arrival phase | 23.1% | +24.8% |
| plus a uniform draw on [0, `T_iter`] | 25.1% | +17.5% |
| plus `T_iter`, an upper bound | 6.0% | +4.3% |

A correctly placed probe puts the deployable bulk between roughly 6% and 23%
rather than 53%. Quote 23% as the defensible figure, since the mean arrival phase
within an iteration is `T_iter`/2. This correction models the missing quantity
rather than measuring it. Only a re-run with the probe moved upstream settles it,
by additionally patching the EngineCore input-queue put and emitting an arrival
timestamp alongside `t_enq`. Keeping both timestamps would make the new capture a
superset, so the committed reference reports still reconcile.

**The queued tail does not improve.** It stays near 87%, and zeroing the
admission term makes it slightly worse at 89.7%. That error survives every
counterfactual, so it is not a measurement artifact. The estimator sees the depth
of the waiting queue but not the volume of chunked-prefill work already committed
ahead of the request, so it cannot anticipate a multi-second stall. A router
trusting it will treat admission as nearly free when slots are open, which is
correct, and will underestimate the cost of routing into a prefill-saturated
pool, which is the documented blind spot.

---

## 9. Known defects and honest scope

### 9a. The mixed-regime number needs two rows removed

`report.py` prints `mixed_mape` 606.8% on all twelve rows of `mixed.csv`. The
defensible number is 17.6% on ten. Two rows, both at batch size 8 with `P_k` 0
and 8192, record 17 ms for a step the model puts at 580 ms and 667 ms. An
8192-token prefill chunk takes 589 ms standalone, so 17 ms is physically
impossible for a step containing one. Those two decode gaps fell inside the
injection window by timestamp but executed before the injected prefill began,
because at batch size 8 the scheduling latency is about two fast decode gaps.
At batch size 16 and 32 the gaps are slower and all rows align.

This is a defect in the stage-0 client's window detection, not in (E1). The fix
needs a client re-run, detecting prefill onset by the magnitude jump, meaning the
first gap crossing about ten times decode time, rather than assuming onset at the
injection instant. `report.py` cannot detect the misalignment on its own, so it
reports all twelve rows and a reproducer sees 606.8%. Removing the two rows is
justified by the physical impossibility rather than by their being outliers.

### 9b. The paper figure carries a literal backslash

`plot_titer_modeb.py` labels its legend `MAPE {:.1f}\%`, and matplotlib renders
that literally as `2.7\%` without `usetex`. The committed reference PNG shows it.
The scripts here are kept byte-faithful so they reproduce the published figure
exactly. Fixing it means dropping the backslash in both this copy and the paper's.

### 9c. The dedicated overload job did not overload

The fixed-rate job targeted 110 requests per second and delivered about 7.5.
guidellm's constant profile is concurrency-bound, and 512-token requests taking
roughly 7 seconds each cap the achievable open-loop rate far below the target.
Its rows show a median realized delay of 0.147 ms, so no standing queue formed.
The genuine saturation signal comes from the sweep's own throughput phases, which
reached 75.2 requests per second at 1004 ms first-token time. The queued tail
analyzed in section 8 is bucketed by realized delay above 500 ms for that reason,
rather than by which job produced the request.

### 9d. What the driver's own verification does and does not cover

The driver's input is a 1.1 GB capture that exists only on the cluster PVC, so it
cannot be re-run offline. Its composition core is verified two ways instead.
`test_ttft_driver.py` exercises the full path on a synthetic trajectory where
every expected value is computed by hand. `figures/verify_compute_term.py` runs
(E1) through (E3) and (E5) against the real vendored capture, 18,277 requests,
and reproduces a prior independent computation of the same quantity:

| quantity | prior computation | this reconstruction |
|---|---|---|
| requests | ~18,278 | 18,277 |
| overall MAPE | ~9.7% | 9.70% |
| single-chunk MAPE | ~10% | 10.0% |
| two-chunk MAPE | ~1% | 1.1% |
| single-chunk median | predictions ~8% high | 8.4% high |
| two-chunk median | predictions ~1% low | 1.0% low |

Mind the sign convention on those last two rows, because two are in use across
this repository. `verify_compute_term.py` reports `median_ratio` as realized over
predicted and `bias` as that ratio minus one, which matches the committed
`ttft_seg*.json` reports. Under that convention a prediction running high shows a
negative bias, so the script prints `bias=-7.8%` for single-chunk requests, which
is the same fact as predictions running 8.4% high. `plot_ttft_full.py` inherits
the opposite convention from the figure it builds and reports predicted over
realized. `check_reconciliation.py` asserts both forms so neither can drift.

What remains unverified offline is the driver's segmentation and admission replay
on the original raw files. The committed `ttft_seg1.json` and `ttft_seg2.json` are
the reference for that, and `figures/ttft_rows.json` reproduces `ttft_seg1.json`
to the digit, at 8.4% oracle MAPE and 60.9% deployable MAPE on 18,880 rows.

### 9e. Scope

Only a single collocated instance was measured. Prefill and decode ran on the
same GPUs throughout, so a disaggregated deployment's KV transfer term is
untested. Every number here is one model on one hardware configuration, vLLM
0.11.0 with Llama-3.3-70B-Instruct on four H100s. The coefficients are specific
to that pairing. The functional form of (E1) is the portable claim, and
re-running stage 0 is what ports it.

### 9f. Two preconditions the driver now checks rather than assumes

An independent audit of `ttft_driver.py` found two unchecked assumptions. Neither
changes any number reported here, and both are now enforced, because someone
running this on a fresh capture would have no warning if either broke.

The first is ordering. `bisect_enqueue_bucket` binary-searches the step list by
`t_start`, while `add_step_deltas` sorts by step number to match the frozen
parser. Those agree only because `perf_counter` is monotonic within one process
and steps run sequentially, which holds across all 191,800 steps of the vendored
capture with zero violations. The frozen linear scan would tolerate an unordered
list and the binary search cannot, so it now raises instead of returning a
silently wrong step.

The second is self-inclusion. The arrival bracket is closed at its lower edge, so
a request whose `t_enq` fell exactly on the `t_start` of the step that admits it
would appear in its own `B_arrival`, and (E1) would charge its first prefill chunk
inside `T_iter` while (E3) charged the same chunk again in `W_p`. On the vendored
capture the smallest realized admission delay is 4.8 microseconds and no delay is
zero, so `t_enq` always precedes `t_sched` and the bracket always lands on the
prior in-flight step. The driver now filters the arriving request out of
`B_arrival` by id, which enforces (E4)'s stated invariant whatever the timestamps
do, and reports any occurrence as `self_in_arrival_batch`. On a synthetic tie the
guard prevents a 33% inflation of the compute term.

---

## 10. Reconciliation log

Every row below came from running this README's own commands in a throwaway
virtualenv built only from `requirements.txt`, on 2026-07-28.

| check | command | expected | observed |
|---|---|---|---|
| test suite | `pytest calibration -q` | all pass | 105 passed |
| coefficients | section 4a | identical to `coeffs.json` | bit-exact, all five |
| decode fit | section 4a | R² 0.9908, MAPE 0.86% | 0.9908, 0.8647% |
| prefill fit | section 4a | R² 0.9998 | 0.99977 |
| mixed, 12 rows | section 4a | 606.8% | 606.83% |
| mixed, 10 rows | section 9a | 17.6% | 17.56% |
| (E1) overall | `plot_titer_modeb.py` | 3.32% | 3.3181% |
| (E1) decode | same | 2.72% | 2.7207% |
| (E1) prefill | same | 3.68% | 3.6828% |
| (E1) mixed | same | 10.38% | 10.3775% |
| (E4) oracle | `plot_ttft_full.py` | ~7.9% | 7.8931% |
| (E4) deployable | same | ~57% | 57.5168% |
| compute term | `verify_compute_term.py` | ~9.7% | 9.70% |
| error split | `decompose_ttft_error.py` | lands on oracle on bulk | 8.86% vs 8.88% |
| segment 1 against pin | section 4d | 8.4% and 60.9% | 8.3896%, 60.8645% |
| all README numbers | `check_reconciliation.py` | 41/41 | 41/41 passed |
| both PNGs | section 4b | match reference | byte-identical at matplotlib 3.10.8 |

One caveat found while doing this. At matplotlib 3.11.1 both figures reproduce
every number identically and neither PNG matches its reference, because that
release bundles a newer freetype and re-rasterizes the text. `requirements.txt`
pins 3.10.8. Treat `check_reconciliation.py` as authoritative and `cmp` as a
convenience.

---

## 11. Script index

| script | needs a cluster | what it does |
|---|---|---|
| `calibration/client.py` | yes, runs in-pod | drives the stage-0 fit sweeps |
| `calibration/fit.py` | no | the three regressions of section 5 |
| `calibration/report.py` | no | CLI, writes `coeffs.json` |
| `calibration/modeb/sitecustomize.py` | yes, runs in-pod | stage-1 capture hook |
| `calibration/modeb/modeb_capture.py` | no | pure record extraction |
| `calibration/modeb/analysis.py` | no | `predict_step`, that is (E1) |
| `calibration/admission/sitecustomize.py` | yes, runs in-pod | stage-2 capture hook |
| `calibration/admission/estimators.py` | no | online estimator ports |
| `calibration/admission/analysis.py` | no | context reconstruction, replay |
| `calibration/admission/ttft_driver.py` | data from cluster | composes (E4) |
| `calibration/*/run_sweep.sh` | prints only | emits the sweep manifests |
| `figures/plot_titer_modeb.py` | no | `titer_modeb.png` |
| `figures/plot_ttft_full.py` | no | `ttft_parity.png` |
| `figures/verify_compute_term.py` | no | checks (E1)-(E3) on real data |
| `figures/decompose_ttft_error.py` | no | attributes composed error |
| `figures/check_reconciliation.py` | no | verifies every number in this README |
