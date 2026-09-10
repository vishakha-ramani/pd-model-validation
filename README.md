# Real-engine prefill/decode latency-model validation

This repository measures the physical quantities used by the causal SLO
externality router against live vLLM deployments. The current calibration runs
Qwen3-14B on one H100 and Llama-3.3-70B-Instruct on four H100s. It separates
decode-only and prefill-only traffic to identify the latency coefficients, then
uses mixed prefill-plus-decode traffic only as held-out validation.

The newest completed experiment is the Llama-3.3-70B calibration on vLLM 0.26.0.
All three replicates, fitted coefficients, residuals, plot, and cross-replicate
report are committed under
[`calibration/results/llama3.3-70b-tp4-vllm-0.26.0/replicates-v4`](calibration/results/llama3.3-70b-tp4-vllm-0.26.0/replicates-v4).

## What happens during one iteration

During one GPU iteration, vLLM can simultaneously:

- advance every decoding request by one token;
- process chunks of one or more prefilling requests;
- read the KV context required by decode attention; and
- compute attention between new prefill tokens and their preceding context.

We model the duration of that iteration as

$$
T =
\alpha
+c_{\mathrm{dec}}B
+c_{\mathrm{kv}}K
+c_{\mathrm{pf}}S
+c_{\mathrm{attn}}U.
$$

Here $B$ is the number of requests decoding in the iteration, $K$ is the
sum of their resident context lengths, $S$ is the number of prefill tokens
scheduled in the iteration, and $U$ is the prefill attention work. The five
coefficients have the following physical meanings.

| Coefficient | Physical meaning |
|---|---|
| \(\alpha\) | Fixed cost of running an iteration |
| \(c_{\rm dec}\) | Marginal cost of carrying one decoding request |
| \(c_{\rm kv}\) | Marginal cost of reading one resident KV token |
| \(c_{\rm pf}\) | Marginal linear-layer cost of one prefill token |
| \(c_{\rm attn}\) | Marginal cost of one unit of prefill attention |

For a prefill chunk of \(\kappa\) tokens beginning after a causal prefix of
\(P\) tokens,

$$
U=\kappa\left(P+\frac{\kappa}{2}\right).
$$

This is the intended interpretation of the iteration law: one baseline cost
plus the marginal contributions of the decode and prefill work that vLLM
actually schedules.

## Environment calibrated by the newest run

The server is defined in
[`calibration/deploy/vllm-llama70b-tp4-v026.yaml`](calibration/deploy/vllm-llama70b-tp4-v026.yaml).

| Setting | Value |
|---|---|
| Model | `meta-llama/Llama-3.3-70B-Instruct` |
| Model revision | `6f6073b423013f6a7d4d9f39144961bfbfbc386b` |
| vLLM | `0.26.0` |
| Container | `docker.io/vllm/vllm-openai@sha256:770fe65b2c73ee74a5c42165cf3433de4048cc2cd9c57a937ca4e35aba5aa87b` |
| Hardware | 4 H100 GPUs |
| Tensor parallelism | 4 |
| Maximum model length | 32,768 tokens |
| Maximum sequences | 128 |
| Iteration token budget | 8,192 tokens |
| KV block size | 16 tokens |
| Prefix caching | Disabled |
| Chunked prefill | Enabled |
| Measured KV capacity | 506,368 tokens |

These settings are part of the coefficient definition. Changing the model,
model revision, GPU, tensor parallelism, vLLM version, precision, or scheduler
settings can change the measured coefficients.

`max_num_seqs` is a scheduler ceiling, not the batch size measured by the
regression. vLLM uses continuous batching, so the active decode batch $B$
changes from iteration to iteration. The calibration deliberately measures
several values of $B$ below that ceiling.

## Isolating the decode terms

To estimate \(\alpha,c_{\rm dec},c_{\rm kv}\), the client creates iterations
with no prefill work. The law reduces to

$$
T_{\rm decode}=\alpha+c_{\rm dec}B+c_{\rm kv}K.
$$

The experiment sweeps:

- decode batch \(B\) over 1, 2, 4, 8, 16, 32, and 64;
- prompt context over 64, 256, 1,024, 4,096, and 6,144 tokens; and
- 256 generated tokens per request.

For a synchronized cell whose requests have equal prompt length $n$ and
generated-token index $k$, the aggregate resident context is

$$
K=B(n+k).
$$

The client submits $B$ concurrent streaming requests and timestamps every
generated token. It waits until every submitted request has emitted four
warm-up tokens before retaining any gaps. This wall-clock barrier matters:
without it, a large prompt could still be prefilling while the client labels
the submitted concurrency as a resident decode batch. That would contaminate
the decode-only fit with prefill work. The implementation is
[`steady_decode_steps`](calibration/client.py).

For each aligned iteration, the client takes the median inter-token gap across
the streams. Each replicate produced 35 decode cells and approximately 8,591
usable iteration rows.

Before ordinary least squares, the fit median-aggregates every \((B,n)\) cell.
Within one cell, the change caused by a few additional KV tokens is smaller than
ordinary step-time jitter. The independent changes across batch size and prompt
length identify the coefficients much more reliably. The regression uses the
feature vector

$$
[1,\ B,\ B(n+k)]
$$

against the observed iteration duration. The implementation is
[`fit_decode`](calibration/fit.py).

## Isolating the prefill terms

To estimate \(c_{\rm pf}\) and \(c_{\rm attn}\), the client submits one request
at a time and asks for one output token. There are no resident decode requests.
Prompt lengths are 64, 128, 256, 512, 1,024, 2,048, 4,096, 8,192, 12,000,
16,000, and 24,000 tokens, with ten repetitions per length and seed.

For prompt length $N$, vLLM needs

$$
n_c=\left\lceil\frac{N}{8192}\right\rceil
$$

iterations. The first-token model is

$$
T_{\rm prefill}
=n_c\alpha+c_{\rm pf}N+c_{\rm attn}\frac{N^2}{2}.
$$

The attention term follows from the chunk identity

$$
\sum_j \kappa_j\left(P_j+\frac{\kappa_j}{2}\right)=\frac{N^2}{2}.
$$

We subtract the already fitted baseline,

$$
y=T_{\rm prefill}-n_c\alpha,
$$

then regress $y$ on $N$ and $N^2/2$. The prefill fit therefore does not
introduce a second free intercept that could hide an error in the baseline.
The three replicates contain 330 prefill observations in total.

## Testing, rather than fitting, mixed iterations

Mixed data is not used to estimate any coefficient. The client first creates a
resident decode batch with 4, 8, 16, or 32 requests, each with a 4,096-token
context. It then injects a 24,000-token prompt. With the configured token budget,
that prompt nominally occupies three prefill iterations: 8,192, 8,192, and
7,616 tokens.

Client-side HTTP admission and first-token delivery add ordinary decode gaps at
the edges of the observed window. For each decode stream, the client therefore
selects the three longest overlapping gaps and restores them to chronological
order before labeling the corresponding prefill positions. The resulting
measurement is compared with

$$
\widehat T_{\rm mixed}
=\alpha+c_{\rm dec}B+c_{\rm kv}K
+c_{\rm pf}\kappa
+c_{\rm attn}\kappa(P+\kappa/2).
$$

Because mixed observations do not participate in fitting, their error tests the
central additive assumption: coefficients learned from isolated decode and
prefill must predict an iteration in which both phases share the accelerator.

## Repeating the experiment and holding out each seed

The Kubernetes job is defined in
[`calibration/deploy/calibrate-llama70b-v026-replicates-job.yaml`](calibration/deploy/calibrate-llama70b-v026-replicates-job.yaml).
It ran seeds 0, 1, and 2 sequentially against the same engine. Each seed issued
635 decode requests, 110 prefill requests, and 640 mixed-regime requests: 1,385
requests per seed and 4,155 requests overall.

[`calibration.replicate_report`](calibration/replicate_report.py) performs three
leave-one-replicate-out folds. Each fold fits two seeds and predicts the unseen
third seed. This reveals whether a coefficient is tied to one random-token
sample or one moment in the run rather than to the serving configuration.

The automated coverage gate passed for all three replicates:

- all 35 decode cells are present;
- every decode cell has at least 202 retained rows, versus 32 required;
- every prefill length has ten observations;
- every mixed batch has 30 observations; and
- every decode replicate records that the steady-state barrier was active.

The complete report is
[`replicate-validation.json`](calibration/results/llama3.3-70b-tp4-vllm-0.26.0/replicates-v4/replicate-validation.json).

## Final vLLM 0.26.0 Llama coefficients

The fit over all three replicates produced:

| Coefficient | Seconds | EPP microseconds |
|---|---:|---:|
| \(\alpha\) | 0.0149963823 | 14,996.382 |
| \(c_{\rm dec}\) | \(2.57620\times10^{-5}\) | 25.762/request |
| \(c_{\rm kv}\) | \(2.69182\times10^{-8}\) | 0.026918/KV token |
| \(c_{\rm pf}\) | \(6.55559\times10^{-5}\) | 65.556/prefill token |
| \(c_{\rm attn}\) | \(1.13626\times10^{-9}\) | 0.001136/attention unit |

Machine-readable values are in
[`combined-fit/coeffs.json`](calibration/results/llama3.3-70b-tp4-vllm-0.26.0/replicates-v4/combined-fit/coeffs.json).

The leave-one-replicate-out coefficient variation is small:

| Coefficient | Coefficient of variation |
|---|---:|
| \(\alpha\) | 0.78% |
| \(c_{\rm dec}\) | 1.09% |
| \(c_{\rm kv}\) | 0.36% |
| \(c_{\rm pf}\) | 0.04% |
| \(c_{\rm attn}\) | 0.18% |

Held-out prediction errors are:

| Regime | Median absolute percentage error | 95th percentile |
|---|---:|---:|
| Decode-only | 1.0--2.4% | 3.1--4.6% |
| Prefill-only TTFT | 2.5--2.9% | 11.9--12.8% |
| Mixed | 3.1--3.9% | 9.2--10.2% |

The final combined fit reports decode MAPE 0.94%, decode \(R^2=0.995\),
prefill \(R^2=0.99988\), and held-out mixed MAPE 3.68%. Its reported raw
`prefill_mape` is 9.0% because that diagnostic first subtracts the iteration
baseline and divides by the remaining marginal work, which is very small for
short prompts. The held-out full-TTFT error in the table is the more
interpretable user-visible quantity.

## One iteration worked by hand

Consider 32 decoding requests with aggregate decode context
\(K=135{,}168\), sharing an iteration with the first 8,192-token chunk of a
long prompt. The fitted model predicts

$$
T_{\rm decode}=19.46\ \mathrm{ms},
\qquad
T_{\rm prefill}=590.16\ \mathrm{ms},
$$

and

$$
T_{\rm mixed}=594.62\ \mathrm{ms}.
$$

The prefill work dominates this iteration. Adding that work to a decoder delays
every resident request, which is exactly the interference that the causal SLO
externality term is intended to value.

## Reproducing the v0.26 analysis locally

After copying `replicates-v4` from the results PVC, run:

```bash
python3 -m pytest \
  calibration/tests/test_client_logic.py \
  calibration/tests/test_fit.py \
  calibration/tests/test_report.py \
  calibration/tests/test_replicate_report.py -q

R=calibration/results/llama3.3-70b-tp4-vllm-0.26.0/replicates-v4

python3 -m calibration.replicate_report \
  --replicate-dir "$R/replicate-0" \
  --replicate-dir "$R/replicate-1" \
  --replicate-dir "$R/replicate-2" \
  --chunk-bud 8192 \
  --out "$R/replicate-validation.json"

python3 -m calibration.report \
  --decode "$R/replicate-0/decode.csv" \
  --decode "$R/replicate-1/decode.csv" \
  --decode "$R/replicate-2/decode.csv" \
  --prefill "$R/replicate-0/prefill.csv" \
  --prefill "$R/replicate-1/prefill.csv" \
  --prefill "$R/replicate-2/prefill.csv" \
  --mixed "$R/replicate-0/mixed.csv" \
  --mixed "$R/replicate-1/mixed.csv" \
  --mixed "$R/replicate-2/mixed.csv" \
  --chunk-bud 8192 \
  --out-dir "$R/combined-fit"
```

The first command tests the timestamp alignment, fitting, reporting, and
cross-replicate logic. The second checks sample coverage, coefficient stability,
and held-out prediction. The third fits the final coefficients using all three
replicates and regenerates the mixed prediction plot.

## What this result does and does not establish

The experiment supports the iteration law inside its measured region:

- decode batch at most 64;
- prompt length at most 24,000 tokens;
- Llama-3.3-70B on four H100s with tensor parallelism four; and
- the exact vLLM and scheduler configuration listed above.

It does not yet establish complete router correctness or a goodput improvement.
Four limitations remain important.

1. These measurements use token timestamps at a colocated streaming client,
   not an internal vLLM scheduler timestamp.
2. The mixed client infers which gaps carried prefill. It does not directly
   record the scheduler's actual \(B,K,S,U\) for every iteration.
3. Decode tokens consume part of `max_num_batched_tokens`, so the actual
   prefill grant can be slightly smaller than the nominal chunk label.
4. The three seeds vary requests and time, but use the same engine and node;
   they do not measure node-to-node hardware variation.

The next validation step is therefore teacher-forced instrumentation. vLLM
must record the actual scheduled \(B,K,S,U\) and iteration duration. Feeding
that exact batch composition into the law separates formula error from errors
in EPP's view of server state. After that, a controlled Qwen 1P2D workload can
trace every request from arrival, through candidate scores and placement, to
measured TTFT, ITL, end-to-end latency, and SLO goodput.

---

## Legacy vLLM 0.11.0 Mode-B and admission-validation study

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
- **deployable** uses an admission estimator a router can run online. The
  schema-v2 `token_rollforward` estimator replays the running-first, FIFO-waiting
  scheduler under token, sequence-slot, and KV budgets, recomputing iteration
  latency after every simulated step. The frozen `rollforward` remains available
  to reproduce the original results.

### (E5) Realized quantities

Schema-v2 captures retain two arrival-side timestamps. `t_engine_arrive` is
stamped before the request enters EngineCore's input queue; `t_enq` is stamped
later in `Scheduler.add_request`. Let `t_arrive = t_engine_arrive` when present,
falling back to `t_enq` for the legacy capture. `t_sched` is the start of the
first step in which the request appears. `t_first` is the start of the first step
in which `computed_r >= prompt_len_r`, where its prefill has completed and the
first token exists. Then

```
T_adm_realized   = t_sched - t_arrive
prefill_realized = t_first - t_sched
TTFT_realized    = t_first - t_arrive
```

Section 8 explains why the original `t_enq`-only results are retained as legacy
references and why new accuracy numbers require a cluster rerun.

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

Expect the full calibration test suite to pass.

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

**What is instrumented now.** `calibration/admission/sitecustomize.py` writes
capture schema 2.

- `EngineCore.preprocess_add_request` stamps `t_engine_arrive` in the input
  socket thread before `EngineCoreProc.input_queue`. `Scheduler.add_request`
  retains the later `t_enq`, so `engine_input_wait = t_enq - t_engine_arrive`
  directly measures the wait the original probe missed.
- Each step records every running request's prompt, computed tokens, current
  scheduled grant and estimated KV blocks. Waiting-request detail is capped at
  512, but the uncapped queue count, prompt/computed/cached tokens, remaining
  prefill work, and full-prompt KV demand are aggregated exactly.
- `free_kv_blocks` is read after scheduling from vLLM's block pool. Together
  with `max_num_batched_tokens` and `max_num_seqs` in `meta.json`, this is enough
  to replay the scheduler's token, slot, and KV gates.

A deployable estimator's whole input is a snapshot of occupancy, so the replay is
only as faithful as these fields.

**What was collected.** `admission_events.jsonl`, one row per enqueue:

```json
{"req_id":"cmpl-...","t_engine_arrive":2956240.094,"t_enq":2956240.113,
 "engine_input_wait":0.019,"prompt_len":256}
```

and `trajectory.jsonl` as in stage 1 with the scheduler-work snapshots. The
original schema-v1 run produced 290,300 steps and 34,800 enqueue events across
two process lifetimes; it does not contain these new fields.

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

The gate line must name all four patches. Then confirm `meta.json` carries
`admission_capture_schema: 2`, the named upstream arrival probe,
`async_scheduling: false`, and a positive `block_size`. The offline
reconstruction divides by `block_size` to get per-request KV blocks, so a
missing or zero value makes every estimate wrong.

Send one request and read five rows of each file by hand before spending GPU
time. Every trajectory row needs `waiting_count`, `free_kv_blocks`,
`running_reqs`, `waiting_reqs`, and `waiting_work`. Every event row needs both
`t_engine_arrive` and `t_enq`, with a nonnegative `engine_input_wait`. A missing
field means the live-path assumption is wrong for your vLLM build, and a full
sweep on that assumption wastes hours.

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
  --trajectory /mnt/pvc/admission-v2/trajectory.jsonl \
  --events     /mnt/pvc/admission-v2/admission_events.jsonl \
  --meta       /mnt/pvc/admission-v2/meta.json \
  --coeffs     coeffs.json \
  --estimator  token_rollforward \
  --out-rows   figures/ttft_rows.json \
  --out-censored figures/ttft_censored.json \
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
- A request enqueued but never scheduled is a right-censored observation, not a
  missing row with zero delay. The driver writes it to `ttft_censored.json` with
  the lower bound `capture_end - t_arrive` and excludes that bound from MAPE.

**Legacy reference only.** On the original schema-v1 capture,
`figures/plot_ttft_full.py` prints:

```
n 30734
oracle        (30734, 4.45, 7.89)
deploy all    (30734, 33.44, 57.52)
deploy nqueue (26622, 64.49, 52.96)
deploy queued (4112, -89.27, 86.99)
```

reading as count, median bias percent, MAPE percent.

Those values validate reproducibility of the old capture, not the six model
improvements above. Real vLLM accuracy for `token_rollforward` is intentionally
not claimed until the cluster experiment is rerun with schema 2.

---

## 8. Where the composed error lives

**How to read MAPE.** MAPE is the mean, over completed requests, of
`abs(predicted - realized) / realized * 100`. It is easy to compare across
latency scales, but it becomes unstable when realized admission delay is near
zero and it says nothing about right-censored requests. The reports retain MAPE
for comparison with the original figure, but also emit MAE, median and p90
absolute percentage error, WAPE (total absolute error divided by total realized
latency), signed median bias, and the fraction of censored predictions already
below their observed lower bounds. MAPE should not be the sole admission-model
claim; for composed TTFT it is useful alongside these tail and absolute metrics.

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

**Legacy probe diagnosis.** In the schema-v1 capture, realized admission delay on the un-queued subset has
median 11 microseconds, with 85% of requests under 0.1 ms and p99 at 1.44 ms. A
step occupies the GPU for about 20 ms, so a request arriving mid-iteration cannot
physically be admitted in 11 microseconds. If `t_enq` were an arrival instant
independent of the engine's rhythm, the delay would spread across the whole
iteration and the median would sit near 10 ms.

The explanation is that the old `t_enq` is synchronized to the step boundary.
vLLM V1's busy loop drains its input queue and then calls `schedule()`
immediately. Thus the legacy capture measured the scheduler-queue wait but
missed the preceding wait for an in-flight GPU iteration.

- **Part one** waits in the EngineCore input queue for the in-flight iteration to
  drain. Schema 1 cannot see it. Schema 2 stamps `t_engine_arrive` before that
  queue and retains `t_enq`, measuring both parts directly.
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

A correctly placed probe was therefore expected to put the deployable bulk
between roughly 6% and 23% rather than 53%, but that table is a counterfactual,
not a validation result. The schema-v2 rerun will replace it with a measured
number. The new estimator charges only the predicted residual of the in-flight
iteration, rather than always charging a full iteration.

**The legacy queued tail does not improve.** It stays near 87%, and zeroing the
admission term makes it slightly worse at 89.7%. That error survives every
counterfactual, so it is not a measurement artifact. The estimator sees the depth
of the waiting queue but not the volume of chunked-prefill work already committed
ahead of the request, so it cannot anticipate a multi-second stall. A router
trusting it will treat admission as nearly free when slots are open, which is
correct, and will underestimate the cost of routing into a prefill-saturated
pool, which is the documented blind spot.

`token_rollforward` addresses that blind spot by requiring an empty queue and
available token budget before the immediate path, retaining queued prompt/cache/
KV work, and simulating vLLM's actual running-first then FIFO-waiting scheduling
order. It recomputes batch composition and `T_iter` each step until the target is
admitted. The 3,083 schema-v1 requests that were enqueued but never scheduled
(1,608 in one process segment and 1,475 in the other) are now reported as
right-censored lower bounds instead of disappearing from tail reporting. None of
these changes has a new real-engine accuracy number until schema 2 is collected.

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
