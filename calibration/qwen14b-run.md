# Qwen3-14B TP=1 calibration and real-system experiment

This path calibrates the same Qwen3-14B, TP=1, vLLM 0.23.0 configuration used
by the successful 1P2D causal-routing run. It deliberately leaves the existing
Llama-70B calibration artifacts unchanged.

## Stage 0: fit the compute law on one H100

Use the dedicated `vramani-perfcal` namespace. The deployment disables prefix
caching so each requested prompt length corresponds to a known amount of
prefill work. Its scheduler geometry matches the P/D deployment: 2,048 tokens
per iteration, 16-token KV blocks, and 256 request slots.

```bash
oc -n vramani-perfcal create configmap calibrate-qwen14b-code \
  --from-file=client.py=calibration/client.py \
  --dry-run=client -o yaml | oc apply -f -

oc apply -f calibration/deploy/vllm-qwen14b-tp1.yaml
oc -n vramani-perfcal rollout status deploy/vllm-cal-qwen14b --timeout=30m

oc -n vramani-perfcal delete job calibrate-qwen14b-smoke --ignore-not-found
oc apply -f calibration/deploy/calibrate-qwen14b-smoke-job.yaml
oc -n vramani-perfcal wait --for=condition=complete \
  job/calibrate-qwen14b-smoke --timeout=20m

oc -n vramani-perfcal delete job calibrate-qwen14b --ignore-not-found
oc apply -f calibration/deploy/calibrate-qwen14b-job.yaml
oc -n vramani-perfcal wait --for=condition=complete \
  job/calibrate-qwen14b --timeout=90m
```

Fit the five coefficients from the captured CSVs:

The full manifest takes three independent samples for every prefill length and
three injections for every mixed batch size. `metadata.json` records the exact
grid and seed beside the CSVs.

```bash
oc -n vramani-perfcal delete pod qwen-cal-inspect --ignore-not-found
oc -n vramani-perfcal run qwen-cal-inspect --image=alpine:3.19 \
  --restart=Never --overrides='{"spec":{"containers":[{"name":"qwen-cal-inspect","image":"alpine:3.19","command":["sleep","3600"],"volumeMounts":[{"name":"results","mountPath":"/results"}]}],"volumes":[{"name":"results","persistentVolumeClaim":{"claimName":"cal-results-pvc"}}]}}'
oc -n vramani-perfcal wait --for=condition=Ready pod/qwen-cal-inspect --timeout=120s

mkdir -p calibration/results/qwen3-14b-tp1-vllm-0.23.0/stage0
oc -n vramani-perfcal cp \
  qwen-cal-inspect:/results/qwen3-14b-tp1-vllm-0.23.0/stage0/. \
  calibration/results/qwen3-14b-tp1-vllm-0.23.0/stage0

python3 -m calibration.report \
  --decode calibration/results/qwen3-14b-tp1-vllm-0.23.0/stage0/decode.csv \
  --prefill calibration/results/qwen3-14b-tp1-vllm-0.23.0/stage0/prefill.csv \
  --mixed calibration/results/qwen3-14b-tp1-vllm-0.23.0/stage0/mixed.csv \
  --chunk-bud 2048 \
  --out-dir calibration/results/qwen3-14b-tp1-vllm-0.23.0/stage0/fit
```

Scale the deployment down as soon as capture finishes:

```bash
oc -n vramani-perfcal scale deploy/vllm-cal-qwen14b --replicas=0
oc -n vramani-perfcal delete pod qwen-cal-inspect --ignore-not-found
```

Do not copy the coefficients into the routing policy until the following gates
pass:

- decode coefficients are nonnegative, with high cell-median R-squared and low
  MAPE;
- prefill R-squared is high across both one-chunk and multi-chunk prompts;
- the held-out mixed rows contain real prefill/decode overlap and have bounded
  error after removing the known leading-edge alignment artifact;
- repeated cells show that timing jitter is small relative to the effects being
  fitted.

## Other Qwen-specific inputs

The current Qwen configuration must also replace the Llama transfer geometry.
Qwen3-14B has 40 layers, 8 KV heads, head dimension 128, and two-byte KV
elements, so TP=1 stores

```text
2 * 40 * 8 * 128 * 2 = 163,840 bytes/token/GPU.
```

The successful P/D run independently supports that value: its two transfers
were about 2.34--2.38 GB for roughly 14.3--14.5k uncached tokens. Fit transfer
base latency and bandwidth from multiple prompt sizes rather than using the
two large transfers as a complete transfer model.

## From calibration to the real comparison

After the compute and transfer gates pass:

1. create a Qwen causal scenario with the fitted H100 coefficients,
   `kvBytesPerTokenPerGPU: 163840`, and measured NIXL transfer parameters;
2. add request-keyed logs for every candidate's input snapshot, predicted
   timing, resident externality, arriving-request value, and final score;
3. replay captured snapshots through the simulator and EPP scoring cores;
4. measure Qwen capacity for each compared policy using loose SLOs;
5. freeze offered rates and policy settings before the comparison;
6. run matched seeds at low, medium, and high fractions of capacity; and
7. report TTFT, mean ITL, E2E attainment, composite goodput, placement mix,
   transfer failures, and prediction error.

The minimum first comparison is the calibrated causal policy against the
stock llm-d decomposed policy on the same homogeneous-H100 1P2D topology. An
externality ablation and a decode-first version should follow before making the
paper's mechanism claim on the real system.
