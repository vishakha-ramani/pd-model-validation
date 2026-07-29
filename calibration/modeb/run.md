# calibration/modeb/run.md

Mode-B runbook: deploy the instrumented vLLM 70B, sweep guidellm archetypes
against it, and pull the per-step trajectory for offline analysis.

Run every command below from this repository's root.

## Standing constraints

The operator drives every cluster action in this runbook. Nothing here runs
until the operator explicitly approves it and types the command.

Use the `nvidia.com/gpu` toleration only. Never add an
`llm-d-benchmark-harness` toleration or taint. Never uncordon a node. These
manifests are authored and validated locally only. No command in this file
has been run against a cluster.

## 1. Namespace, secret, and results PVC

The namespace `vramani-perfcal`, the secret `hf-token-secret`, and the PVC
`cal-results-pvc` already exist from the base calibration run (see
`calibration/run.md`). Recreate them only if they are missing.

```bash
oc create namespace vramani-perfcal
# copy hf-token-secret into vramani-perfcal from wherever it is provisioned
oc apply -f calibration/deploy/results-pvc.yaml
```

## 2. Deploy the ConfigMap, then the Mode-B vLLM deployment

```bash
oc apply -f calibration/deploy/modeb-configmap.yaml
oc apply -f calibration/deploy/vllm-70b-tp4-modeb.yaml
oc logs -f deploy/vllm-cal -n vramani-perfcal | grep -m1 "Application startup complete"
```

## 3. Pre-flight gate

Do not proceed until both checks below pass.

The hook must confirm it patched the live scheduler loop.

```bash
oc logs deploy/vllm-cal -n vramani-perfcal | grep MODEB
```

Expect a line like `hook active in pid ... patched Scheduler.schedule +
EngineCore.step`. If this line is missing, the sitecustomize hook did not
load in the EngineCore child process. Stop and diagnose before running any
sweep.

The captured scheduler config must match what the deployment asked for.

```bash
oc exec deploy/vllm-cal -n vramani-perfcal -- cat /results/modeb/meta.json
```

Confirm `async_scheduling` is `false` and `max_num_batched_tokens` is `8192`.
If either value is wrong, the trajectory will not correspond to the intended
run and the sweep results will not be usable.

## 4. Sanity probe

Send a single 256-token request and inspect the raw trajectory before
committing GPU time to the full sweep.

```bash
oc port-forward deploy/vllm-cal 8000:8000 -n vramani-perfcal &
PF_PID=$!; sleep 3
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"meta-llama/Llama-3.3-70B-Instruct","prompt":"Count to five:","max_tokens":256,"temperature":0}'
kill $PF_PID
oc exec deploy/vllm-cal -n vramani-perfcal -- tail -n 5 /results/modeb/trajectory.jsonl
```

Check the rows by hand. A decode-only step's `t_iter` (the gap between
consecutive `t_start` values) should land near 16 ms. Every request's
`kappa`, `computed`, and `prompt_len` should be small non-negative integers
with `computed <= prompt_len` during prefill and `kappa` equal to 1 per
request during steady decode. If any row looks physically implausible, stop
and diagnose before running the sweep.

## 5. Run the archetype sweep

```bash
bash calibration/modeb/run_sweep.sh
```

This script only prints the five `oc apply` lines, one per archetype
(decode-corner, balanced, prefill-lean, prefill-bound, conversation). The
operator reviews each line and pipes it to `oc apply -n vramani-perfcal -f -`
by hand.

## 6. Pull results

```bash
oc apply -f calibration/deploy/extractor.yaml
oc wait --for=condition=ready pod/cal-extractor -n vramani-perfcal --timeout=120s
oc cp vramani-perfcal/cal-extractor:/mnt/pvc/modeb/trajectory.jsonl ./trajectory.jsonl
oc cp vramani-perfcal/cal-extractor:/mnt/pvc/modeb/meta.json ./meta.json
oc cp vramani-perfcal/cal-extractor:/mnt/pvc/guidellm ./guidellm-modeb-results
```

## 7. Analyze offline

`calibration/modeb/analysis.py` is a function library, not a CLI script, so
call it directly. `coeffs.json` is the fit produced by the base calibration
pipeline (see `calibration/RESULTS.md`).

```bash
python - <<'PY'
from calibration.modeb.analysis import load_coeffs, parse_trajectory, write_report

coeffs = load_coeffs("coeffs.json")
steps = parse_trajectory("trajectory.jsonl")
write_report(steps, coeffs, out_dir=".")
PY
```

This produces `modeb_report.json` and `modeb_pred_vs_meas.png` in the
current directory. No cluster access is needed for this step.

## 8. Tear down

```bash
oc scale deploy/vllm-cal --replicas=0 -n vramani-perfcal
```

Scale back to zero replicas as soon as the sweep and extraction finish. Do
not leave the GPUs reserved.
