# calibration/admission/run.md

Admission-validation runbook. Deploy the admission-instrumented vLLM 70B.
Sweep guidellm archetypes against it. Run one fixed-rate overload job on top
of the sweep. Pull the trajectory and enqueue-event files. Analyze them
offline with the Task 4 and Task 5 estimators.

Run every command below from this repository's root.

## Standing constraints

The operator drives every cluster action in this runbook. Nothing here runs
until the operator explicitly approves it and types the command.

Use the `nvidia.com/gpu` toleration only. Never add an
`llm-d-benchmark-harness` toleration or taint. Never uncordon a node. Never
kill another tenant's GPU job to make room for this one. These manifests are
authored and validated locally only. No command in this file has been run
against a cluster.

Scale `vllm-cal` back to 0 replicas as soon as the sweep, the overload job,
and the extraction finish. Do not leave the GPUs reserved.

## 1. Namespace, secret, and results PVC

The namespace `vramani-perfcal`, the secret `hf-token-secret`, and the PVC
`cal-results-pvc` already exist from the base calibration and Mode-B runs
(see `calibration/run.md` and `calibration/modeb/run.md`). Recreate them
only if they are missing.

```bash
oc create namespace vramani-perfcal
# copy hf-token-secret into vramani-perfcal from wherever it is provisioned
oc apply -f calibration/deploy/results-pvc.yaml
```

## 2. Deploy the ConfigMap, then the admission vLLM deployment

If the Mode-B deployment is still running, scale it to 0 first. The
admission deployment reuses the same `vllm-cal` Deployment and Service
names, so the two configurations do not coexist.

```bash
oc scale deploy/vllm-cal --replicas=0 -n vramani-perfcal 2>/dev/null || true
oc apply -f calibration/deploy/admission-configmap.yaml
oc apply -f calibration/deploy/vllm-70b-tp4-admission.yaml
oc logs -f deploy/vllm-cal -n vramani-perfcal | grep -m1 "Application startup complete"
```

## 3. Pre-flight gate

Do not proceed until both checks below pass.

The hook must confirm it patched the live scheduler loop.

```bash
oc logs deploy/vllm-cal -n vramani-perfcal | grep ADMISSION
```

Expect a line like `hook active in pid ... patched Scheduler.add_request +
Scheduler.schedule + EngineCore.step`. If this line is missing, the
sitecustomize hook did not load in the EngineCore child process. Stop and
diagnose before running any sweep.

The captured scheduler config must match the run this capture assumes.

```bash
oc exec deploy/vllm-cal -n vramani-perfcal -- cat /results/admission/meta.json
```

Confirm `async_scheduling` is `false`. The offline replay assumes
synchronous single-step scheduling, the same discipline Mode B captured.
Confirm `block_size` is present and a positive integer. The offline
reconstruction of per-request KV-block counts divides by `block_size`, so a
missing or zero value makes every downstream estimate wrong. Stop and
diagnose if either check fails.

## 4. Sanity probe

Send a single 256-token request and inspect the raw trajectory before
committing GPU time to the full sweep and the overload job.

```bash
oc port-forward deploy/vllm-cal 8000:8000 -n vramani-perfcal &
PF_PID=$!; sleep 3
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"meta-llama/Llama-3.3-70B-Instruct","prompt":"Count to five:","max_tokens":256,"temperature":0}'
kill $PF_PID
oc exec deploy/vllm-cal -n vramani-perfcal -- tail -n 5 /results/admission/trajectory.jsonl
oc exec deploy/vllm-cal -n vramani-perfcal -- tail -n 5 /results/admission/admission_events.jsonl
```

Check the rows by hand. Every `trajectory.jsonl` row must carry a
`waiting_count` field and a `free_kv_blocks` field, not just the Mode-B
running-batch fields. Every `admission_events.jsonl` row must carry a
`t_enq` field alongside `req_id` and `prompt_len`. A missing field here
means the live-path guess in `admission_capture.py` or `sitecustomize.py`
is wrong for this vLLM build. Stop and diagnose before running the sweep.
Running the full sweep on a wrong live-path guess wastes GPU time and
produces a trajectory the offline analysis cannot use.

## 5. Run the archetype sweep

```bash
bash calibration/admission/run_sweep.sh
```

This script prints six lines. The first five are the guidellm sweep
archetypes (decode-corner, balanced, prefill-lean, prefill-bound,
conversation). The sixth is the fixed-rate overload job, covered in the
next section. The operator reviews each line and pipes it to
`oc apply -n vramani-perfcal -f -` by hand.

Run only the first five lines now. Wait for all five sweep Jobs to
complete before moving to the overload job.

```bash
oc get jobs -n vramani-perfcal -l experiment=cal-validate-70b -w
```

## 6. Run the overload job

The overload job drives one fixed-rate load, above the sweep's observed
peak throughput for the decode-corner archetype, so a standing
waiting-queue backlog forms. It targets the same `vllm-cal` deployment, so
`trajectory.jsonl` and `admission_events.jsonl` keep accumulating across
the sweep and the overload run without a restart.

Before applying the overload job, find the sweep's peak achieved rate for
decode-corner from its guidellm output, then pick a fixed rate clearly
above it.

```bash
oc exec deploy/vllm-cal -n vramani-perfcal -- \
  cat /results/guidellm-admission/decode-corner-i256-o512/benchmarks.csv
```

Note the boundary between sweep traffic and overload traffic before you
submit the overload job. This boundary is what later separates the
"sub_capacity" rows from the "overload" rows in the offline report.

```bash
oc exec deploy/vllm-cal -n vramani-perfcal -- \
  wc -l /results/admission/admission_events.jsonl
```

Record this line count as `N_BOUNDARY`. Every enqueue event at or before
this line belongs to the sweep. Every enqueue event after this line
belongs to the overload run.

Now set `OVERLOAD_RATE` to the fixed request rate chosen above and rerun
the launcher. It reprints the same six lines with the overload line's
`__GUIDELLM_RATE__` placeholder substituted.

```bash
# replace 12 below with your chosen rate, in requests/sec, above the sweep's
# observed decode-corner peak
OVERLOAD_RATE=12 bash calibration/admission/run_sweep.sh
```

Apply only the sixth (overload) line. Let it run for its full
`GUIDELLM_MAX_SECONDS` window before pulling results. The window is set
longer than the sweep's per-archetype window so the backlog has time to
build and sustain past queue saturation.

## 7. Pull results

```bash
oc apply -f calibration/deploy/extractor.yaml
oc wait --for=condition=ready pod/cal-extractor -n vramani-perfcal --timeout=120s
oc cp vramani-perfcal/cal-extractor:/mnt/pvc/admission/trajectory.jsonl ./trajectory.jsonl
oc cp vramani-perfcal/cal-extractor:/mnt/pvc/admission/admission_events.jsonl ./admission_events.jsonl
oc cp vramani-perfcal/cal-extractor:/mnt/pvc/admission/meta.json ./meta.json
oc cp vramani-perfcal/cal-extractor:/mnt/pvc/guidellm-admission ./guidellm-admission-results
```

## 8. Analyze offline

`calibration/admission/analysis.py` is a function library, not a CLI
script, so call it directly. `coeffs.json` is the fit produced by the base
calibration pipeline (see `calibration/RESULTS.md`). Use the same
`N_BOUNDARY` you recorded in section 6.

```bash
python - <<'PY'
try:
    from modeb.analysis import load_coeffs
except ImportError:
    from calibration.modeb.analysis import load_coeffs
from calibration.admission.analysis import (
    load_meta, parse_admission_trajectory, parse_enqueue_events, request_traces,
    replay, admission_report, write_admission_report, block_accounting_diag,
)

coeffs = load_coeffs("coeffs.json")
meta = load_meta("meta.json")
steps = parse_admission_trajectory("trajectory.jsonl")
enq_events = parse_enqueue_events("admission_events.jsonl")
traces = request_traces(steps)

N_BOUNDARY = 0  # fill in with the line count recorded in section 6
enq_order = {rid: i for i, rid in enumerate(enq_events)}

def load_of(req_id):
    return "overload" if enq_order[req_id] >= N_BOUNDARY else "sub_capacity"

rows_by_key = {}
for estimator_name in ("fluid", "rollforward"):
    for use_oracle in (True, False):
        variant = "oracle" if use_oracle else "deployable"
        rows, never_scheduled = replay(steps, enq_events, traces, meta, coeffs,
                                        estimator_name, use_oracle, load_of)
        rows_by_key[(estimator_name, variant)] = rows
        print(estimator_name, variant, "rows:", len(rows),
              "never_scheduled:", never_scheduled)

diag_rows = [block_accounting_diag(s, meta) for s in steps]
report = admission_report(rows_by_key, diag_rows=diag_rows)
paths = write_admission_report(report, rows_by_key, out_dir=".")
print(paths)
PY
```

This produces `admission_report.json` and, if matplotlib is importable,
`admission_pred_vs_realized.png` in the current directory. No cluster
access is needed for this step. The report carries the sub_capacity and
overload rows for the fluid and rollforward estimators and both variants,
plus a block-accounting diagnostic comparing the captured free-KV-block
count against the count reconstructed from per-request `computed` tokens.

`replay` returns a `(rows, never_scheduled_count)` tuple for each
estimator and variant. A nonzero `never_scheduled_count` under the
overload job is expected and is the defining symptom of a standing
backlog. It counts requests that sat in the waiting queue for the entire
capture window and never entered the running batch.

The harness intentionally excludes the occupancy-blind `waiting`
estimator from this report. A faithful offline `waiting` prediction needs
the trained-physics `muDecode`/`muPrefill` values and a waiting-backlog
work sum ported from `edpp.go`. This harness does not port either one.
The report validates only the occupancy-aware `fluid` and `rollforward`
estimators.

A request enqueued between two scheduler snapshots is assigned the
nearest snapshot's observed `waiting_count` as its queue position. Finer
position is not observable at snapshot granularity. This is an
acknowledged fidelity limit. It biases toward slight over-prediction
rather than under-prediction.

## 9. Tear down

```bash
oc delete pod/cal-extractor -n vramani-perfcal
oc scale deploy/vllm-cal --replicas=0 -n vramani-perfcal
```

Scale back to zero replicas as soon as the sweep, the overload job, and
the extraction finish. Do not leave the GPUs reserved.
