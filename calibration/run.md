# calibration/run.md

## 0. Namespace + secret (dedicated, isolated from others' work)
# Namespace vramani-perfcal + hf-token-secret (copied from vllm-test) are already provisioned.
# To recreate: oc create namespace vramani-perfcal; copy hf-token-secret; then:

## 1. Deploy vLLM 70B TP4
oc apply -f deploy/results-pvc.yaml
oc apply -f deploy/model-cache-pvc.yaml
oc apply -f deploy/vllm-70b-tp4.yaml
# NOTE: the prefill fit assumes RTT ~= 0. calibrate-job.yaml pins the client to the vLLM node
# (podAffinity, topologyKey hostname), so client-side TTFT ~= server compute time; no RTT subtraction.
# Wait for weights + readiness (first run downloads ~140GB):
oc logs -f deploy/vllm-cal -n vramani-perfcal | grep -m1 "Application startup complete"
# Confirm no preemption/prefix-cache surprises:
oc logs deploy/vllm-cal -n vramani-perfcal | grep -iE "prefix cach|preempt|swap|GPU KV cache size"

## 1b. P1 GATE — verify SSE framing before trusting decode.csv (DO NOT SKIP)
# _stream_tokens timestamps every chunk whose choices[0].text is not None. If vLLM 0.11.0
# emits a TERMINAL empty-text chunk (or a chunk carrying only finish_reason), that adds one
# spurious timestamp per stream and corrupts the last decode gap in every (B,n) cell.
# Inspect the RAW frames on the live engine before running the sweep:
oc port-forward deploy/vllm-cal 8000:8000 -n vramani-perfcal &
PF_PID=$!; sleep 3
curl -sN http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"meta-llama/Llama-3.3-70B-Instruct","prompt":"Count to five:","max_tokens":5,"stream":true,"ignore_eos":true,"temperature":0}'
kill $PF_PID
# EXPECTED: exactly max_tokens (=5) frames each with a non-null "text", then "data: [DONE]".
# DECISION:
#   - If the LAST pre-[DONE] frame has "text": "" (empty) or text:null with a finish_reason,
#     the timestamp filter is too loose. Guard _stream_tokens: only append when
#     choices[0].get("finish_reason") is None (equivalently, drop the final frame).
#     Re-ship the ConfigMap after editing client.py.
#   - If you see exactly 5 non-empty text frames + [DONE], P1 is a non-issue; proceed unchanged.

## 2. Ship the client code as a ConfigMap and run the Job
oc create configmap calibrate-code --from-file=client.py=./client.py -n vramani-perfcal --dry-run=client -o yaml | oc apply -f -
oc apply -f deploy/calibrate-job.yaml
oc wait --for=condition=complete job/calibrate -n vramani-perfcal --timeout=3600s

## 3. Pull results
oc apply -f deploy/extractor.yaml
oc wait --for=condition=ready pod/cal-extractor -n vramani-perfcal --timeout=120s
oc cp vramani-perfcal/cal-extractor:/mnt/pvc/decode.csv ./decode.csv
oc cp vramani-perfcal/cal-extractor:/mnt/pvc/prefill.csv ./prefill.csv
oc cp vramani-perfcal/cal-extractor:/mnt/pvc/mixed.csv ./mixed.csv

## 4. Fit and report (local)
python report.py --decode decode.csv --prefill prefill.csv --mixed mixed.csv --chunk-bud 8192

## 5. Tear down GPUs when done
oc scale deploy/vllm-cal --replicas=0 -n vramani-perfcal
