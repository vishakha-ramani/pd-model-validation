#!/usr/bin/env bash
# Launch one guidellm sweep Job per admission-validation archetype, then the
# fixed-rate overload Job. Requires the vLLM admission deployment to be Ready
# and the sitecustomize sentinel confirmed in logs.
# Does NOT deploy vLLM and does NOT apply anything without the operator running it.
set -euo pipefail
NS=vramani-perfcal
TPL=calibration/deploy/guidellm-sweep-admission-job.yaml
OVERLOAD_TPL=calibration/deploy/guidellm-overload-admission-job.yaml

# name:ISL:OSL
ARCHETYPES=(
  "decode-corner:256:512"
  "balanced:2048:128"
  "prefill-lean:8192:64"
  "prefill-bound:16000:16"
  "conversation:1024:256"
)

for a in "${ARCHETYPES[@]}"; do
  IFS=: read -r name isl osl <<<"$a"
  data="prompt_tokens=${isl},prompt_tokens_min=${isl},prompt_tokens_max=${isl},output_tokens=${osl},output_tokens_min=${osl},output_tokens_max=${osl},random_seed=0"
  outdir="/results/guidellm-admission/${name}-i${isl}-o${osl}"
  echo "# ${name} (ISL ${isl} / OSL ${osl})"
  echo "sed -e 's#__GUIDELLM_DATA__#${data}#' -e 's#__GUIDELLM_OUTPUT_DIR__#${outdir}#' -e 's#guidellm-sweep-admission#guidellm-sweep-admission-${name}#' ${TPL} | oc apply -n ${NS} -f -"
done

# Overload extension: fixed --profile constant rate ABOVE the sweep's observed
# peak throughput for decode-corner, sustained long enough (>= sweep
# GUIDELLM_MAX_SECONDS + margin) for a standing waiting-queue backlog to form.
# Set OVERLOAD_RATE (requests/sec) from the sweep results before running this
# line; the placeholder below is a reminder, not a usable value.
rate="${OVERLOAD_RATE:-REPLACE_ME_RATE_ABOVE_SWEEP_PEAK}"
echo "# overload (decode-corner, ISL 256 / OSL 512, fixed rate ${rate} req/s)"
echo "sed -e 's#__GUIDELLM_RATE__#${rate}#' ${OVERLOAD_TPL} | oc apply -n ${NS} -f -"

echo "# NOTE: this script only prints. The operator reviews each command above and runs it after approving GPU use."
echo "# NOTE: run the overload line only after inspecting the sweep results and setting OVERLOAD_RATE above the observed peak."
