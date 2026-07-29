#!/usr/bin/env bash
# Launch one guidellm sweep Job per Mode-B archetype. Requires the vLLM
# Mode-B deployment to be Ready and the sitecustomize sentinel confirmed in logs.
# Does NOT deploy vLLM and does NOT apply anything without the operator running it.
set -euo pipefail
NS=vramani-perfcal
TPL=calibration/deploy/guidellm-sweep-modeb-job.yaml

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
  outdir="/results/guidellm/modeb-${name}-i${isl}-o${osl}"
  echo "# ${name} (ISL ${isl} / OSL ${osl})"
  echo "sed -e 's#__GUIDELLM_DATA__#${data}#' -e 's#__GUIDELLM_OUTPUT_DIR__#${outdir}#' -e 's#guidellm-sweep-modeb#guidellm-sweep-modeb-${name}#' ${TPL} | oc apply -n ${NS} -f -"
done
echo "# NOTE: this script only prints. The operator reviews each command above and runs it after approving GPU use."
