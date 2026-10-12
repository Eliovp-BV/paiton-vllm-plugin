#!/usr/bin/env bash
set -euo pipefail
image=${PAITON_NEO_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin@sha256:534287969135f581744ae481b578599468b0bf7ac9a4051b0941500e4c18da4d}
port=${PAITON_NEO_PORT:-8000}
cache=${PAITON_NEO_CACHE:-paiton-qwen38-neo-cache}
# The API listens on every interface (0.0.0.0); PAITON_BIND=127.0.0.1 keeps it on this machine.
bind=${PAITON_BIND:-0.0.0.0}
extra=()
# Opt-in: PAITON_API_KEY makes vLLM require that key on /v1 requests (passed by variable name, not as a literal argument).
if [[ -n "${PAITON_API_KEY:-}" ]]; then
  export VLLM_API_KEY="$PAITON_API_KEY"
  extra+=(-e VLLM_API_KEY)
fi
exec docker run --rm --name paiton-qwen38-neo \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -p "${bind}:${port}:8000" "${extra[@]}" \
  --mount "type=volume,src=${cache},dst=/models/cache" \
  "$image" "$@"
