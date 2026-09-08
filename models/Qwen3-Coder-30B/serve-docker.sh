#!/usr/bin/env bash
set -euo pipefail
cache_dir="${PAITON_CACHE_DIR:-${HOME}/.cache/paiton}"
mkdir -p "$cache_dir"
exec docker run --rm --name paiton-qwen3-coder --device /dev/kfd --device /dev/dri \
  --ipc=host -p 127.0.0.1:8010:8010 -v "$cache_dir:/models/cache" \
  "${PAITON_QWEN3_CODER_IMAGE:-paiton-qwen3-coder:local}" "$@"
