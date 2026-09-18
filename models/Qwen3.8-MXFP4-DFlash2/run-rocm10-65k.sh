#!/usr/bin/env bash
set -euo pipefail
# Serves http://127.0.0.1:18982/v1; select one context profile at a time.
# Benchmarked requests set chat_template_kwargs.enable_thinking=false.
# First startup loads/converts weights and compiles; measure after warmup.
: "${PAITON_TARGET_DIR:?Set PAITON_TARGET_DIR to the pinned Unsloth target snapshot}"
: "${PAITON_DRAFT_DIR:?Set PAITON_DRAFT_DIR to the qualified DFlash draft weights}"
: "${PAITON_CACHE_DIR:?Set PAITON_CACHE_DIR to a persistent writable cache directory}"
exec docker run --rm --name paiton-qwen38-65k --network host \
  --device /dev/kfd --device /dev/dri --shm-size 2g \
  -v "$PAITON_TARGET_DIR:/models/target:ro" \
  -v "$PAITON_DRAFT_DIR:/models/draft:ro" \
  -v "$PAITON_CACHE_DIR:/cache:rw" \
  ghcr.io/eliovp-bv/paiton-vllm-plugin:qwen38-rocm10-vllm029-65k-20260918-r2
