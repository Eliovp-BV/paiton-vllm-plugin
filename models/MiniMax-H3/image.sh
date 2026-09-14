#!/usr/bin/env bash
# Shared local image preparation. Sourced by run.sh and launch.sh.
prepare_h3_image() {
  export PAITON_H3_BASE_IMAGE="${PAITON_H3_BASE_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:minimax-h3-rdna4-v1.0.0}"
  export PAITON_H3_IMAGE="${PAITON_H3_IMAGE:-paiton-minimax-h3:local-v1.0.1}"
  if [[ "${PAITON_H3_BUILD:-0}" == 1 ]]; then
    docker build --provenance=false -t "$PAITON_H3_BASE_IMAGE" -f "$package_dir/Dockerfile" "$package_dir"
  elif ! docker image inspect "$PAITON_H3_BASE_IMAGE" >/dev/null 2>&1; then
    docker pull "$PAITON_H3_BASE_IMAGE"
  fi
  if [[ "${PAITON_H3_BUILD:-0}" == 1 ]] || ! docker image inspect "$PAITON_H3_IMAGE" >/dev/null 2>&1; then
    printf 'Preparing the local ComfyUI image from pinned upstream source (first launch only).\n'
    docker build --provenance=false --build-arg "PAITON_H3_BASE_IMAGE=$PAITON_H3_BASE_IMAGE" \
      -t "$PAITON_H3_IMAGE" -f "$package_dir/Dockerfile.local" "$package_dir"
  fi
}
