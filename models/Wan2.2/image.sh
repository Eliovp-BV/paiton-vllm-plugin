#!/usr/bin/env bash
# Source from launch.sh or run.sh. This review package builds locally only.
prepare_wan_image() {
  export PAITON_WAN_BASE_IMAGE="${PAITON_WAN_BASE_IMAGE:-paiton-wan22-artifacts:local-v1}"
  export PAITON_WAN_IMAGE="${PAITON_WAN_IMAGE:-paiton-wan22:local-v1}"
  if [[ "${PAITON_WAN_BUILD:-0}" == 1 ]] || ! docker image inspect "$PAITON_WAN_BASE_IMAGE" >/dev/null 2>&1; then
    docker build --provenance=false -t "$PAITON_WAN_BASE_IMAGE" -f "$package_dir/Dockerfile" "$package_dir"
  fi
  if [[ "${PAITON_WAN_BUILD:-0}" == 1 ]] || ! docker image inspect "$PAITON_WAN_IMAGE" >/dev/null 2>&1; then
    docker build --provenance=false --build-arg "PAITON_WAN_BASE_IMAGE=$PAITON_WAN_BASE_IMAGE" \
      -t "$PAITON_WAN_IMAGE" -f "$package_dir/Dockerfile.local" "$package_dir"
  fi
}
