#!/usr/bin/env bash

set -euo pipefail

IMAGE="ghcr.io/eliovp/paiton-vllm-plugin:ornith15-mxfp4-rdna4-v1.0.0"
CONTAINER_NAME="${PAITON_CONTAINER_NAME:-paiton-ornith}"
HOST_PORT="${PAITON_HOST_PORT:-8000}"
CACHE_VOLUME="${PAITON_CACHE_VOLUME:-paiton-ornith-cache}"

if [[ ! -e /dev/kfd || ! -d /dev/dri ]]; then
    printf 'ROCm devices /dev/kfd and /dev/dri are required.\n' >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    printf 'Docker is unavailable to this user.\n' >&2
    exit 1
fi
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    printf 'Container %s already exists. Remove or rename it first.\n' "$CONTAINER_NAME" >&2
    exit 1
fi

exec docker run -d \
    --name "$CONTAINER_NAME" \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --ipc=host \
    -p "${HOST_PORT}:8000" \
    --mount "type=volume,src=${CACHE_VOLUME},dst=/models/cache" \
    "$IMAGE"
