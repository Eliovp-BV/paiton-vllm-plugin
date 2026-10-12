#!/usr/bin/env bash

set -euo pipefail

IMAGE="ghcr.io/eliovp/paiton-vllm-plugin:ornith15-mxfp4-rdna4-v1.0.0"
CONTAINER_NAME="${PAITON_CONTAINER_NAME:-paiton-ornith}"
HOST_PORT="${PAITON_HOST_PORT:-8000}"
# The API listens on every interface (0.0.0.0); PAITON_BIND=127.0.0.1 keeps it on this machine.
BIND="${PAITON_BIND:-0.0.0.0}"
CACHE_VOLUME="${PAITON_CACHE_VOLUME:-paiton-ornith-cache}"
MODE="start"

if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [--chat]\n' "$0" >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    if [[ "$1" != "--chat" ]]; then
        printf 'Usage: %s [--chat]\n' "$0" >&2
        exit 2
    fi
    MODE="chat"
fi
# Opt-in: PAITON_API_KEY makes vLLM require that key on /v1 requests (passed by variable name, not as a literal argument).
key_args=()
if [[ -n "${PAITON_API_KEY:-}" ]]; then
    if [[ "$MODE" == "chat" ]]; then
        printf -- '--chat uses the terminal client in the image, which does not send an API key; unset PAITON_API_KEY for --chat.\n' >&2
        exit 2
    fi
    export VLLM_API_KEY="$PAITON_API_KEY"
    key_args=(--env VLLM_API_KEY)
fi

if [[ ! -e /dev/kfd || ! -d /dev/dri ]]; then
    printf 'ROCm devices /dev/kfd and /dev/dri are required.\n' >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    printf 'Docker is unavailable to this user.\n' >&2
    exit 1
fi
container_exists=0
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    container_exists=1
fi
if [[ "$container_exists" -eq 1 && "$MODE" == "start" ]]; then
    printf 'Container %s already exists. Remove or rename it first.\n' "$CONTAINER_NAME" >&2
    exit 1
fi

if [[ "$container_exists" -eq 0 ]]; then
    docker run -d \
        --name "$CONTAINER_NAME" \
        --device /dev/kfd \
        --device /dev/dri \
        --group-add video \
        --ipc=host \
        -p "${BIND}:${HOST_PORT}:8000" \
        "${key_args[@]}" \
        --mount "type=volume,src=${CACHE_VOLUME},dst=/models/cache" \
        "$IMAGE"
elif [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" != "true" ]]; then
    docker start "$CONTAINER_NAME" >/dev/null
fi

if [[ "$MODE" == "start" ]]; then
    printf 'Paiton Ornith is starting. Follow it with: docker logs -f %s\n' \
        "$CONTAINER_NAME"
    printf 'API: http://%s:%s/v1\n' "${BIND/#0.0.0.0/127.0.0.1}" "$HOST_PORT"
    if [[ "$BIND" == 0.0.0.0 ]]; then
        printf 'From another system: http://<this-machine-ip>:%s/v1\n' "$HOST_PORT"
    fi
    exit 0
fi

printf 'Waiting for Paiton Ornith to become ready'
for ((attempt = 1; attempt <= 3600; ++attempt)); do
    if docker exec "$CONTAINER_NAME" python3 -c \
        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" \
        >/dev/null 2>&1; then
        printf '\n'
        exec docker exec -it "$CONTAINER_NAME" paiton-chat --model ornith
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" != "true" ]]; then
        printf '\nContainer %s stopped before the server became ready.\n' \
            "$CONTAINER_NAME" >&2
        docker logs --tail 80 "$CONTAINER_NAME" >&2
        exit 1
    fi
    if ((attempt % 30 == 0)); then
        printf '.'
    fi
    sleep 1
done

printf '\nTimed out waiting for Paiton Ornith. Inspect: docker logs %s\n' \
    "$CONTAINER_NAME" >&2
exit 1
