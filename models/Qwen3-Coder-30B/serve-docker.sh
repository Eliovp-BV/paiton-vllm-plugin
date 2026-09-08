#!/usr/bin/env bash
set -euo pipefail

image="${PAITON_QWEN3_CODER_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:qwen3-coder-30b-awq-rdna4-v1.0.0}"
container="${PAITON_CONTAINER_NAME:-paiton-qwen3-coder}"
port="${PAITON_HOST_PORT:-8010}"
cache_dir="${PAITON_CACHE_DIR:-${HOME}/.cache/paiton}"
chat=0
stock=0
download=0
server_args=()
for option in "$@"; do
    case "$option" in
        --chat) chat=1 ;;
        --stock) stock=1; server_args+=(--stock) ;;
        --offline) server_args+=(--offline) ;;
        --download-only) download=1; server_args+=(--download-only) ;;
        -h|--help)
            cat <<'HELP'
Run Qwen3-Coder 30B on one Radeon AI PRO R9700.
Usage: ./serve-docker.sh [--chat] [--stock] [--offline] [--download-only]

Default: start the API in the background; download the pinned model if needed.
--chat           Wait for readiness and open terminal chat; server stays running.
--stock          Run stock vLLM for the matched comparison.
--offline        Require the pinned model in the persistent cache.
--download-only  Populate the cache without loading the GPU model.

Environment: PAITON_CACHE_DIR (default ~/.cache/paiton), PAITON_HOST_PORT (8010),
PAITON_CONTAINER_NAME (paiton-qwen3-coder), PAITON_QWEN3_CODER_IMAGE.
HELP
            exit 0 ;;
        *) printf 'Unknown option: %s. Use --help.\n' "$option" >&2; exit 2 ;;
    esac
done
if [[ ! "$port" =~ ^[0-9]{1,5}$ ]] || ((10#$port < 1 || 10#$port > 65535)); then
    printf 'PAITON_HOST_PORT must be an integer from 1 to 65535.\n' >&2
    exit 2
fi
port="$((10#$port))"
if ((chat && download)); then
    printf -- '--chat and --download-only cannot be combined.\n' >&2
    exit 2
fi
if ! docker info >/dev/null 2>&1; then
    printf 'Docker is unavailable to this user.\n' >&2
    exit 1
fi
mkdir -p "$cache_dir"
cache_dir="$(cd "$cache_dir" && pwd -P)"
if ((download)); then
    exec docker run --rm --no-healthcheck -v "$cache_dir:/models/cache" "$image" "${server_args[@]}"
fi
if [[ ! -e /dev/kfd || ! -d /dev/dri ]]; then
    printf 'ROCm devices /dev/kfd and /dev/dri are required.\n' >&2
    exit 1
fi
mode=paiton
if ((stock)); then mode=stock; fi
if docker container inspect "$container" >/dev/null 2>&1; then
    existing_image="$(docker inspect -f '{{.Config.Image}}' "$container")"
    existing_mode="$(docker inspect -f '{{index .Config.Labels "dev.paiton.qwen3-coder.mode"}}' "$container")"
    if [[ "$existing_image" != "$image" || "$existing_mode" != "$mode" ]]; then
        printf 'Container %s uses a different image or mode. Choose PAITON_CONTAINER_NAME or remove it first.\n' "$container" >&2
        exit 1
    fi
    existing_port="$(docker inspect -f '{{(index (index .HostConfig.PortBindings "8010/tcp") 0).HostPort}}' "$container")"
    existing_cache="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/models/cache"}}{{.Source}}{{end}}{{end}}' "$container")"
    if [[ "$existing_port" != "$port" || "$existing_cache" != "$cache_dir" ]]; then
        printf 'Container %s uses a different port or cache. Reuse those settings or choose another container name.\n' "$container" >&2
        exit 1
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != true ]]; then
        docker start "$container" >/dev/null
    fi
else
    docker run -d --name "$container" --device /dev/kfd --device /dev/dri \
        --group-add video --ipc=host --label "dev.paiton.qwen3-coder.mode=$mode" \
        -p "127.0.0.1:$port:8010" -v "$cache_dir:/models/cache" "$image" "${server_args[@]}" >/dev/null
fi
if ((!chat)); then
    printf 'Qwen3-Coder is starting. API: http://127.0.0.1:%s/v1\n' "$port"
    printf 'Model: qwen3-coder. Logs: docker logs -f %s\n' "$container"
    printf 'Run this command again with --chat to open terminal chat.\n'
    exit 0
fi
printf 'Waiting for Qwen3-Coder; the first launch downloads 18.1 GB of weights.\n'
for ((attempt=0; attempt<3600; attempt++)); do
    if docker exec "$container" python3 -c \
        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=2)" >/dev/null 2>&1; then
        chat_flags=(-i)
        if [[ -t 0 && -t 1 ]]; then chat_flags=(-it); fi
        exec docker exec "${chat_flags[@]}" "$container" paiton-chat \
            --url http://127.0.0.1:8010/v1/chat/completions --model qwen3-coder \
            --temperature 0 --max-tokens 1024
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != true ]]; then
        docker logs --tail 60 "$container" >&2
        printf 'Qwen3-Coder stopped before it became ready.\n' >&2
        exit 1
    fi
    sleep 1
done
printf 'Timed out waiting for readiness. Inspect: docker logs %s\n' "$container" >&2
exit 1
