#!/usr/bin/env bash
set -euo pipefail

image="${PAITON_KIMI31_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:kimi31pro-rocm10.1.0-pytorch2.13.0-cdna4-v1}"
container="${PAITON_CONTAINER_NAME:-paiton-kimi31}"
port="${PAITON_HOST_PORT:-8031}"
# The API listens on every interface (0.0.0.0); PAITON_BIND=127.0.0.1 keeps it on this machine.
bind="${PAITON_BIND:-0.0.0.0}"
model_dir="${PAITON_MODEL_DIR:-${PAITON_CACHE_DIR:-${HOME}/.cache/paiton}/Kimi-K3.1-Pro}"
gpus="${PAITON_GPUS:-0,1,2,3}"
max_len="${PAITON_MAX_MODEL_LEN:-32768}"
revision=d9e99b0792a5e93b78ea9bea62a68167e36ddd9e
wait=0
download=0
for option in "$@"; do
    case "$option" in
        --wait) wait=1 ;;
        --download-only) download=1 ;;
        -h|--help)
            cat <<'HELP'
Run the Kimi-K3.1-Pro community build (plain PyTorch, slow) on 4 or more AMD Instinct CDNA4 GPUs.
Usage: ./serve-docker.sh [--wait] [--download-only]

Default: start the API in the background. The checkpoint (1.56 TB) must already be in PAITON_MODEL_DIR.
--download-only  Download the pinned checkpoint into PAITON_MODEL_DIR and exit.
--wait           Wait until the server is ready (loading takes 6 to 25 minutes).

Environment: PAITON_MODEL_DIR (default ~/.cache/paiton/Kimi-K3.1-Pro), PAITON_GPUS (0,1,2,3),
PAITON_HOST_PORT (8031), PAITON_BIND (0.0.0.0, every interface; 127.0.0.1 keeps the API on this machine),
PAITON_API_KEY (opt-in: require this key on /v1 requests), PAITON_MAX_MODEL_LEN (32768),
PAITON_CONTAINER_NAME (paiton-kimi31), PAITON_KIMI31_IMAGE.
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
if [[ ! "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    printf 'PAITON_GPUS must be a comma-separated list of GPU indices, for example 0,1,2,3.\n' >&2
    exit 2
fi
if ! docker info >/dev/null 2>&1; then
    printf 'Docker is unavailable to this user.\n' >&2
    exit 1
fi
mkdir -p "$model_dir"
model_dir="$(cd "$model_dir" && pwd -P)"
if ((download)); then
    printf 'Downloading moonshotai/Kimi-K3.1-Pro (1.56 TB) into %s\n' "$model_dir"
    exec docker run --rm -v "$model_dir:/model" --entrypoint hf "$image" \
        download moonshotai/Kimi-K3.1-Pro --revision "$revision" --local-dir /model
fi
if [[ ! -f "$model_dir/config.json" || ! -f "$model_dir/model.safetensors.index.json" ]]; then
    printf 'No Kimi-K3.1-Pro checkpoint in %s. Run ./serve-docker.sh --download-only first, or set PAITON_MODEL_DIR.\n' \
        "$model_dir" >&2
    exit 1
fi
if [[ ! -e /dev/kfd || ! -d /dev/dri ]]; then
    printf 'ROCm devices /dev/kfd and /dev/dri are required.\n' >&2
    exit 1
fi
key_args=()
if [[ -n "${PAITON_API_KEY:-}" ]]; then
    export PAITON_API_KEY
    key_args=(-e PAITON_API_KEY)
fi
if docker container inspect "$container" >/dev/null 2>&1; then
    printf 'Container %s already exists. Remove it with "docker rm -f %s" or choose PAITON_CONTAINER_NAME.\n' \
        "$container" "$container" >&2
    exit 1
fi
docker run -d --name "$container" --device /dev/kfd --device /dev/dri --group-add video --ipc=host --shm-size 16g \
    -e HIP_VISIBLE_DEVICES="$gpus" -p "$bind:$port:8000" "${key_args[@]}" -v "$model_dir:/model:ro" \
    "$image" --max-model-len "$max_len" >/dev/null
printf 'Kimi-K3.1-Pro is loading on GPUs %s; this takes 6 to 25 minutes. API: http://%s:%s/v1\n' \
    "$gpus" "${bind/#0.0.0.0/127.0.0.1}" "$port"
if [[ "$bind" == 0.0.0.0 ]]; then
    printf 'From another system: http://<this-machine-ip>:%s/v1\n' "$port"
fi
printf 'Model: kimi-k3.1-pro. Logs: docker logs -f %s\n' "$container"
if ((!wait)); then
    exit 0
fi
for ((attempt=0; attempt<1800; attempt++)); do
    if docker logs "$container" 2>&1 | grep -q '^serving '; then
        printf 'Ready.\n'
        exit 0
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != true ]]; then
        docker logs --tail 60 "$container" >&2
        printf 'Kimi-K3.1-Pro stopped before it became ready.\n' >&2
        exit 1
    fi
    sleep 2
done
printf 'Timed out waiting for readiness. Inspect: docker logs %s\n' "$container" >&2
exit 1
