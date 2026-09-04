#!/usr/bin/env bash

set -euo pipefail

IMAGE="ghcr.io/eliovp/paiton-vllm-plugin:qwen38-qronos-rdna4-v1.3.0"
BASE_REPOSITORY="models--amd--Qwen3.8-27B-Quark-Qronos-INT4-W4A16"
BASE_REVISION="649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2"
CHECKPOINT_SIZE="19893384832"
VOLUME_NAME="paiton-qwen38-cache"
CONTAINER_NAME="${PAITON_CONTAINER_NAME:-paiton-qwen38}"

MODE="auto"
DRY_RUN=0
WITH_HF_TOKEN=0
HOST_PORT="${PAITON_HOST_PORT:-8000}"

usage() {
    cat <<'EOF'
Run Paiton Qwen3.8 on an RDNA4 gfx1201 GPU.

Usage:
  ./models/Qwen3.8/serve-docker.sh [option]

Options:
  --require-cache  Use only the exact model already in the host HF cache.
                   Exit before Docker starts if that cache is incomplete.
  --download       Ignore the host cache. Reuse the Docker volume or download
                   the original AMD checkpoint into it once.
  --with-hf-token  Explicitly pass your Hugging Face token for higher download
                   rate limits. The public model does not require a token.
  --dry-run        Show the checked paths and Docker command without running it.
  -h, --help       Show this help.

Environment:
  HF_HOME          Hugging Face home (default: ~/.cache/huggingface).
  HF_HUB_CACHE     Exact Hub cache directory (default: $HF_HOME/hub).
  HUGGINGFACE_HUB_CACHE
                   Legacy Hub cache variable, used only when HF_HUB_CACHE is unset.
  HF_TOKEN_PATH    Token file created by `hf auth login`; used only with
                   --with-hf-token.
  HF_TOKEN         Token value; used only with --with-hf-token and passed by
                   variable name, never as a literal command-line value.
  PAITON_HOST_PORT Host port for the API (default: 8000).
  PAITON_CONTAINER_NAME
                   Container name (default: paiton-qwen38).
EOF
}

while (($#)); do
    case "$1" in
        --require-cache)
            if [[ "$MODE" != "auto" && "$MODE" != "require-cache" ]]; then
                printf 'Conflicting options: --require-cache and --download\n' >&2
                exit 2
            fi
            MODE="require-cache"
            ;;
        --download)
            if [[ "$MODE" != "auto" && "$MODE" != "download" ]]; then
                printf 'Conflicting options: --require-cache and --download\n' >&2
                exit 2
            fi
            MODE="download"
            ;;
        --with-hf-token)
            WITH_HF_TOKEN=1
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

for command_name in docker realpath stat; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$command_name" >&2
        exit 1
    fi
done

if [[ -n "${HF_HOME:-}" ]]; then
    host_hf_home="$HF_HOME"
elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
    host_hf_home="${XDG_CACHE_HOME%/}/huggingface"
else
    host_hf_home="${HOME:?HOME is not set}/.cache/huggingface"
fi
if [[ -n "${HF_HUB_CACHE:-}" ]]; then
    host_hub_cache="$HF_HUB_CACHE"
elif [[ -n "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
    host_hub_cache="$HUGGINGFACE_HUB_CACHE"
else
    host_hub_cache="${host_hf_home%/}/hub"
fi
if [[ "$host_hub_cache" != /* ]]; then
    printf 'Hugging Face cache path must be absolute: %s\n' "$host_hub_cache" >&2
    printf 'Do not use an unexpanded path such as ~/cache inside an environment variable.\n' >&2
    exit 1
fi
snapshot_dir="${host_hub_cache%/}/${BASE_REPOSITORY}/snapshots/${BASE_REVISION}"

runtime_files=(
    chat_template.jinja
    crc32.txt
    generation_config.json
    merges.txt
    model.safetensors
    preprocessor_config.json
    processor_config.json
    tokenizer.json
    tokenizer_config.json
    video_preprocessor_config.json
    vocab.json
)

cache_complete=1
missing_files=()
for filename in "${runtime_files[@]}"; do
    if [[ ! -f "${snapshot_dir}/${filename}" ]]; then
        cache_complete=0
        missing_files+=("$filename")
    fi
done

if ((cache_complete)); then
    actual_size="$(stat -L -c '%s' "${snapshot_dir}/model.safetensors")"
    if [[ "$actual_size" != "$CHECKPOINT_SIZE" ]]; then
        cache_complete=0
        missing_files+=("model.safetensors has ${actual_size} bytes; expected ${CHECKPOINT_SIZE}")
    fi
fi

if ((!DRY_RUN)); then
    if [[ ! -e /dev/kfd || ! -d /dev/dri ]]; then
        printf 'ROCm devices were not found: /dev/kfd and /dev/dri are required.\n' >&2
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        printf 'Docker is installed, but its daemon is unavailable to this user.\n' >&2
        printf 'Start Docker and verify that docker info works without sudo.\n' >&2
        exit 1
    fi
fi

printf 'Hugging Face Hub cache: %s\n' "$host_hub_cache"
printf 'Required AMD revision:   %s\n' "$BASE_REVISION"

if ((cache_complete)); then
    printf 'Host checkpoint:        complete (19.9 GB)\n'
else
    printf 'Host checkpoint:        incomplete or not found\n'
fi

if [[ "$MODE" == "require-cache" ]] && ((!cache_complete)); then
    printf '\nCache-only mode stopped before Docker was started.\n' >&2
    printf 'Missing or invalid files under:\n  %s\n' "$snapshot_dir" >&2
    printf 'First issue: %s\n' "${missing_files[0]:-unknown}" >&2
    printf '\nIf your cache is elsewhere, run:\n' >&2
    printf '  HF_HUB_CACHE=/absolute/path/to/hub %q --require-cache\n' "$0" >&2
    exit 1
fi

use_host_cache=0
if [[ "$MODE" != "download" ]] && ((cache_complete)); then
    use_host_cache=1
fi

docker_args=(
    run --rm
    --name "$CONTAINER_NAME"
    --device /dev/kfd
    --device /dev/dri
    --group-add video
    --ipc=host
    -p "${HOST_PORT}:8000"
    --mount "type=volume,src=${VOLUME_NAME},dst=/models/cache"
)

if ((use_host_cache)); then
    host_model_repo="$(realpath -e "${host_hub_cache%/}/${BASE_REPOSITORY}")"
    docker_args+=(
        --mount "type=bind,src=${host_model_repo},dst=/models/host-base-repo,readonly"
        --env "PAITON_BASE_MODEL=/models/host-base-repo/snapshots/${BASE_REVISION}"
        --env HF_HUB_OFFLINE=1
    )
    printf 'Action:                 reuse host cache; network download disabled\n'
    printf 'Mounted cache scope:    only the AMD Qwen3.8 repository\n'
    if ((WITH_HF_TOKEN)); then
        printf 'Hugging Face token:     not mounted; cache-only mode does not need it\n'
    fi
else
    printf 'Action:                 reuse Docker volume, or download once if empty\n'
    printf 'Possible first download: up to 19.9 GB (file count may remain at 10/11)\n'

    if ((WITH_HF_TOKEN)); then
        if [[ -n "${HF_TOKEN:-}" ]]; then
            docker_args+=(--env HF_TOKEN)
            printf 'Hugging Face token:     explicitly enabled from HF_TOKEN\n'
        else
            configured_token_path="${HF_TOKEN_PATH:-${host_hf_home%/}/token}"
            if [[ "$configured_token_path" != /* ]]; then
                printf 'Hugging Face token path must be absolute: %s\n' \
                    "$configured_token_path" >&2
                exit 1
            fi
            if [[ ! -r "$configured_token_path" || ! -f "$configured_token_path" ]]; then
                printf 'No readable Hugging Face token was found at:\n  %s\n' \
                    "$configured_token_path" >&2
                printf 'Run hf auth login, or export HF_TOKEN, then retry with --with-hf-token.\n' >&2
                exit 1
            fi
            token_source="$(realpath -e "$configured_token_path")"
            docker_args+=(
                --mount "type=bind,src=${token_source},dst=/run/secrets/hf_token,readonly"
                --env HF_TOKEN_PATH=/run/secrets/hf_token
            )
            printf 'Hugging Face token:     explicitly enabled from a read-only file\n'
        fi
    else
        printf 'Hugging Face token:     not passed (anonymous public download)\n'
    fi
fi

docker_args+=("$IMAGE")

printf '\nThe server stays attached to this terminal. Open a second terminal for curl.\n'
printf 'Each cold start takes about ten minutes after any download completes.\n'
printf 'Wait for Application startup complete or a successful /health response.\n\n'

if ((DRY_RUN)); then
    printf 'Dry run; Docker was not started:\n'
    printf 'docker'
    printf ' %q' "${docker_args[@]}"
    printf '\n'
    exit 0
fi

docker "${docker_args[@]}"
