#!/usr/bin/env bash
set -euo pipefail
# --w3 (or PAITON_W3=1) picks the 3-bit image: it downloads its 3-bit experts into the cache at first start, so nothing
# else is mounted. PAITON_W3X=<folder> (below, your own W3X folder) implies it. PAITON_IMAGE wins over both.
w3="${PAITON_W3:-0}"
args=()
for arg in "$@"; do
  if [[ $arg == --w3 ]]; then w3=1; else args+=("$arg"); fi
done
if [[ -n "${PAITON_W3X:-}" ]]; then w3=1; fi
if [[ $w3 == 1 ]]; then default_image=ghcr.io/eliovp/paiton-vllm-plugin:inkling-2-mini-w3-rdna4-v1.2.0@sha256:7f1eaad71457b00f548d97fa66af9389ee6730dc8e718339cfeee5d7ed3a449e
else default_image=ghcr.io/eliovp/paiton-vllm-plugin:inkling-2-mini-rdna4-v1.2.0@sha256:bc6b7b6d567f7afe4f40457be6b6a8d2f2d685365a263ae252a6f564a0b592cb; fi
image="${PAITON_IMAGE:-$default_image}"
cache="${PAITON_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/paiton-inkling-2-mini}"
port="${PAITON_PORT:-8042}"
# The API listens on every interface (0.0.0.0); PAITON_BIND=127.0.0.1 keeps it on this machine.
bind="${PAITON_BIND:-0.0.0.0}"
name="${PAITON_CONTAINER:-paiton-inkling-2-mini}"
mkdir -p "$cache"
extra=()
if [[ -n "${PAITON_HF_CACHE:-}" ]]; then
  extra+=(-v "$PAITON_HF_CACHE:/models/cache/huggingface:ro")
fi
# Opt-in: PAITON_API_KEY makes vLLM require that key on /v1 requests (passed by variable name, not as a literal argument).
if [[ -n "${PAITON_API_KEY:-}" ]]; then
  export VLLM_API_KEY="$PAITON_API_KEY"
  extra+=(-e VLLM_API_KEY)
fi
# Not exact: PAITON_W3X=<folder> serves the 3-bit experts of that W3X folder (launcher --w3x) instead of the 3-bit
# image's download. Its parent is mounted read-only, so a manifest whose files link into a sibling folder
# (../i3g64e_i3g64m/...) finds them.
w3x=()
if [[ -n "${PAITON_W3X:-}" ]]; then
  w3x_dir="$(cd "$PAITON_W3X" && pwd)"
  extra+=(-v "$(dirname "$w3x_dir"):/models/w3x:ro")
  w3x=(--w3x "/models/w3x/$(basename "$w3x_dir")")
fi
ipc=(--ipc=host)
# The disk tier of --extend-cache stages blocks in a /dev/shm file that only a clean shutdown removes: a private IPC
# namespace (its own /dev/shm, freed when the container stops) keeps a killed server from leaving it in the host's.
for arg in "${args[@]}"; do
  if [[ $arg == --extend-cache* ]]; then ipc=(--ipc=private --shm-size=13g); fi
done
# GPU_MAX_HW_QUEUES=1 is the setting of every RDNA4 launcher here (no measured difference for this model on ROCm 10.2).
exec docker run --rm --name "$name" --device /dev/kfd --device /dev/dri -e GPU_MAX_HW_QUEUES=1 \
  "${ipc[@]}" -p "$bind:$port:8042" -v "$cache:/models/cache" "${extra[@]}" "$image" "${w3x[@]}" "${args[@]}"
