#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -eq 0 || ${1:-} == --help ]]; then
  echo 'Usage: run-docker.sh [--stock] RECORDING NEW_OUTPUT_DIRECTORY [process options]'
  echo 'Set PAITON_MEETING_CACHE to the prepared cache and PAITON_MEETING_IMAGE to a local image.'
  exit 0
fi
backend=(--artifact /opt/paiton/meeting-artifacts/meeting_lstm_float16_gfx1201.so)
if [[ ${1:-} == --stock ]]; then backend=(); shift; fi
if [[ $# -lt 2 ]]; then echo 'A recording and new output directory are required.' >&2; exit 2; fi
recording=$(realpath -e -- "$1")
output=$(realpath -m -- "$2")
shift 2
if [[ ! -f $recording || -e $output || -L $output ]]; then
  echo 'Choose an existing recording and a new output directory; existing files are preserved.' >&2
  exit 2
fi
cache=${PAITON_MEETING_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/paiton-meeting}
models=$(realpath -e -- "$cache/models")
mkdir -p -- "$cache/runtime" "$(dirname -- "$output")"
runtime_cache=$(realpath -e -- "$cache/runtime")
image=${PAITON_MEETING_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:meeting-rdna4-v1.0.0-rc1}
exec flock "/tmp/paiton-studio-gpu-$(id -u).lock" docker run --rm --pull=never \
  --name "paiton-meeting-$(id -u)-$$" --network none \
  --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" \
  --user "$(id -u):$(id -g)" --shm-size 2g -e HOME=/tmp \
  -e OMP_NUM_THREADS=1 -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
  -v "$recording:/recording/input:ro" -v "$models:/models/meeting:ro" \
  -v "$runtime_cache:/models/cache" -v "$(dirname -- "$output"):/results" \
  "$image" process /recording/input --models /models/meeting \
  --output "/results/$(basename -- "$output")" "${backend[@]}" "$@"
