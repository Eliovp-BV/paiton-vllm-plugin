#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-start}" in
  --help|-h) printf 'Usage: %s [--build|--stop|--logs]\n' "$0"; exit 0 ;;
  start|--build|--stop|--logs) ;;
  *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
esac
command -v docker >/dev/null || { printf 'Install Docker with the Compose plugin first.\n' >&2; exit 1; }
docker info >/dev/null 2>&1 || { printf 'Docker is unavailable for this user.\n' >&2; exit 1; }
docker compose version >/dev/null
[[ -e /dev/kfd && -d /dev/dri ]] || { printf 'ROCm GPU devices are required.\n' >&2; exit 1; }
export PAITON_H3_DATA="${PAITON_H3_DATA:-${XDG_DATA_HOME:-$HOME/.local/share}/paiton/minimax-h3}"
export PAITON_H3_OUTPUTS="${PAITON_H3_OUTPUTS:-$HOME/paiton-videos}"
mkdir -p "$PAITON_H3_DATA/user" "$PAITON_H3_DATA/custom_nodes" "$PAITON_H3_OUTPUTS"
export PAITON_H3_DATA="$(realpath "$PAITON_H3_DATA")"
export PAITON_H3_OUTPUTS="$(realpath "$PAITON_H3_OUTPUTS")"
export PAITON_RENDER_GID="$(stat -c '%g' /dev/kfd)"
export PAITON_UID="$(id -u)" PAITON_GID="$(id -g)"
compose=(docker compose -f "$package_dir/compose.yaml")
case "${1:-start}" in
  --stop) exec "${compose[@]}" down ;;
  --logs) exec "${compose[@]}" logs --follow ;;
  --build) export PAITON_H3_BUILD=1 ;;
esac
source "$package_dir/image.sh"
prepare_h3_image
unset PAITON_H3_BUILD
"$package_dir/run.sh" download --profile "${PAITON_H3_PRESET:-turbo8}"
"${compose[@]}" up -d --no-build
printf 'Waiting for ComfyUI...\n'
for ((attempt=0; attempt<90; attempt++)); do
  if "${compose[@]}" exec -T comfyui python3 -c 'from urllib.request import urlopen; urlopen("http://127.0.0.1:8188/system_stats",timeout=2).close()' >/dev/null 2>&1; then
    printf 'Open http://127.0.0.1:%s/?paiton=1&preset=%s\n' "${PAITON_H3_PORT:-8190}" "${PAITON_H3_PRESET:-turbo8}"
    printf 'The included workflow opens on first visit. Edit the prompt and click Run.\n'
    printf 'Videos: %s\nLogs: %s --logs\nStop: %s --stop\n' "$PAITON_H3_OUTPUTS" "$0" "$0"
    exit 0
  fi
  sleep 1
done
printf 'ComfyUI did not become ready. Check %s --logs\n' "$0" >&2
exit 1
