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
export PAITON_WAN_DATA="${PAITON_WAN_DATA:-${XDG_DATA_HOME:-$HOME/.local/share}/paiton/wan22}"
export PAITON_WAN_OUTPUTS="${PAITON_WAN_OUTPUTS:-$HOME/paiton-videos}"
mkdir -p "$PAITON_WAN_DATA/input" "$PAITON_WAN_DATA/user" "$PAITON_WAN_DATA/custom_nodes" "$PAITON_WAN_OUTPUTS"
export PAITON_WAN_DATA="$(realpath "$PAITON_WAN_DATA")"
export PAITON_WAN_OUTPUTS="$(realpath "$PAITON_WAN_OUTPUTS")"
export PAITON_RENDER_GID="$(stat -c '%g' /dev/kfd)"
export PAITON_UID="$(id -u)" PAITON_GID="$(id -g)"
compose=(docker compose -f "$package_dir/compose.yaml")
case "${1:-start}" in
  --stop) exec "${compose[@]}" down ;;
  --logs) exec "${compose[@]}" logs --follow ;;
  --build) export PAITON_WAN_BUILD=1 ;;
esac
source "$package_dir/image.sh"
prepare_wan_image
unset PAITON_WAN_BUILD
"$package_dir/run.sh" download --preset "${PAITON_WAN_DOWNLOAD_PRESET:-base}"
"${compose[@]}" up -d --no-build
printf 'Waiting for ComfyUI...\n'
for ((attempt=0; attempt<90; attempt++)); do
  if "${compose[@]}" exec -T comfyui python3 -c 'from urllib.request import urlopen; urlopen("http://127.0.0.1:8188/system_stats",timeout=2).close()' >/dev/null 2>&1; then
    printf 'ComfyUI is listening on 0.0.0.0:%s\n' "${PAITON_WAN_PORT:-8192}"
    printf 'On this host: http://127.0.0.1:%s/?paiton=1&preset=%s\n' "${PAITON_WAN_PORT:-8192}" "${PAITON_WAN_PRESET:-base}"
    printf 'From another system: http://<server-ip>:%s/?paiton=1&preset=%s\n' "${PAITON_WAN_PORT:-8192}" "${PAITON_WAN_PRESET:-base}"
    printf 'The included workflow opens on first visit. Edit the prompt and click Run.\n'
    printf 'Videos: %s\nLogs: %s --logs\nStop: %s --stop\n' "$PAITON_WAN_OUTPUTS" "$0" "$0"
    exit 0
  fi
  sleep 1
done
printf 'ComfyUI did not become ready. Check %s --logs\n' "$0" >&2
exit 1
