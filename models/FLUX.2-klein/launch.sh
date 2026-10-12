#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ui=comfyui
build=0
action=start
while (($#)); do
  case "$1" in
    --ui) ui="${2:?Choose comfyui or simple}"; shift ;;
    --host) PAITON_BIND="${2:?Give an address, e.g. 127.0.0.1}"; shift ;;
    --build) build=1 ;;
    --stop) action=stop ;;
    --logs) action=logs ;;
    --help|-h)
      printf 'Usage: %s [--ui comfyui|simple] [--host ADDRESS] [--build] [--stop|--logs]\n' "$0"
      printf 'Downloads the pinned model once, keeps its cache, and starts the image interface.\n'
      printf 'The ports listen on all interfaces (0.0.0.0); --host 127.0.0.1 (or PAITON_BIND=127.0.0.1) keeps them local.\n'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done
[[ "$ui" == comfyui || "$ui" == simple ]] || { printf 'Choose --ui comfyui or --ui simple.\n' >&2; exit 2; }
command -v docker >/dev/null || { printf 'Install Docker with the Compose plugin first.\n' >&2; exit 1; }
docker info >/dev/null 2>&1 || { printf 'Docker is unavailable. Check that docker info works for your user.\n' >&2; exit 1; }
docker compose version >/dev/null
[[ -e /dev/kfd && -d /dev/dri ]] || { printf 'ROCm GPU devices were not found. Check the Radeon driver installation.\n' >&2; exit 1; }
export PAITON_RENDER_GID="$(stat -c '%g' /dev/kfd)"
export PAITON_BIND="${PAITON_BIND:-0.0.0.0}"
# open_hint PORT PATH: where to open the interface for the chosen bind address.
open_hint() {
  case "$PAITON_BIND" in
    0.0.0.0|::|'[::]') printf '\nOpen http://127.0.0.1:%s%s on this machine, or http://<this-machine-ip>:%s%s from another system.\n' "$1" "$2" "$1" "$2" ;;
    127.*|localhost|::1|'[::1]') printf '\nOpen http://127.0.0.1:%s%s\n' "$1" "$2" ;;
    *) printf '\nOpen http://%s:%s%s\n' "$PAITON_BIND" "$1" "$2" ;;
  esac
}
export PAITON_IMAGE="${PAITON_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:flux2-klein-rdna4-v1.0.1}"
export PAITON_TOOLS_IMAGE="${PAITON_TOOLS_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:flux2-tools-rdna4-v1.0.1}"
export PAITON_COMFYUI_IMAGE="${PAITON_COMFYUI_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:flux2-comfyui-rdna4-v1.0.1}"
export PAITON_OUTPUTS="${PAITON_OUTPUTS:-$PWD/paiton-images}"
export PAITON_CACHE="${PAITON_CACHE:-paiton-flux2-cache}"
if [[ "$PAITON_CACHE" == /* ]]; then
  export PAITON_CACHE_BIND="$PAITON_CACHE"
  export PAITON_CACHE_VOLUME=paiton-flux2-cache
  export PAITON_RUNTIME_CACHE_BIND="$PAITON_CACHE/cache"
  export PAITON_RUNTIME_CACHE_VOLUME=paiton-flux2-cache-runtime
  mkdir -p "$PAITON_RUNTIME_CACHE_BIND"
else
  export PAITON_CACHE_VOLUME="$PAITON_CACHE"
  export PAITON_RUNTIME_CACHE_VOLUME="${PAITON_CACHE}-runtime"
  unset PAITON_CACHE_BIND PAITON_RUNTIME_CACHE_BIND
fi
compose=(docker compose -f "$package_dir/compose.yaml")
if [[ "$action" == stop ]]; then "${compose[@]}" down; exit; fi
if [[ "$action" == logs ]]; then "${compose[@]}" logs --follow; exit; fi
mkdir -p "$PAITON_OUTPUTS"
services=(paiton stock)
[[ "$ui" != comfyui ]] || services+=(comfyui)
if ((build)); then
  "${compose[@]}" build "${services[@]}"
else
  "${compose[@]}" pull "${services[@]}"
fi
printf '\nPreparing the pinned FLUX.2 klein cache. First download: 5.46 GB; converted tensors: about 12 GB.\n'
"$package_dir/run.sh" download
if [[ "$ui" == simple ]]; then
  "${compose[@]}" down
  open_hint 7860 ''
  printf 'Keep this terminal open.\n'
  exec "$package_dir/run.sh" serve --port 7860
fi
"${compose[@]}" up -d --no-build
port="${PAITON_UI_PORT:-8188}"
printf '\nWaiting for the local interface to start...\n'
ready=0
for ((attempt=0; attempt<60; attempt++)); do
  if "${compose[@]}" exec -T comfyui python3 -c 'from urllib.request import urlopen; urlopen("http://127.0.0.1:8188/system_stats",timeout=2).close()' >/dev/null 2>&1; then
    ready=1; break
  fi
  sleep 1
done
if ((!ready)); then
  printf 'The interface did not become ready. Check: %s --logs\n' "$0" >&2
  exit 1
fi
open_hint "$port" '/?paiton=1'
printf 'The Paiton workflow opens on your first visit. Edit the prompt and click Run.\n'
printf 'The first image prepares the engine; later images reuse it. Switching engines reloads the model.\n'
printf 'Images are saved in %s\n' "$PAITON_OUTPUTS"
printf 'Logs: %s --logs\nStop: %s --stop\n' "$0" "$0"
