#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PAITON_H3_DATA="${PAITON_H3_DATA:-$PWD/paiton-h3-data}"
export PAITON_H3_OUTPUTS="${PAITON_H3_OUTPUTS:-$PWD/paiton-videos}"
mkdir -p "$PAITON_H3_DATA/user" "$PAITON_H3_DATA/custom_nodes" "$PAITON_H3_OUTPUTS"
export PAITON_H3_DATA="$(realpath "$PAITON_H3_DATA")"
export PAITON_H3_OUTPUTS="$(realpath "$PAITON_H3_OUTPUTS")"
export PAITON_RENDER_GID="$(stat -c '%g' /dev/kfd)"
export PAITON_UID="$(id -u)" PAITON_GID="$(id -g)"
compose=(docker compose -f "$package_dir/compose.yaml")
case "${1:-start}" in
  --stop) exec "${compose[@]}" down ;;
  --logs) exec "${compose[@]}" logs --follow ;;
  --build) "${compose[@]}" build ;;
  start) ;;
  *) printf 'Usage: %s [--build|--stop|--logs]\n' "$0"; exit 2 ;;
esac
"$package_dir/run.sh" download --profile "${PAITON_H3_PRESET:-turbo8}"
"${compose[@]}" up -d --no-build
printf 'Open http://127.0.0.1:%s and load a workflow from comfyui/workflows.\n' "${PAITON_H3_PORT:-8190}"
printf 'Outputs: %s\n' "$PAITON_H3_OUTPUTS"
