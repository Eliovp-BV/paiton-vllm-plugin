#!/usr/bin/env bash
# Talk to the running server container (start it with launch.sh first).
#   ./run.sh generate "PROMPT" [--size 1280x704] [--frames 121] [--seed 42] [--out video.mp4]
#   ./run.sh generate "PROMPT" --image first-frame.png [--size 1920x1088] [--out video.mp4]
#   ./run.sh benchmark [--size 1280x704] [--runs 3]
#   ./run.sh stop
# Output files are written to the current directory.
set -euo pipefail
name="${PAITON_CONTAINER:-paiton-ltx-2.5}"
port="${PAITON_PORT:-30000}"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  stop) exec docker stop "$name" ;;
  generate|benchmark)
    py="$(command -v python3 || true)"
    [ -n "$py" ] || { echo "run.sh needs python3 on the host (standard library only)" >&2; exit 2; }
    exec "$py" "$here/client.py" "$@" --url "http://127.0.0.1:$port" ;;
  *) sed -n '2,7p' "$0" >&2; exit 2 ;;
esac
