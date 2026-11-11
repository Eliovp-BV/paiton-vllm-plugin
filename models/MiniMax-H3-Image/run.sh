#!/usr/bin/env bash
# Talk to the running server container (start it with launch.sh first).
#   ./run.sh generate "PROMPT" [--size 4096x4096] [--seed 42] [--out out.png]
#   ./run.sh edit "PROMPT" --image ref.png [--size 2048x2048] [--out edit.png]
#   ./run.sh benchmark [--size 4096x4096] [--runs 3]
#   ./run.sh stop
# Output files are written to the current directory.
set -euo pipefail
name="${PAITON_CONTAINER:-paiton-minimax-h3-image}"
port="${PAITON_PORT:-30000}"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  stop) exec docker stop "$name" ;;
  generate|edit|benchmark)
    py="$(command -v python3 || true)"
    [ -n "$py" ] || { echo "run.sh needs python3 on the host (standard library only)" >&2; exit 2; }
    exec "$py" "$here/client.py" "$@" --url "http://127.0.0.1:$port" ;;
  *) sed -n '2,7p' "$0" >&2; exit 2 ;;
esac
