#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 4-bit experts in the 200,000-token mode (speculative decoding off) with the expert host tier. Options you pass come later.
exec python3 "$script_dir/launch-flashnext.py" --mode 4bit-200k --name paiton-flashnext-4bit-200k "$@"
