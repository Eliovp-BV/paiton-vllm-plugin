#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 4-bit experts, 262,144-token window with speculative decoding on, ~461K tokens of KV cache. Options you pass come later.
exec python3 "$script_dir/launch-flashnext.py" --mode 4bit-long --name paiton-flashnext-4bit-long "$@"
