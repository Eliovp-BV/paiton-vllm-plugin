#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Exact-arithmetic prefill, about 7 % slower (decode-nopf; prefill-long-nopf is the 200K counterpart). Options you pass come later.
exec python3 "$script_dir/launch-flashnext.py" --mode decode-nopf --name paiton-flashnext-exact "$@"
