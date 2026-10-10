#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The 200,000-token mode (prefill-long: speculative decoding off). Options you pass come later and override these.
exec python3 "$script_dir/launch-flashnext.py" --mode prefill-long --name paiton-flashnext-200k "$@"
