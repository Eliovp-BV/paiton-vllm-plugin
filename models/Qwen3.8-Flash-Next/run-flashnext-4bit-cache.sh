#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Opt-in 4-bit cache mode for agents and coding tools: 4-bit decode mode + prefix caching + an 8 GiB RAM prefix tier (about 104 GiB of pinned RAM in GTT;
# without the free GTT the tier is left off and prefix caching stays in VRAM). Options you pass come later and override these.
exec python3 "$script_dir/launch-flashnext.py" --mode 4bit-cache --name paiton-flashnext-4bit-cache "$@"
