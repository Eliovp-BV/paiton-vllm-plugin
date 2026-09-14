#!/usr/bin/env bash
# Reproduce the fixed paired set. Run while other GPU generation services are idle.
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_id="${PAITON_WAN_BENCHMARK_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
for setting in fast-480-2 fast-720-5 fast-720-2-person fast-480-5 base-480-2-image base-480-5-image base-480-2; do
  IFS=- read -r preset resolution duration content <<< "$setting"
  for engine in stock paiton; do
    args=(benchmark --preset "$preset" --resolution "$resolution" --duration "$duration" --engine "$engine" --runs 4 --warmups 2 --output "/outputs/$run_id/$setting-$engine")
    if [[ ${content:-} == image ]]; then
      args+=(--portrait --image /data/input/wan-official-example.jpg --prompt 'The cat wearing sunglasses sits in the boat and slowly turns its head. Gentle water ripples. Preserve the cat, sunglasses and boat composition. Smooth continuous motion.')
    fi
    if [[ ${content:-} == person ]]; then
      args+=(--seed 2702 --prompt 'A woman in a yellow raincoat walks along a wet city sidewalk, stops beside a flower stall and smiles at the flowers. Raindrops glisten on the pavement. Natural human movement, consistent face and clothing, a steady medium-wide camera shot.')
    fi
    "$package_dir/run.sh" "${args[@]}"
  done
done
