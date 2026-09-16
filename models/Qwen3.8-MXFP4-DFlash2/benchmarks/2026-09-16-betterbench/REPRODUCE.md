# Reproduce the single-GPU screen

Use the [methodology](METHOD.md), [provenance](provenance.json), pinned [checkpoint files](../../checkpoint.lock.json), and [quick profile](profile-quick.json). Run one server at a time on the R9700. The commands below create new output directories and unique container names; they stop only containers they start.

## Prepare the client and checkpoints

Start in this benchmark directory in a checkout of `Eliovp-BV/paiton-vllm-plugin`. Use Bash and Python 3.10 or later; measurements used Python 3.12.3. Provide absolute paths to snapshots matching every hash in `../../checkpoint.lock.json`.

```bash
set -euo pipefail
BENCH_DIR=$(pwd -P)
BENCH_WORK=$(mktemp -d -t paiton-betterbench.XXXXXXXX)
BENCH_STAMP=$(date -u +%Y%m%d-%H%M%S)
PAITON_NAME="betterbench-paiton-$BENCH_STAMP"
GGZ_NAME="betterbench-ggz-$BENCH_STAMP"

# Replace these with your verified snapshot directories.
TARGET_SNAPSHOT=$(realpath /path/to/target-snapshot)
DRAFT_SNAPSHOT=$(realpath /path/to/draft-snapshot)

git clone https://github.com/GGZ14/BetterBench.git "$BENCH_WORK/BetterBench"
git -C "$BENCH_WORK/BetterBench" checkout --detach \
  d00ad5ec8098c06584a88ec3468bacd37d5ed098
python3 -m venv "$BENCH_WORK/venv"
"$BENCH_WORK/venv/bin/python" -m pip install "$BENCH_WORK/BetterBench"
BENCH_PY="$BENCH_WORK/venv/bin/python"

wait_for_server() {
  local bench_name=$1 bench_port=$2 bench_attempt
  for bench_attempt in $(seq 1 180); do
    if curl --fail --silent "http://127.0.0.1:$bench_port/health" >/dev/null; then
      return 0
    fi
    if [ "$(docker inspect --format '{{.State.Running}}' "$bench_name")" != true ]; then
      docker logs --tail 80 "$bench_name"
      return 1
    fi
    sleep 5
  done
  docker logs --tail 80 "$bench_name"
  return 1
}

run_betterbench() {
  local bench_label=$1 bench_port=$2
  mkdir -p "$BENCH_WORK/$bench_label"
  "$BENCH_PY" -m betterbench.cli run \
    --endpoint "http://127.0.0.1:$bench_port/v1" \
    --model Qwen3.8-27B-Quark-AWQ-MXFP4 \
    --config "$BENCH_DIR/profile-quick.json" \
    --corpus "$BENCH_DIR/corpus" \
    --max-model-len 8192 --note scope=quick \
    --out "$BENCH_WORK/$bench_label/results.json" \
    | tee "$BENCH_WORK/$bench_label/client.log"
}

docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
```

Resolve any existing GPU ownership before continuing. The published client source and corpus manifests permit independent hash checks. Dependency resolution is not fully locked; retain `pip freeze` with new results. Installing the pinned BetterBench package supplies its NumPy dependency; the HTML renderer needs no additional packages.

## Paiton release

The public [launcher](../../serve.py) reads the immutable image digest in [runtime.lock.json](../../runtime.lock.json): `ghcr.io/eliovp/paiton-vllm-plugin@sha256:9b2dae214076d35de785e073b31294b033a376b16e6bc1ec1fdada4e54d96c59`. It verifies the supplied checkpoints and uses the release's measured defaults, including the explicit 5 GiB cache.

```bash
python3 "$BENCH_DIR/../../serve.py" --offline --detach \
  --name "$PAITON_NAME" --port 8011 \
  --target "$TARGET_SNAPSHOT" --draft "$DRAFT_SNAPSHOT"
wait_for_server "$PAITON_NAME" 8011
run_betterbench paiton-initial 8011
docker logs --timestamps "$PAITON_NAME" \
  > "$BENCH_WORK/paiton-initial/engine.log" 2>&1
docker stop -t 30 "$PAITON_NAME"
```

Save logs before stopping: the launcher uses `--rm`. Keep the resulting compile cache for the repeat. The existing default cache volume is reused; the supplied checkpoints remain read-only mounts.

## GGZ14 reference stack

Build this external runtime locally before launching it. Its tested inputs are recorded in [provenance.json](provenance.json):

- [GGZ14/vllm-mxfp4](https://github.com/GGZ14/vllm-mxfp4/tree/92eed82fcbb1cca31f7a9108c6371a0bba647ee2) at `92eed82fcbb1cca31f7a9108c6371a0bba647ee2`.
- Base `stilldeadcode/vllm-radiance@sha256:45694209177a55a1ab3ba6702fe6e978b1b66a6e66ae3fc066f8d579f7bc4c25`.
- [libr4d](https://codeberg.org/StillDeadcode/libr4d) at `b9e42ab7202f53a3bc13d415f5d41481f9ca311b`, with `r4d_radiance_extras.patch` from that GGZ revision.

The base image alone is insufficient. Follow the pinned [upstream setup/build documentation](https://github.com/GGZ14/vllm-mxfp4/blob/92eed82fcbb1cca31f7a9108c6371a0bba647ee2/README.md) and [serve-mxfp4.sh](https://github.com/GGZ14/vllm-mxfp4/blob/92eed82fcbb1cca31f7a9108c6371a0bba647ee2/serve-mxfp4.sh): apply its startup patches and module/config copies, compile its `radiance_mxfp4_fp8.hip` for `gfx1201`, and install the pinned, patched `r4d.so`. The measured build used two compilation jobs. Perform build preparation without GPU devices. Preserve the original target checkpoint and select DFlash2; this experiment does not use the upstream MTP rewrite.

The experiment's built and warmed image hashes identify **local artifacts**, not published images you can pull. A rebuild is not guaranteed to reproduce their bytes or compilation-cache state. Save your source revisions, build log, image ID and installed package versions with new measurements. The following launch command expects a locally prepared image with the same patched stack already installed; it is not a replacement build recipe.

```bash
GGZ_IMAGE=your-locally-prepared-image:92eed82
mkdir -p "$BENCH_WORK/ggz-cache"
mapfile -d '' GGZ_ARGS < <(
  "$BENCH_PY" - "$BENCH_DIR/reference-arguments.json" <<'PY'
import json, sys
for arg in json.load(open(sys.argv[1])):
    sys.stdout.write(arg + '\0')
PY
)

docker run -d --name "$GGZ_NAME" --network host \
  --device /dev/kfd --device /dev/dri --shm-size 2g \
  --env-file "$BENCH_DIR/reference.env" \
  -e REFERENCE_SAFETENSORS_PREAD=1 -e PYTHONPATH=/loader \
  -v "$TARGET_SNAPSHOT:/models/target:ro" \
  -v "$DRAFT_SNAPSHOT:/models/draft:ro" \
  -v "$BENCH_DIR/reference-loader:/loader:ro" \
  -v "$BENCH_WORK/ggz-cache:/cache:rw" \
  --entrypoint /opt/vllm/bin/vllm "$GGZ_IMAGE" serve "${GGZ_ARGS[@]}"

wait_for_server "$GGZ_NAME" 8012
run_betterbench ggz-hipmatched-initial 8012
docker logs --timestamps "$GGZ_NAME" \
  > "$BENCH_WORK/ggz-hipmatched-initial/engine.log" 2>&1
```

[reference.env](reference.env) includes the matched HIP settings; [reference-arguments.json](reference-arguments.json) preserves the measured engine options. The [loader shim](reference-loader/sitecustomize.py) selects safetensors' `pread` backend to fit loading within 16 GB host RAM; it changes I/O, not model arithmetic. Check that the built image's safetensors supports this backend before launching.

For a warmed reference repeat, keep this server running and rerun the identical client profile under a new output name. Save logs, then stop it before starting Paiton again:

```bash
run_betterbench ggz-hipmatched-repeat 8012
docker logs --timestamps "$GGZ_NAME" \
  > "$BENCH_WORK/ggz-hipmatched-repeat/engine.log" 2>&1
docker stop -t 30 "$GGZ_NAME"

PAITON_NAME="betterbench-paiton-repeat-$BENCH_STAMP"
python3 "$BENCH_DIR/../../serve.py" --offline --detach \
  --name "$PAITON_NAME" --port 8011 \
  --target "$TARGET_SNAPSHOT" --draft "$DRAFT_SNAPSHOT"
wait_for_server "$PAITON_NAME" 8011
run_betterbench paiton-repeat 8011
docker logs --timestamps "$PAITON_NAME" \
  > "$BENCH_WORK/paiton-repeat/engine.log" 2>&1
docker stop -t 30 "$PAITON_NAME"
"$BENCH_PY" -m pip freeze > "$BENCH_WORK/client-packages.txt"
```

This replay uses the stronger matched HIP environment from the outset. The original four-run history additionally contains an earlier vendor-environment GGZ run, followed by a container restart with matched flags and reused caches; see [METHOD.md](METHOD.md). Do not merge scores from differing environments or call cache states identical without checking them. Keep all new runs, including regressions and failures.

## Regenerate visuals without a GPU

BetterBench writes a standalone HTML report beside each new JSON. To regenerate one of the published reports from sanitized raw measurements, use the same pinned client:

```bash
"$BENCH_PY" -m betterbench.cli report \
  "$BENCH_DIR/data/paiton-release-repeat.json" \
  --html --out "$BENCH_WORK/paiton-release-repeat.html"

"$BENCH_PY" -m pip install matplotlib==3.10.7
"$BENCH_PY" "$BENCH_DIR/render_charts.py" \
  --data-dir "$BENCH_DIR/data" \
  --provenance "$BENCH_DIR/provenance.json" \
  --output-dir "$BENCH_WORK/charts"
```

Repeat the HTML command for the other three JSON files. The chart script derives performance values from the published raw measurements; cache capacities come from the startup evidence recorded in provenance. BetterBench's corpus and generated report template retain [their license](BETTERBENCH-LICENSE). No compiler or GPU is needed to verify the published measurements and render these visuals.
