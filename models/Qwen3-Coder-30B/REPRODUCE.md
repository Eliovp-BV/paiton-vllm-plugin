# Reproduce or rebuild Qwen3-Coder

The prebuilt container is the normal installation path. Download the release
bundle for its compiled artifact, public source and licensed AMD SMI runtime.
The private compiler is not required for these steps.

## Benchmark

Start the server with `serve-docker.sh --stock` or `serve-docker.sh` for Paiton.
Use the same cache and stop/remove the comparison container before switching
modes, or use distinct `PAITON_CONTAINER_NAME` values sequentially on this GPU.
Wait for `http://127.0.0.1:8010/health` to return HTTP 200.

Generate the request file once inside the server container:

```bash
docker exec paiton-qwen3-coder python3 /opt/paiton/qwen3-coder/benchmark/workload.py \
  /models/cache/huggingface/hub/models--cyankiwi--Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit/snapshots/4bd30395b72ea6045edd04806c4fea448d4467b3 \
  /models/cache/requests.json
```

Run the benchmark from the host with Python 3.12 and `aiohttp` installed so GPU
telemetry is retained. Set `REQUESTS` to that same retained request file for
both engines (normally `~/.cache/paiton/requests.json`), `RESULTS` to an output
directory and `LABEL` to `stock-final` or `paiton-final`:

```bash
python3 models/Qwen3-Coder-30B/benchmark/benchmark.py --requests "$REQUESTS" \
  --output "$RESULTS" --label "$LABEL" --concurrency 1 --runs 2 --warmup 16
python3 models/Qwen3-Coder-30B/benchmark/benchmark.py --requests "$REQUESTS" \
  --output "$RESULTS" --label "$LABEL" --concurrency 2 --runs 2 --warmup 2
python3 models/Qwen3-Coder-30B/benchmark/quality.py \
  --image ghcr.io/eliovp/paiton-vllm-plugin:qwen3-coder-30b-awq-rdna4-v1.0.0 \
  --output "$RESULTS/$LABEL-quality.json"
```

After both engines complete, summarize and check request equivalence:

```bash
python3 models/Qwen3-Coder-30B/benchmark/summarize.py \
  --evidence "$RESULTS" --output "$RESULTS/comparison.json"
```

The 16 coding requests have 229–241 actual input tokens and exactly 256 output
tokens. All performance responses deliberately end at the length cap. The
separate natural-EOS quality suite executes four coding checks in restricted
containers with no network/GPU, and checks instructions, arithmetic and JSON.
Aggregate throughput is output tokens divided by run makespan; per-request
rate includes TTFT. There are 32 request samples per engine/concurrency across
two measured runs. Hardware telemetry is sampled every 0.5 seconds. Use
`--gpu-pci-device` if more than one AMD GPU is installed.

## Offline image rebuild

From the extracted release bundle, after its pinned base image is available:

```bash
docker build --pull=false --network=none \
  --build-context qwen3_coder_overlay=./overlay \
  -f models/Qwen3-Coder-30B/Dockerfile -t paiton-qwen3-coder:local .
PAITON_QWEN3_CODER_IMAGE=paiton-qwen3-coder:local ./run.sh --chat
```

From a Git checkout, use the extracted bundle's `overlay/` directory as the
`qwen3_coder_overlay` build context and use the model's `serve-docker.sh` launcher.
Docker validates the compiled artifact hash and checks that its manifest
matches the model directory. `artifact-manifest.json` pins the checkpoint,
config hash, artifact hash and GPU architecture. The bundle's SHA-256 file
checks the full download. `container-images.json` records the published image.

The container uses the base image digest in its Dockerfile, which supplies the
qualified vLLM/ROCm/PyTorch versions. This recipe assembles that exact runtime
with public integration code and compiled artifacts; it does not rebuild the
private compiler or vLLM from source.
