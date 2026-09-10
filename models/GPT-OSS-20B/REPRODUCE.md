# Reproduce the local candidate

The release bundle supplies a compiled `gptoss_overlay` directory with the
shared object and its manifest. The private compiler source is not distributed.
The binary SHA-256 is
`db2d6d6b4855b7cc3e97be0f499630c40758ebebdf97cb8abf34bb6c2a8f2e5b`.
The plugin checks the manifest, checksum, GPU architecture and model dimensions
before enabling the artifact.

From the public repository root, using Docker BuildKit:

```bash
docker buildx build --load \
  --build-context gptoss_overlay=./gptoss_overlay \
  -f models/GPT-OSS-20B/Dockerfile \
  -t paiton-gpt-oss:local .
```

Both base images and the Dockerfile frontend are pinned by digest. The ROCm
runtime is copied intact into separate layers below GHCR's 10 GB layer limit.
Torch is separated from the remaining Python/ROCm files; the SDK components stay
together to preserve their shared library hard links. The build checks all 2,984
hard-link groups against a manifest generated from the pinned base image.
Model weights are downloaded at launch, not baked into the image. Downloads are
pinned and exclude duplicate original/Metal representations. Original checkpoints
remain unchanged. The model license and usage policy are included.

## Qualification

The benchmark-only `--qualification` server mode binds to loopback and enables
worker memory telemetry. Do not use that mode for a public service. It also fixes
the Harmony date through `VLLM_SYSTEM_START_DATE=2026-09-09` for reproducible
comparisons. GPT-OSS's special Harmony renderer bypasses the Jinja template date;
the environment override is required. Ordinary chat retains the upstream
current-date behavior.

Start a local qualification server using host networking so its loopback-only
API is reachable from the host. Run this from the model directory:

```bash
docker run --rm --name paiton-gpt-oss-qualification --network host \
  --device /dev/kfd --device /dev/dri --ipc=host \
  -v "$HOME/.cache/paiton-gpt-oss:/models/cache" \
  paiton-gpt-oss:local --qualification --stock
# Remove --stock for the Paiton run, after stopping the stock instance.
```

In another terminal, prepare the benchmark client:

```bash
python3 -m venv .bench-venv
.bench-venv/bin/pip install -r benchmark/requirements.txt
source .bench-venv/bin/activate
mkdir -p results
```

Use the same image, pinned checkpoint and settings for both modes. Stop each
owned server before starting the next. Run without other GPU workloads or image
builds. The benchmark scripts require Python with `aiohttp`; the image includes
it. Mount this directory at `/bench` and a writable results directory at `/results`
when running the scripts in a separate container with host networking.

```bash
mode=stock
python3 benchmark/benchmark.py --url http://127.0.0.1:8020 \
  --requests benchmark/requests/requests-p512-o256.json --output results \
  --label "$mode-p512-o256" --concurrency 1 --runs 2 --warmup 2
# Repeat after starting Paiton with mode=paiton.
python3 benchmark/chat_benchmark.py --out "results/$mode-chat.json"
python3 benchmark/api_checks.py --out "results/$mode-api.json" \
  --long-cases benchmark/long-cases.json
```

For the rest of the matrix, repeat in each serving mode and set `mode` accordingly:

```bash
mode=stock
python3 benchmark/benchmark.py --url http://127.0.0.1:8020 \
  --requests benchmark/requests/requests-p512-o256-c2.json --output results \
  --label "$mode-p512-o256" --concurrency 2 --runs 2 --warmup 2
for shape in p128-o128 p2048-o128 p6144-o128 p512-o1024; do
  python3 benchmark/benchmark.py --url http://127.0.0.1:8020 \
    --requests "benchmark/requests/requests-$shape.json" --output results \
    --label "$mode-$shape" --concurrency 1 --runs 1 --warmup 2
done
python3 benchmark/quality.py --cases benchmark/quality_cases.json \
  --out "results/$mode-quality.json" --code-image paiton-gpt-oss:local
# After both modes have completed:
python3 benchmark/compare_quality.py results/stock-quality.json \
  results/paiton-quality.json --out results/quality-comparison.json
```

The comparison audit rejects different request bodies or tokenized prompts. It
reports task regressions and improvements without equating identical text with
numerical parity.

The forced-length primary workload includes all generated tokens in throughput;
reasoning, final-channel, control and post-stop tokens are reported separately.
It does not represent useful-answer throughput. Natural-stop chat records the
first non-whitespace answer content, reasoning, final content and EOS behavior.
Raw requests, returned token IDs, streamed events, responses, metrics and telemetry
are retained with the result archive. The quality runner executes generated Python
in a separate unprivileged, networkless, resource-bounded Docker container.

## Download the compiled overlay

```bash
hf download EliovpAI/GPT-OSS-20B-MXFP4-Paiton-RDNA4 --revision v1.0.0 \
  --include 'overlay/*' --local-dir ./paiton-gptoss-hf
cp -a ./paiton-gptoss-hf/overlay ./gptoss_overlay
```

Use this directory as the `gptoss_overlay` build context in the recipe above.
The overlay contains the tested compiled region and manifest; model weights
continue to come from the pinned OpenAI repository.
