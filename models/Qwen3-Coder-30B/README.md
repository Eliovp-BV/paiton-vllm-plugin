# Qwen3-Coder 30B on Radeon AI PRO R9700

Local release candidate for one 32 GB R9700 (`gfx1201`). Nothing in this
directory implies that a container or artifact has been published. The
benchmark evidence and supported scope are in [BENCHMARKS.md](BENCHMARKS.md).

Uses [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
through the pinned [cyankiwi INT4 checkpoint](https://huggingface.co/cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit/tree/4bd30395b72ea6045edd04806c4fea448d4467b3).
Both model cards declare Apache-2.0. The quantized weights use compressed-tensors
W4A16, symmetric groups of 32 and BF16 activations. The quantizer does not pin
its upstream weight revision; the exact quantized snapshot is pinned here.

## Local build and launch

The separate `QWEN3_CODER_OVERLAY` directory must contain the compiled `.so`,
its adjacent `.json` manifest, and the qualified AMD SMI `amdsmi/` runtime
package with its license. These are local reviewed build inputs. The private
compiler checkout is not a build context and is not needed to serve the model.

From the public runtime checkout:

```bash
docker build --pull=false --network=none \
  --build-context qwen3_coder_overlay="$QWEN3_CODER_OVERLAY" \
  -f models/Qwen3-Coder-30B/Dockerfile -t paiton-qwen3-coder:local .
./models/Qwen3-Coder-30B/serve-docker.sh
```

The first launch downloads only the pinned 18.1 GB quantized checkpoint.
Weights and runtime caches persist under `~/.cache/paiton`; set
`PAITON_CACHE_DIR` to change that directory. Existing downloads can be reused
by placing the Hugging Face cache at `$PAITON_CACHE_DIR/huggingface`.
Use `--offline` once the snapshot is present, or `--download-only` to separate
download time from startup. The API binds to localhost port 8010. Stop the
container before starting another configuration on that port.

Chat:

```bash
docker exec -it paiton-qwen3-coder paiton-chat \
  --url http://127.0.0.1:8010/v1/chat/completions --model qwen3-coder
```

API:

```bash
curl http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-coder","messages":[{"role":"user","content":"Write a Python binary search."}],"max_tokens":512,"temperature":0}'
```

## Reproduce the comparison

Launch the same image with `--stock --offline` to use stock vLLM with Paiton
disabled. Both modes use the same model snapshot, tokenizer, precision,
4096 context limit, two sequence slots, 512 batched tokens, 2 GiB BF16 KV pool,
ROCM_ATTN, O2 graphs with sizes 1 and 2, and disabled prefix caching.

Generate the request file once inside the server container:

```bash
docker exec paiton-qwen3-coder python3 /opt/paiton/qwen3-coder/benchmark/workload.py \
  /models/cache/huggingface/hub/models--cyankiwi--Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit/snapshots/4bd30395b72ea6045edd04806c4fea448d4467b3 \
  /models/cache/requests.json
```

Run the benchmark from the host with Python 3.12 and `aiohttp` installed, so
GPU telemetry is retained. `REQUESTS` must name that same retained request file
for both modes; `RESULTS` is an output directory. Use `LABEL=stock-final` or
`LABEL=paiton-final`, as appropriate:

```bash
python3 models/Qwen3-Coder-30B/benchmark/benchmark.py --requests "$REQUESTS" \
  --output "$RESULTS" --label "$LABEL" --concurrency 1 --runs 2 --warmup 16
python3 models/Qwen3-Coder-30B/benchmark/benchmark.py --requests "$REQUESTS" \
  --output "$RESULTS" --label "$LABEL" --concurrency 2 --runs 2 --warmup 2
python3 models/Qwen3-Coder-30B/benchmark/quality.py \
  --image paiton-qwen3-coder:local --output "$RESULTS/$LABEL-quality.json"
```

There are 16 requests per measured run, 229–241 actual input tokens and
exactly 256 output tokens each. Forced-length performance output is separate
from the natural-EOS quality suite. Four coding checks execute in containers
without network or GPU access, with CPU, memory and process limits. Latency
percentiles have only 32 request samples per setting across both runs.

## Requirements and scope

The initial qualification used an R9700, 16 GB host RAM with 4 GB swap,
an i5-8400, and Linux with AMD driver 6.19.14.31400000. Runtime: ROCm/HIP
7.14.60850, PyTorch 2.12.0+rocm7.14.0, vLLM commit
`39bd959b582c85e78e7e0326d49042ce7c3c07ed` in the pinned base image, and
Transformers 5.15.1. Relevant stock source files were checked against that
upstream commit; the image also carries unrelated Quark support patches.

Allow at least 30 GB free disk for the 18.1 GB checkpoint, roughly 5.1 GB image
and caches; 40 GB gives more room for local builds. Loading uses lazy
safetensors access because the checkpoint exceeds host RAM. Observed model
loading was about 46 seconds; cached startup takes roughly two minutes and a
fresh graph/compilation cache adds time. Download and compilation are excluded
from throughput measurements. Sampled warm VRAM use peaked near 20.1 GiB;
short loading or allocation transients may exceed a sampled peak.

Qualified workload: short coding prompts, concurrency 1 or 2, greedy decoding,
BF16 KV cache, and full GPU residency. The configured context cap is 4096;
long-context quality and performance have not been established. No LoRA,
expert parallelism or speculative decoding is qualified. Keep the GPU profile
at AUTO/COMPUTE when reproducing the results; no power or clock changes are
required. Qwen3-Coder is a non-thinking model.

Model license: [QWEN_LICENSE](QWEN_LICENSE). Runtime and retained component
notices: [NOTICE.md](NOTICE.md), repository [LICENSE](../../LICENSE), and
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
