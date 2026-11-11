# Kimi-K3.1-Pro on 4 AMD Instinct GPUs: community build

Run Moonshot AI's [Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) (1.56 TB, MXFP4) on 4 AMD Instinct
CDNA4 GPUs with one container and an OpenAI-compatible API.

This build is plain PyTorch, about 600 lines of Python. It is unoptimized and contains nothing proprietary: no Paiton
engine, no custom kernels, no batching, no speculative decoding. It is slow: on 4x MI355X expect 2 to 5 seconds per
output token and about 5 prompt tokens per second of prefill, one request at a time. Use it to try the model, check
its answers on your own prompts, or read a complete forward pass of the architecture in one file.

The optimized Paiton engine for these GPUs is proprietary and not part of this repository. On the same 4 GPUs it is
70 to 180 times faster per token and serves 16 requests at once; on 6 MI355X at 800 W it reaches 944 tokens per
second per GPU on 8K-in, 1K-out requests. See https://eliovp.com for its results.

[Start](#start) · [Requirements](#requirements) · [Use the API](#use-the-api) · [Settings](#settings) ·
[How it works](#how-it-works) · [Limits](#limits) · [Benchmarks](BENCHMARKS.md)

## Start

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
./models/Kimi-K3.1-Pro/serve-docker.sh --download-only   # 1.56 TB into ~/.cache/paiton/Kimi-K3.1-Pro
./models/Kimi-K3.1-Pro/serve-docker.sh --wait
```

Already have the checkpoint? Point the launcher at it with `PAITON_MODEL_DIR=/path/to/Kimi-K3.1-Pro` and skip the
download. The launcher pins revision `d9e99b07` of `moonshotai/Kimi-K3.1-Pro`.

Loading copies about 1 TB onto the GPUs and takes 6 to 25 minutes, depending on whether the checkpoint is already in
the page cache. The first request after loading is slower still. To run the container yourself:

```bash
docker run -d --name paiton-kimi31 \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host --shm-size 16g \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -v /path/to/Kimi-K3.1-Pro:/model:ro -p 8031:8000 \
  ghcr.io/eliovp/paiton-vllm-plugin:kimi31pro-rocm10.1.0-pytorch2.13.0-cdna4-v1
```

## Requirements

| Requirement | What you need |
| --- | --- |
| **GPU** | 4 or more AMD Instinct CDNA4 GPUs (gfx950). Tested on 4x MI355X, 288 GB each, at an 800 W power limit. |
| **Host memory** | 768 GB or more. 4x MI355X hold about 1 TB of the checkpoint; the other experts are read from the page cache. With less RAM, every token reads from disk and generation gets much slower. |
| **Disk** | 1.6 TB for the checkpoint, plus 37 GB for the image. |
| **Host** | Linux with ROCm and Docker with access to `/dev/kfd` and `/dev/dri`. |

More GPUs means fewer experts read from host memory. With 8x MI355X the whole checkpoint fits on the GPUs.

## Use the API

```bash
curl http://localhost:8031/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "kimi-k3.1-pro",
  "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
  "max_tokens": 64
}'
```

The server answers `/v1/chat/completions` and `/v1/completions` (streaming or not), `/v1/models` and `/health`.
Thinking is off by default. Turn it on per request with `"chat_template_kwargs": {"thinking": true}`; the reasoning
comes back in `reasoning_content`.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `PAITON_MODEL_DIR` | `~/.cache/paiton/Kimi-K3.1-Pro` | Checkpoint folder |
| `PAITON_GPUS` | `0,1,2,3` | GPUs to use, in HIP numbering (see below) |
| `PAITON_HOST_PORT` | `8031` | Port on this machine |
| `PAITON_BIND` | `0.0.0.0` | `127.0.0.1` keeps the API on this machine |
| `PAITON_API_KEY` | unset | When set, `/v1` requests need `Authorization: Bearer <key>` |
| `PAITON_MAX_MODEL_LEN` | `32768` | Prompt plus output tokens per request |

`PAITON_GPUS` sets `HIP_VISIBLE_DEVICES`. HIP can number GPUs differently from `amd-smi` and `rocm-smi`; on our
8-GPU test node, HIP device 3 was `rocm-smi` GPU 1. If you need specific GPUs, check the HIP order with `rocminfo`
(GPU agents in order, with their PCI bus in `BDFID`) and confirm with `amd-smi monitor` after the start.

When you run the container yourself, the server also takes `--prefill-chunk` (512 prompt tokens per forward pass;
lower it if a long prompt runs out of GPU memory) and `--reserve-gb` (40 GB per GPU left free for activations and the
KV cache).

## How it works

`kimi31_community/model.py` is the whole model: KDA linear attention (as a token-by-token loop), gated MLA with the
DSA indexer, the latent MoE with 896 experts, attention residuals and MXFP4 unpacking. Layers are split evenly across
the GPUs. Each GPU keeps its layers' attention and dense weights, plus as many of its layers' experts as fit (59 of 92
MoE layers on 4x MI355X). A token routed to an expert that did not fit reads it from host memory for that step.

`kimi31_community/server.py` is a small OpenAI-compatible HTTP server. Prompts use the chat encoding and tokenizer
shipped with the checkpoint.

`tests/test_tiny.py` checks, on CPU with a tiny random checkpoint, that a prompt run at once and the same prompt run
token by token give the same logits.

## Limits

- One request at a time; others wait.
- Text only. Tool calls in the output are not parsed.
- No prefix caching: every request starts from its full prompt.
- The default context is 32,768 tokens. Longer prompts work but take many minutes of prefill.

## License

The code in this folder is under this repository's Apache-2.0 license. The model is under Moonshot AI's Kimi K3.1-Pro
License ([MODEL_LICENSE](MODEL_LICENSE)); see [NOTICE.md](NOTICE.md).
