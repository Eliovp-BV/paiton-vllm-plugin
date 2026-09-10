# GPT-OSS-20B on Radeon AI PRO R9700

Local community release candidate: original OpenAI MXFP4 weights with Paiton
expert decode kernels, served through vLLM's OpenAI-compatible API. One R9700
(gfx1201), 32 GB VRAM. Publication is pending; no new registry image is public yet.

The model is pinned to `6cee5e81ee83917806bbde320786a8fb61efebee`.
It has approximately 21B total / 3.6B active parameters. All experts remain on
the GPU. This package does not requantize the checkpoint.

## Measured performance

Primary scenario: 512 input / 256 output tokens, one request at a time, original
MXFP4 and BF16 KV. Paiton measured 2.214 s median over 32 requests.

| Configuration | Median end-to-end latency |
|---|---:|
| Fastest earlier stock vLLM reference | 4.819 s |
| Corrected release image, stock | 6.110 s |
| Corrected release image, Paiton | 2.214 s |

The conservative comparison is **54.0% lower latency** against the faster stock
reference. The stock timing shift also occurs on the original developer runtime;
its cause is not established. Output throughput includes explicitly counted
reasoning tokens. See [all measurements and limitations](BENCHMARKS.md).

## Launch

For local review, build the image using [REPRODUCE.md](REPRODUCE.md), then run from this directory:

```bash
PAITON_IMAGE=paiton-gpt-oss:local ./serve-docker.sh
```

After publication, `./serve-docker.sh` defaults to the new versioned GHCR image.
That registry tag is currently a proposed publication target. The first launch
downloads only the pinned model files. Model, Triton and vLLM
caches persist in `~/.cache/paiton-gpt-oss`. The pinned Harmony vocabulary is
embedded in the image, so cached-model offline launches also support chat. The API listens on `0.0.0.0:8020`
inside the container and is mapped to port 8020 on the host. Keep access limited
to your intended clients; this launch has no authentication configured.

```bash
# Same checkpoint and serving settings, stock expert implementation:
PAITON_IMAGE=paiton-gpt-oss:local PAITON_CONTAINER=paiton-gpt-oss-stock ./serve-docker.sh --stock

# Reuse an existing Hugging Face cache without modifying it:
PAITON_IMAGE=paiton-gpt-oss:local PAITON_HF_CACHE="$HOME/.cache/huggingface" ./serve-docker.sh --offline

# Terminal client (Python standard library only):
python3 chat.py
python3 chat.py 'Explain binary search in two sentences.'

curl http://localhost:8020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"What is 17 + 25?"}],"reasoning_effort":"low","max_tokens":512,"temperature":0,"stream":false}'
```

For streamed output and schema-constrained JSON:

```bash
curl --no-buffer http://localhost:8020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"Explain binary search in two sentences."}],"reasoning_effort":"low","max_tokens":256,"temperature":0,"stream":true}'

curl http://localhost:8020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"Return 17 plus 25 as JSON with integer field answer."}],"reasoning_effort":"low","max_tokens":256,"temperature":0,"response_format":{"type":"json_schema","json_schema":{"name":"sum","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}}}}'
```

Stop the existing instance before switching modes. Override `PAITON_IMAGE`,
`PAITON_PORT`, `PAITON_CONTAINER` or `PAITON_CACHE` as needed.

## Resources

Tested on a Linux host with 15.52 GiB usable RAM and 4 GiB swap. Whole-host swap
usage reached about 3 GiB during qualification; a swap-free 16 GB deployment is
not qualified. Reserve roughly 70 GB of free disk for the downloaded container,
its extracted runtime, 13.8 GB of model tensor data and persistent caches. Building
from the full developer base needs substantially more temporary disk than running
the compact image.

At the tested settings, sampled driver VRAM peaked at about 17.1 GiB stock and
17.0 GiB Paiton. Torch peak allocated memory was about 16.3 GiB and peak reserved
memory 16.4 GiB. These include all experts, padded weights, scales, nonquantized
weights, workspaces, graphs and the 2 GiB KV allocation. Original checkpoint
tensors occupy 12.82 GiB; the padded loaded model occupies about 14.17 GiB.

One startup with cached model weights and empty compilation caches took 155 s
stock and 142 s Paiton. These are individual startup observations, not a startup
speedup claim; initial download time is additional and network-dependent.

## Scope and limits

The release configuration uses 8,192 tokens **including prompt and generated
reasoning/answer**, at most two scheduled requests, a 2 GiB BF16 KV cache,
512-token prefill chunks, and graph capture for decode batches one and two.
Longer prompts and larger batches use the stock expert fallback. The advertised
131,072-token architecture limit is not this package's tested serving limit.

Attention, positional encoding, sinks, sliding windows, top-4 routing, sampling,
Harmony formatting and parsers remain in vLLM. Paiton replaces only supported
one/two-token expert computation. Low-bit storage is distinct from arithmetic:
Paiton unpacks original MXFP4 and accumulates in FP32, preserving the BF16
activation and weighted-partial rounding boundaries. KV cache stays BF16.

Reasoning effort `low`, `medium` and `high`, streaming, tool calls and JSON-schema
output are covered by local checks. Reasoning consumes the output token budget;
a small budget can finish before any answer. A strict local coding prompt can
produce a malformed `final code` channel even in stock, leaving no final answer.
An unconstrained JSON-only instruction can produce Markdown fences; request a
JSON schema when machine-readable output is required. This is a small regression
suite, not a claim of general quality parity.

This entry qualifies one GPU, no LoRA, no speculative decoding, and no
multi-GPU serving. It does not change clock, power, fan or voltage settings.

See [selection](SELECTION.md), [measured results](BENCHMARKS.md),
[reproduction](REPRODUCE.md), and [licenses/provenance](NOTICE.md).
