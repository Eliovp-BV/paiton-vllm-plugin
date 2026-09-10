# MiniCPM5-2B: quick local chat and code on R9700

A small Apache-2.0 instruction model, with Paiton-compiled W4A16 decode and an
OpenAI-compatible vLLM API. Qualified target: one Radeon AI PRO R9700, gfx1201,
32 GB VRAM. This separate image preserves every existing community package.

The exact checkpoint is [openbmb/MiniCPM5-2B-GPTQ](https://huggingface.co/openbmb/MiniCPM5-2B-GPTQ/tree/6c1ee6fa521aa53f47cfb32696e6d8ef5b0db805),
revision `6c1ee6fa521aa53f47cfb32696e6d8ef5b0db805`. Despite the repository name,
its tensors use **AWQ GEMM packing, asymmetric INT4, group size 128**. Activations
and KV cache use FP16; embeddings and output head are unquantized. It has 2.517B
dense parameters. The original checkpoints remain unchanged.

This is for short chat, concise coding help and lightweight tool requests.
Small-model limitations matter: deterministic direct-answer quality was 14/20
on the retained task set, versus 15/20 for the BF16/FP16 source. Paiton matched
all 20 stock W4 responses, including its failures. Arithmetic, strict filtering,
long reasoning and unfamiliar code need review. This is not a replacement for
a larger reasoning model. [Selection and alternatives](SELECTION.md). [Quantization evidence](QUANTIZATION.md).

## Run

The versioned GHCR image is published. [Compiled artifacts and manifests](https://huggingface.co/EliovpAI/MiniCPM5-2B-W4A16-Paiton-RDNA4/tree/v1.0.0) are also available on Hugging Face; weights download from the pinned upstream checkpoint.

```bash
./serve-docker.sh
```

The server listens on `0.0.0.0:8036`. Downloads are pinned and SHA-256 verified.
Weights and runtime caches persist in `~/.cache/paiton-minicpm5-2b`.
Use `PAITON_CACHE` to choose another directory, or `PAITON_PORT` to change the
published port. Do not run competing GPU workloads during benchmarking.

```bash
./serve-docker.sh --download-only     # download and verify ahead of time
./serve-docker.sh --offline           # reuse the prepared checkpoint
./serve-docker.sh --offline --stock   # same checkpoint, native vLLM projections
python3 chat.py
```

```bash
curl http://localhost:8036/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minicpm5-2b","messages":[{"role":"user","content":"Explain binary search in two sentences."}],"max_tokens":256,"temperature":0,"stream":true}'
```

Default: thinking disabled, 8,192-token context, two sequences, 512 scheduled
prefill tokens, 1 GiB FP16 KV cache, decode graphs for batches one and two,
no prefix caching or speculation. The 128K advertised upstream context is not
qualified here. Requests may set `chat_template_kwargs.enable_thinking=true`,
but thinking-on W4 is experimental and excluded from this release qualification. Reasoning can consume the output budget; the fast-chat profile is tested
with thinking off. Reasoning parsing uses `qwen3`, tools use `minicpm5`.

Studio uses the existing chat adapter, pinned image/checkpoint identities,
read-only weight mount, persistent runtime cache and exclusive GPU lease.
A ready model is reused for 120 seconds while eligible; switching releases the
owned model before loading another. The separate Studio change must be installed
with this image; it does not modify an active Studio service.

## Optional thinking mode — experimental for W4

The pinned template supports thinking. This release defaults it **off in both the
server and terminal client**. Direct mode is the qualified fast-chat profile;
thinking-on W4 is available for experimentation, not a qualified reasoning result.

Using the updated repository client against the existing published container:

```bash
python3 models/MiniCPM5-2B/chat.py --thinking --max-tokens 4096 --timeout 600
```

Run that command from the repository root. The extra budget/timeout flags belong
to the updated host script; the immutable v1.0.0 image still contains its original
client. No image rebuild is required to send these API settings. Omit `--thinking`
for direct answers. The client displays final-answer text, so it may stay quiet
while reasoning is generated.

The same setting works directly through the API:

```bash
curl --fail http://127.0.0.1:8036/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minicpm5-2b","messages":[{"role":"user","content":"Solve 3x + 7 = 22 and check the result."}],"max_tokens":4096,"temperature":0,"chat_template_kwargs":{"enable_thinking":true}}'
```

Set `"enable_thinking":false` to turn it off. The parser separates reasoning from
`message.content`; API clients can inspect `reasoning`/`reasoning_content` and the
corresponding streaming delta fields. Reasoning and the final answer share the
output budget, and prompt plus output must fit the tested 8,192-token context.
Reduce the output budget as conversation history grows. Thinking may repeat or
consume the budget before a useful answer; 4,096 tokens does not guarantee completion.
The published 14/20 quality result and loading/speed figures use thinking **off**.

## Reproduce

[Loading and generation measurements](BENCHMARKS.md) include cache boundaries,
repeats, quality results and rejected approaches. [Build and benchmark commands](REPRODUCE.md)
include stock/Paiton selection. [Licenses and provenance](THIRD_PARTY_NOTICES.md)
cover redistribution and the publisher's quantization metadata limitations.
