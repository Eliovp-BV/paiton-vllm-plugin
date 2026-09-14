# Paiton Ornith 1.5 on Radeon AI PRO R9700

This package runs the public
[`Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4`](https://huggingface.co/Capicua25x/Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4)
checkpoint on one 32 GB Radeon AI PRO R9700.

## Start

For a one-line local chat experience, clone the public package and let the
helper start the server, wait for it, and open the terminal client:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git && cd paiton-vllm-plugin && ./models/Ornith-1.5/serve-docker.sh --chat
```

To start only the OpenAI-compatible server, run the container directly:

```bash
docker run -d \
  --name paiton-ornith \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc=host \
  -p 8000:8000 \
  --mount type=volume,src=paiton-ornith-cache,dst=/models/cache \
  ghcr.io/eliovp/paiton-vllm-plugin:ornith15-mxfp4-rdna4-v1.0.0
```

The first start downloads the pinned 22.9 GB target checkpoint and its 772 MB
DFlash draft from Hugging Face. Paiton losslessly splits the target checkpoint
into smaller shards once. Allow about 48 GB of free disk space for the download
and ready-to-run copy. The Docker volume preserves both for later starts.

Follow startup:

```bash
docker logs -f paiton-ornith
```

## Chat

After a detached server reports that it is ready:

```bash
docker exec -it paiton-ornith paiton-chat --model ornith
```

Use `/reset` to clear the conversation and `/quit` to exit. The server also
provides an OpenAI-compatible endpoint at
`http://127.0.0.1:8000/v1/chat/completions`.

## Thinking / reasoning mode

The pinned checkpoint's chat template supports thinking, and the packaged server
uses the `qwen3` reasoning parser. The **terminal client disables thinking by
default**. Enable it per chat session with a larger output budget:

```bash
docker exec -it paiton-ornith paiton-chat --model ornith --thinking --max-tokens 4096
```

Omit `--thinking` for direct answers. No server restart or different checkpoint is
required. The terminal client displays the final answer, not the separate reasoning
stream, so it may stay quiet while the model thinks.

API callers should explicitly choose the mode. The server does not impose the
terminal client's thinking-off default; an omitted setting follows the upstream
chat template. To enable thinking:

```bash
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ornith","messages":[{"role":"user","content":"Solve 3x + 7 = 22 and check the result."}],"max_tokens":4096,"temperature":0.6,"chat_template_kwargs":{"enable_thinking":true}}'
```

Set `"enable_thinking":false` for direct mode. Reasoning is returned separately
from `message.content` (in `reasoning` or `reasoning_content`, depending on the
runtime); streaming clients should handle the equivalent delta fields. Do not
assume a `reasoning_effort` setting replaces this template toggle.

The token budget includes reasoning **and** the final answer. Keep prompt plus
output within the configured 8,192-token context; reduce the output budget for
long conversations. A budget exhausted during thinking can leave no final answer.
The 4,096-token example is a starting budget, not a quality or completion guarantee.

The published throughput benchmark explicitly used **thinking disabled**. The toggle is available, but thinking-on quality and throughput are not qualified by those results, including the default DFlash configuration.

## Qualified scope

- Radeon AI PRO R9700 with `gfx1201`
- ROCm 7.14
- one GPU, tensor parallel size 1
- batch size 1 and up to 8,192 tokens
- MXFP4 target model with 16-token DFlash speculation
- public checkpoint revision `9e488f46c0f7969f84c9923ee0256311cd50316e`

DFlash is enabled by default. To serve the same Paiton target without
speculation, add `-e PAITON_ORNITH_DFLASH=0` to `docker run`, or export that
variable before using the helper.

The container downloads model weights directly from the public Hugging Face
repository. It contains no checkpoint weights and no Paiton compiler source.

## Performance

The qualified package averaged 44.628 output tokens per second across two
complete runs on one R9700. This is 27.03% faster than the fastest obtained
stock vLLM result of 35.132 output tokens per second. At equal hardware cost,
that corresponds to 27.03% more output tokens per unit of time and a 21.28%
lower hardware cost per generated token.

The [benchmark record](BENCHMARKS.md) documents the stock settings, Paiton
settings, workload, individual results, and reproduction command.
