# Qwen3.8 27B Qronos on RDNA4

This package serves AMD's public
[`Qwen3.8-27B-Quark-Qronos-INT4-W4A16`](https://huggingface.co/amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16)
checkpoint on one Radeon AI PRO R9700.

## Scope

| Component | Value |
| --- | --- |
| GPU | AMD Radeon AI PRO R9700, `gfx1201`, 32 compute units |
| ROCm | 7.14 |
| Model revision | `649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2` |
| Tensor parallelism | 1 |
| Active batch size | 1 |
| Maximum context | 8,192 tokens |
| Input | Text |

The runtime processes one active request. Additional requests wait in the
server queue. The package is intended for local chat, coding, and generation
by one user.

## Docker

Requirements:

- Linux x86-64
- Docker
- `/dev/kfd` and `/dev/dri` access
- an AMD Radeon AI PRO R9700

Start the server:

```bash
docker run -d \
  --name paiton-qwen38 \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc=host \
  --network host \
  --mount type=volume,src=paiton-qwen38-cache,dst=/models/cache \
  ghcr.io/eliovp/paiton-vllm-plugin:qwen38-qronos-rdna4-v1.3.0
```

The first start downloads the 19.9 GB model from Hugging Face. The named
volume keeps the download. With cached weights, model assembly takes roughly
10 to 12 minutes.

Follow startup:

```bash
docker logs -f paiton-qwen38
```

The server is ready when the log reports `Application startup complete.`

## Launch helper

The repository includes a helper that checks the GPU, Docker access, model
cache, port, and image:

```bash
./models/Qwen3.8/serve-docker.sh
```

Useful options:

```bash
./models/Qwen3.8/serve-docker.sh --dry-run
./models/Qwen3.8/serve-docker.sh --download
./models/Qwen3.8/serve-docker.sh --require-cache
./models/Qwen3.8/serve-docker.sh --with-hf-token
```

AMD's model repository is public. A Hugging Face token is optional. The
`--with-hf-token` option passes an existing token only when requested.

To use an unpacked checkpoint directory:

```bash
docker run --rm \
  --name paiton-qwen38 \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc=host \
  --network host \
  -e PAITON_BASE_MODEL=/models/base \
  -v /absolute/path/to/amd-qwen38:/models/base:ro \
  -v paiton-qwen38-cache:/models/cache \
  ghcr.io/eliovp/paiton-vllm-plugin:qwen38-qronos-rdna4-v1.3.0
```

## Chat

After startup:

```bash
docker exec -it paiton-qwen38 paiton-chat
```

Commands:

- `/reset` clears the conversation
- `/help` shows client help
- `/quit` exits

Thinking mode is disabled by default. Enable it with:

```bash
docker exec -it paiton-qwen38 paiton-chat --thinking
```

## API

The server uses model name `qwen38` and provides an OpenAI-compatible API:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38","messages":[{"role":"user","content":"Explain why the sky is blue."}],"temperature":0,"max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Existing ROCm environment

Local installation requires Python 3.12, ROCm 7.14, PyTorch 2.12 for ROCm
7.14, and vLLM commit
`39bd959b582c85e78e7e0326d49042ce7c3c07ed` built for `gfx1201`.

```bash
python3 -m pip install --no-deps \
  "git+https://github.com/Eliovp-BV/paiton-vllm-plugin.git@paiton-qwen38-qronos-w4a16-gfx1201-v1.3.0"
paiton-qwen38-serve --check-runtime
paiton-qwen38-serve
```

The server downloads the Paiton runtime overlay from the tagged GitHub
release and resolves the original AMD checkpoint from the normal Hugging Face
cache.

## Runtime behavior

The package uses:

- O2 full-and-piecewise graph capture at batch size 1
- model-specific merged projection paths
- AOT decode attention
- tiled recurrent decode
- a compiled W4 language-model head
- fitted MLP decode shadows
- the source checkpoint W4 path for prefill

Fitted decode shadows intentionally change decode quantization. Outputs are
deterministic at temperature zero in the supported configuration, but they are
not specified as bit-identical to the checkpoint's W4 execution path.

The release does not claim support for other GPUs, ROCm versions, larger
batches, tensor parallelism, or multimodal inputs.
