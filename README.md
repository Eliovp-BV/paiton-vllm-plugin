# Paiton vLLM Plugin

Paiton provides model-specific AMD GPU inference paths for vLLM. This
repository contains the public runtime plugin, container definition, launch
tools, licenses, and checks required to run published Paiton artifacts.

## Supported configuration

| Model | GPU | Runtime scope | Instructions |
| --- | --- | --- | --- |
| AMD Qwen3.8 27B Qronos | Radeon AI PRO R9700 (`gfx1201`) | Text, TP1, batch size 1, 8,192-token context | [Qwen3.8 guide](models/Qwen3.8/README.md) |
| Ornith 1.5 35B A3B MXFP4 | Radeon AI PRO R9700 (`gfx1201`) | Text, TP1, batch size 1, 8,192-token context | [Ornith 1.5 guide](models/Ornith-1.5/README.md) |

The source checkpoint is AMD's public
[`Qwen3.8-27B-Quark-Qronos-INT4-W4A16`](https://huggingface.co/amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16)
at revision `649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2`.

The Ornith package uses Capicua25x's public
[`Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4`](https://huggingface.co/Capicua25x/Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4)
checkpoint at revision `9e488f46c0f7969f84c9923ee0256311cd50316e`.

## Start the server

Start Ornith and enter an interactive local chat with one command:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git && cd paiton-vllm-plugin && ./models/Ornith-1.5/serve-docker.sh --chat
```

For the Qwen3.8 server:

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

The first start downloads the 19.9 GB public checkpoint. The Docker volume
keeps it for later starts. Model assembly with cached weights takes roughly
10 to 12 minutes on the supported GPU.

Follow startup:

```bash
docker logs -f paiton-qwen38
```

## Chat locally

After the log reports that the application is ready:

```bash
docker exec -it paiton-qwen38 paiton-chat
```

Use `/reset` to clear the conversation and `/quit` to exit. The server also
provides an OpenAI-compatible endpoint at
`http://127.0.0.1:8000/v1/chat/completions` with model name `qwen38`.

## One-line helper

The checked helper handles cache discovery, device arguments, offline mode,
and the persistent Docker volume:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git && cd paiton-vllm-plugin && ./models/Qwen3.8/serve-docker.sh
```

## Runtime facts

- ROCm 7.14
- vLLM commit `39bd959b582c85e78e7e0326d49042ce7c3c07ed`
- O2 full-and-piecewise graph capture at batch size 1
- Qronos group-128 packed weights
- model-specific projection, attention, recurrent, and language-model-head paths
- fitted MLP decode shadows; source W4 path retained for prefill
- original AMD weights downloaded directly from Hugging Face

The fitted decode shadows intentionally change decode quantization. The
published scope is single-user text inference on the exact GPU and software
configuration above.

## Install the plugin in an existing environment

The environment must already contain ROCm 7.14, PyTorch 2.12 for ROCm 7.14,
and the pinned vLLM revision.

```bash
python3 -m pip install --no-deps \
  "git+https://github.com/Eliovp-BV/paiton-vllm-plugin.git@paiton-qwen38-qronos-w4a16-gfx1201-v1.3.0"
paiton-qwen38-serve --check-runtime
paiton-qwen38-serve
```

## License

The plugin is licensed under Apache-2.0. Third-party notices and retained
license texts are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`LICENSES/`](LICENSES/).
