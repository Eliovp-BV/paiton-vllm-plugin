# Agentic v1.1.0 launch instructions

Both GHCR images are published and their immutable pulls are verified. Digests, profile hashes and publication records are in [runtime.agentic-v1.1.0.lock.json](runtime.agentic-v1.1.0.lock.json).

These images run the pinned Qwen3.8-27B MXFP4 target and compatible DFlash2 through regular vLLM and the Paiton plugin on one Radeon AI PRO R9700 with 32 GB VRAM. They contain the unchanged native runtime from v1.0.0. End users need no Paiton compiler or separate Radiance distribution.

The new defaults select Qwen XML tool-call parsing, the Qwen reasoning parser and thinking disabled. Individual requests can enable thinking with `chat_template_kwargs: {"enable_thinking": true}`. Model weights are downloaded from the original pinned repositories and hash-verified; they are not bundled in the image.

## Choose one server

Run only one profile on the GPU at a time. Both commands use the same persistent model-cache volume and localhost port 8000. Stop the chosen container before switching profiles.

| Profile | Total context limit | Scheduler sequence limit | Explicit FP8 cache |
|---|---:|---:|---:|
| 64K | 65,536 tokens | 8 | 5 GiB |
| 200K | 200,000 tokens | 1 | 8 GiB |

Context means **the tokenized prompt plus generated output**, including chat and tool-template tokens. Reserve output space within the limit. Increasing the limit does not automatically provide room for more simultaneous requests.

### 64K

This profile retains the smaller cache budget and allows up to eight scheduled sequences. It does not fit eight full-length 64K requests simultaneously; long prompts reduce resident concurrency and can cause queueing or recomputation.

```bash
docker run --rm -d --name paiton-qwen38-agentic-64k \
  --device /dev/kfd --device /dev/dri --group-add video \
  --shm-size 2g -p 127.0.0.1:8000:8000 \
  -v paiton-qwen38-mxfp4-cache:/models/cache \
  ghcr.io/eliovp/paiton-vllm-plugin@sha256:c3ec2528285b484b2e0af7f571f80f1da23c1970210e4d3d09f5d2a909414186
```

Version tag: `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-mxfp4-dflash2-rdna4-v1.1.0`.

### 200K

This profile supports **one active request**. Additional requests wait. The 8 GiB cache and existing recurrent-state buffers leave little spare VRAM on a 32 GB card. Keep the packaged 4,096-token chunked-prefill setting; increasing prefill batch size or concurrency also needs working memory and is outside this profile's validation.

A synthetic 195,999-token prompt with three-marker retrieval and a short response was tested. This is functional evidence, not a broad long-context quality evaluation or a performance benchmark. For a first application test, a prompt at or below 196,000 tokens with a 512-token output budget leaves some space before the configured boundary.

```bash
docker run --rm -d --name paiton-qwen38-agentic-200k \
  --device /dev/kfd --device /dev/dri --group-add video \
  --shm-size 2g -p 127.0.0.1:8000:8000 \
  -v paiton-qwen38-mxfp4-cache:/models/cache \
  ghcr.io/eliovp/paiton-vllm-plugin@sha256:28af1731cfd8aceab51915411e8f879531128c2711b4193c76c5b6eba8ba2ef4
```

Version tag: `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-mxfp4-dflash2-rdna4-200k-v1.1.0`.

The checkpoint declares 262,144 positions, but that is a model configuration limit, not a promise that the entire serving stack fits that length on this 32 GB GPU. The packaged 200K setting is the tested resource configuration; longer limits remain unqualified.

## Connect and switch

Wait for startup and model verification to finish, then check readiness:

```bash
curl --fail http://127.0.0.1:8000/health
```

Use OpenAI-compatible base URL `http://127.0.0.1:8000/v1` and model name `Qwen3.8-27B-Quark-AWQ-MXFP4`.

To switch from 64K to 200K, stop `paiton-qwen38-agentic-64k` before running the 200K command. To switch back, stop `paiton-qwen38-agentic-200k` first. The shared volume retains model downloads. Append `--offline` after the image reference only when both pinned snapshots are already cached.

The original image and tag remain available unchanged:

`ghcr.io/eliovp/paiton-vllm-plugin:qwen38-mxfp4-dflash2-rdna4-v1.0.0`

The historical repository `serve.py` continues to select its existing release lock. The direct image commands above select these new profiles explicitly.

Existing third-party notices and component licenses remain applicable; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This draft does not extend the old throughput benchmark claims to the new context settings.
