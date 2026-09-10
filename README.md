![Paiton: Inference optimization for AMD GPUs](assets/paiton-banner.svg)

<p align="center">
  <a href="#community-releases">Choose a model</a> ·
  <a href="#quick-start">Run locally</a> ·
  <a href="https://github.com/Eliovp-BV/paiton-vllm-plugin/releases">Releases</a> ·
  <a href="https://huggingface.co/EliovpAI">Hugging Face</a> ·
  <a href="https://eliovp.com/products/paiton">Discover Paiton</a>
</p>

**Run language, image and video models locally on AMD GPUs.** Paiton combines
compiler optimizations, custom kernels and runtime integration. These free
community packages include reproducible runtimes, compiled artifacts and setup
guides with measured performance and tested limits.

## Community releases

**Qualified hardware: one Radeon AI PRO R9700 · 32 GB VRAM · RDNA 4 (`gfx1201`).**
Linux and Docker are required; ComfyUI launchers also need Docker Compose.

Choose by the task you want to do:
[Text-only chat](#text-only-chat) · [Text + reasoning](#text--reasoning) ·
[Coding](#coding) · [Image generation](#image-generation) · [Video](#video) ·
[Image understanding and editing](#image-understanding-and-editing).

**GPU memory:** “Tested GPU capacity” is the hardware qualification, not a claim
that every model consumes 32 GB. “Measured GPU use” is sampled driver VRAM at the
linked settings, including runtime overhead. **Minimum capacity on smaller GPUs
has not been verified.** Do not treat checkpoint size or a sampled peak as a
minimum-VRAM guarantee; leave headroom for context, concurrency, resolution and
loading. Host RAM and disk requirements are separate and listed in each guide.

### Text-only chat

| Model & setup guide | Best use / tested interface | Tested GPU capacity | Measured GPU use |
| --- | --- | --- | --- |
| [**MiniCPM5-2B W4A16**](models/MiniCPM5-2B/README.md) | Small, concise chat; lightweight code and tools; 8K context, OpenAI-compatible API | 32 GB | [~4.75 GiB](models/MiniCPM5-2B/BENCHMARKS.md) |
| [**Qwen3.8 27B**](models/Qwen3.8/README.md) | General chat and code; optional thinking (see reasoning below), text input, 8K context | 32 GB | Not reported in the model guide |
| [**Ornith 1.5 35B A3B**](models/Ornith-1.5/README.md) | General chat; 8K context, one active request; DFlash enabled by default | 32 GB | Not reported in the model guide |

MiniCPM5 is the smallest download here: **2.11 GB**, with a **27.98 s median**
prepared-cache launch to a completed useful answer (three trials). Its small
quality suite scored 14/20, so arithmetic and unfamiliar code still need review.
[Loading, quality and generation evidence →](models/MiniCPM5-2B/BENCHMARKS.md)

### Text + reasoning

| Model & setup guide | Best use / tested interface | Tested GPU capacity | Measured GPU use |
| --- | --- | --- | --- |
| [**GPT-OSS-20B**](models/GPT-OSS-20B/README.md) | Reasoning, coding, tool calls and JSON schemas; adjustable reasoning effort, 8K total context | 32 GB | [~17.0 GiB Paiton / 17.1 GiB stock](models/GPT-OSS-20B/BENCHMARKS.md) |
| [**Qwen3.8 27B**](models/Qwen3.8/README.md) | Chat and reasoning with optional thinking mode; disabled by default, 8K context, one active request | 32 GB | Not reported in the model guide |

Models can appear in more than one category: text-only describes the input modality,
while reasoning describes a generation mode. Reasoning consumes the output budget. MiniCPM5's optional thinking mode is
experimental in this package; its qualified default is direct-answer chat.

### Coding

| Model & setup guide | Best use / tested interface | Tested GPU capacity | Measured GPU use |
| --- | --- | --- | --- |
| [**Qwen3-Coder 30B A3B**](models/Qwen3-Coder-30B/README.md) | Dedicated code writing, review and testing; terminal chat or local coding API, 4K context | 32 GB | [~20.1 GiB](models/Qwen3-Coder-30B/BENCHMARKS.md) |

For lightweight code questions, start with MiniCPM5. For coding with explicit
reasoning, see GPT-OSS above. Coding-client compatibility and tool support are
specific to each package; the model guides describe what was tested.

### Image generation

| Model & setup guide | Supported input → output | Tested GPU capacity | Measured GPU use |
| --- | --- | --- | --- |
| [**FLUX.2 klein 4B**](models/FLUX.2-klein/README.md) | Text → image; 1024 × 1024, four steps; ComfyUI, web interface or terminal | 32 GB | [14.6 GiB Paiton / 23.0 GiB stock](models/FLUX.2-klein/BENCHMARKS.md) |

### Video

| Model & setup guide | Supported input → output | Tested GPU capacity | Measured GPU use |
| --- | --- | --- | --- |
| [**FastWan FullAttn 5B**](models/FastWan/README.md) | Text → silent video; three denoiser evaluations, 480/720-class presets | 32 GB | [23.2–30.4 GiB Paiton, by preset](models/Wan2.2/BENCHMARKS.md) |
| [**Wan2.2 TI2V-5B**](models/Wan2.2/README.md) | Text + optional image → silent video; stock is the base workflow default | 32 GB | [23.0–24.4 GiB across measured base cases/engines](models/Wan2.2/BENCHMARKS.md) |
| [**MiniMax H3**](models/MiniMax-H3/README.md) | Text + optional first/last images → video with native stereo audio | 32 GB | [Up to 31.1 GiB across tested presets/engines](models/MiniMax-H3/BENCHMARKS.md) |

### Image understanding and editing

**Text + image → text** (visual chat/captioning) and **text + image → image**
(editing) do not yet have a qualified community package here. Qwen3.8's release
accepts text only; FLUX.2 klein's release qualifies text-to-image only. To animate
an image into video, use Wan2.2 or MiniMax H3 above. Categories describe the
shipped runtime, not every capability of an upstream model family.

## Quick start

Clone the repository once, then choose **one** launcher below:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
```

The first launch downloads and prepares weights and may build or compile the
runtime. Caches persist for subsequent runs. Follow the model guide for readiness
and send a generation request before assuming the model is usable. Run one model
at a time; Wan and FastWan share a service and cache.

<details>
<summary><strong>Text-only chat — MiniCPM5-2B</strong> · Small download, concise answers</summary>

```bash
./models/MiniCPM5-2B/serve-docker.sh
```

In another terminal, from the repository root:

```bash
python3 models/MiniCPM5-2B/chat.py
```

OpenAI-compatible API: `http://127.0.0.1:8036/v1`, model `minicpm5-2b`.
Defaults to thinking off, 8K context and two scheduled requests. Use
`./models/MiniCPM5-2B/serve-docker.sh --download-only` to prepare weights ahead
of time, or `--offline` once cached. Append `--stock` for the matched stock path.

The published container can also run without cloning:

```bash
docker run --rm --name paiton-minicpm5 \
  --device /dev/kfd --device /dev/dri --ipc=host \
  -p 8036:8036 -v paiton-minicpm5-cache:/models/cache \
  ghcr.io/eliovp/paiton-vllm-plugin:minicpm5-2b-w4a16-rdna4-v1.0.0
```

[Setup, quality and loading benchmarks →](models/MiniCPM5-2B/README.md) ·
[Hugging Face artifacts](https://huggingface.co/EliovpAI/MiniCPM5-2B-W4A16-Paiton-RDNA4)

</details>

<details>
<summary><strong>Start Qwen3.8</strong> · Chat, optional reasoning and API</summary>

```bash
./models/Qwen3.8/serve-docker.sh
```

Once the server reports that it is ready, open a terminal chat:

```bash
docker exec -it paiton-qwen38 paiton-chat
```

Thinking is disabled by default. To enable reasoning:

```bash
docker exec -it paiton-qwen38 paiton-chat --thinking
```

API clients can set `chat_template_kwargs.enable_thinking` to `true`.

[Full guide, API examples and existing-environment installation →](models/Qwen3.8/README.md)

</details>

<details>
<summary><strong>Chat with Ornith 1.5</strong> · Chat and API</summary>

```bash
./models/Ornith-1.5/serve-docker.sh --chat
```

The helper starts the server, waits for it to become ready and opens the
terminal chat. Use `/reset` for a new conversation and `/quit` to leave the chat.

[Full guide, API examples and model options →](models/Ornith-1.5/README.md)

</details>

<details>
<summary><strong>Chat, reason and code with GPT-OSS-20B</strong> · 2.18× tested speedup</summary>

```bash
./models/GPT-OSS-20B/serve-docker.sh
```

In another terminal, run `python3 models/GPT-OSS-20B/chat.py` or use the
OpenAI-compatible API on port 8020. Supports streaming, tools and JSON schemas.

[Setup, benchmarks and tested limits →](models/GPT-OSS-20B/README.md)

</details>

<details>
<summary><strong>Code with Qwen3-Coder 30B</strong> · Terminal chat and coding API</summary>

```bash
./models/Qwen3-Coder-30B/serve-docker.sh --chat
```

The prebuilt container downloads and caches the pinned INT4 model on first use.
Coding clients can connect to `http://127.0.0.1:8010/v1`, model `qwen3-coder`.

[Download the bundle](https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/qwen3-coder-30b-awq-rdna4-v1.0.0/paiton-qwen3-coder-r9700-v1.0.0.tar.gz) ·
[Full guide, requirements and coding client settings →](models/Qwen3-Coder-30B/README.md)

</details>

<details>
<summary><strong>Generate images with FLUX.2 klein</strong> · ComfyUI</summary>

```bash
./models/FLUX.2-klein/launch.sh
```

Open [ComfyUI](http://127.0.0.1:8188/?paiton=1), enter a prompt and click **Run**.
The included workflow lets you select **Paiton** or **Stock (Diffusers)**.

[Full guide, requirements and other interfaces →](models/FLUX.2-klein/README.md)

</details>

<details>
<summary><strong>Create fast text-to-video with FastWan 5B</strong> · Three-evaluation generation</summary>

```bash
./models/FastWan/launch.sh
```

Open [ComfyUI](http://127.0.0.1:8192/?paiton=1&preset=fast), enter a prompt and click Run. Choose stock/Paiton, duration and resolution in the connected workflow.

[Setup, measurements and example clips →](models/FastWan/README.md)

</details>

<details>
<summary><strong>Animate images with Wan2.2 TI2V-5B</strong> · Text and optional image input</summary>

```bash
./models/Wan2.2/launch.sh
```

Open [ComfyUI](http://127.0.0.1:8192/?paiton=1&preset=base), upload an optional image, edit the prompt and click Run. The base workflow defaults to stock because its image benchmarks did not show an end-to-end Paiton gain.

[Setup, measurements and example clips →](models/Wan2.2/README.md)

</details>

<details>
<summary><strong>Create videos with MiniMax H3</strong> · ComfyUI with stereo audio</summary>

```bash
./models/MiniMax-H3/launch.sh
```

Open [ComfyUI](http://127.0.0.1:8190/?paiton=1&studio=1), edit the prompt, optionally upload first/last images, set the length and resolution sliders, and click **Run**.
From another system, use `http://<server-ip>:8190/?paiton=1&studio=1` with the host's network address.
The first launch prepares the local runtime and downloads the model; later launches reuse both.

[Generated clips, performance and setup guide →](models/MiniMax-H3/README.md)

</details>

## Performance you can inspect

Results apply to specific workloads and runtime settings. Compiler changes,
quantization and speculative decoding are documented separately in each guide.

| Model | Example measured result | Evidence |
| --- | --- | --- |
| MiniCPM5-2B | Paired warm chat: 0.704 s → 0.471 s median; 33.1% lower latency, with unresolved timing variability | [Loading, generation and quality](models/MiniCPM5-2B/BENCHMARKS.md) |
| GPT-OSS-20B | 512 input / 256 output, concurrency one: 4.819 s → 2.214 s median; 2.18× speedup, 32 requests/mode | [Matched stock settings and caveats](models/GPT-OSS-20B/BENCHMARKS.md) |
| FLUX.2 klein | Sampled driver VRAM: 23.0 GiB stock → 14.6 GiB Paiton at the qualified image settings | [Full-pipeline measurements](models/FLUX.2-klein/BENCHMARKS.md) |

More evidence: [Qwen3.8](https://eliovp.com/blog/paiton-qwen38-radeon-ai-pro-r9700) ·
[Ornith 1.5](models/Ornith-1.5/BENCHMARKS.md) ·
[Qwen3-Coder](models/Qwen3-Coder-30B/BENCHMARKS.md) ·
[FastWan / Wan2.2](models/Wan2.2/BENCHMARKS.md) ·
[MiniMax H3](models/MiniMax-H3/BENCHMARKS.md).
Use the model guides for complete settings, sample counts and reproduction commands.

## Beyond the community releases

Paiton's broader work covers **AMD Instinct (CDNA)** accelerators and multi-GPU
inference for larger language, image and video workloads.

**[Explore Paiton and discuss your workload →](https://eliovp.com/products/paiton)**

## About this repository

This repository distributes public runtimes and compiled artifacts. Paiton's
compiler is developed privately. Text serving uses vLLM; image generation uses
Diffusers with ComfyUI integration. MiniMax H3 uses the native ComfyUI video
pipeline with Paiton artifacts, assembled locally from pinned components.

The vLLM plugin is [Apache-2.0 licensed](LICENSE). Model weights and bundled
components retain their own licenses, including GPL-3.0-only for the separate
image conversion/stock tools and ComfyUI. See the
[third-party notices](THIRD_PARTY_NOTICES.md) and
[image package notices](models/FLUX.2-klein/THIRD_PARTY_NOTICES.md) and
[video package notices](models/MiniMax-H3/THIRD_PARTY_NOTICES.md).

[Release downloads](https://github.com/Eliovp-BV/paiton-vllm-plugin/releases) ·
[Containers](https://github.com/users/Eliovp/packages/container/package/paiton-vllm-plugin) ·
[Hugging Face](https://huggingface.co/EliovpAI) ·
[Report an issue](https://github.com/Eliovp-BV/paiton-vllm-plugin/issues)
