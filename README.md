![Paiton: Inference optimization for AMD GPUs](assets/paiton-banner.svg)

<p align="center">
  <a href="#model-library">Choose a model</a> ·
  <a href="#quick-start">Run locally</a> ·
  <a href="#reading-the-results">Performance</a> ·
  <a href="https://github.com/Eliovp-BV/paiton-studio">Paiton Studio</a> ·
  <a href="https://github.com/Eliovp-BV/paiton-vllm-plugin/releases">Releases</a> ·
  <a href="https://eliovp.com/products/paiton">Discover Paiton</a>
</p>

**Optimized AI models for AMD GPUs.** Run language models, generate images and
videos, or process meeting recordings locally. This repository is Paiton's
community model library: ready-to-run packages, compiled runtime artifacts,
launchers and measured comparisons against stock execution.

Paiton combines compiler optimizations and native GPU kernels with established
runtimes. Language models use **vLLM**; image and video packages use **Diffusers
and ComfyUI** where appropriate. Each model has its own supported features,
setup guide and reproducible benchmark report.

**Prefer a graphical workspace?** [Paiton Studio](https://github.com/Eliovp-BV/paiton-studio)
brings supported Paiton models into a local workspace for images, video, chat
and writing.

## Model library

Choose a model for your task. The packages below were tested on one
**Radeon AI PRO R9700, 32 GB, RDNA4 (`gfx1201`)**.

The advantage column highlights a measured workload, with its baseline and
metric identified. Percentages are workload-specific; click a result for the
complete comparison. **C1 / C2** mean one / two concurrent requests.

### Chat, reasoning, coding and visual understanding

All six packages expose an OpenAI-compatible vLLM API. Context limits include
both input and generated tokens.

| Model & setup | Use / supported input | Context | Measured GPU use | Measured advantage |
| --- | --- | ---: | ---: | --- |
| [**MiniCPM5-2B W4A16**](models/MiniCPM5-2B/README.md) | Lightweight chat, coding and tools; text | 8K | ~4.75 GiB | [**+54.4% output tok/s** vs stock; C1](models/MiniCPM5-2B/BENCHMARKS.md#sustained-generation-and-prefill) |
| [**Qwen3.8 27B Qronos**](models/Qwen3.8/README.md) | General chat, coding and optional reasoning; text | 8K | Not reported | [**+54.3% output tok/s** vs stock; coding workload, C1](https://eliovp.com/blog/paiton-qwen38-radeon-ai-pro-r9700) |
| [**Qwen3.8 NEO CODER MAX 27B Q4_K_M**](models/Qwen3.8-NEO-CODER-MAX/README.md) | Coding and visual chat; text + one image; native GGUF | 8K | ~23.74 GiB | [**6.4% lower request latency** vs llama.cpp; 128 input / 128 output, C1](models/Qwen3.8-NEO-CODER-MAX/BENCHMARKS.md#matched-text-comparison) |
| [**Ornith 1.5 35B A3B**](models/Ornith-1.5/README.md) | Chat and optional reasoning; text | 8K | Not reported | [**+27.0% output tok/s** vs stock; includes DFlash, C1](models/Ornith-1.5/BENCHMARKS.md) |
| [**GPT-OSS-20B**](models/GPT-OSS-20B/README.md) | Reasoning, coding, tools and JSON schemas; text | 8K | ~17.0 GiB | [**54.0% lower request latency** vs fastest qualified stock reference; 512 input / 256 output, C1](models/GPT-OSS-20B/BENCHMARKS.md) |
| [**Qwen3-Coder 30B A3B**](models/Qwen3-Coder-30B/README.md) | Code writing, review, testing and tools; text | 4K | ~20.1 GiB | [**+70.1% output tok/s** vs stock at C2; **+21.3%** at C1](models/Qwen3-Coder-30B/BENCHMARKS.md) |

MiniCPM5 has the smallest model download here, **2.11 GB**, and a measured
**27.98-second** prepared-cache launch to a completed answer. Its W4 thinking
mode remains experimental; the supported direct-answer mode is the default.

NEO supports the pinned author's mixed GGUF weights directly through Paiton's
vLLM integration, including [single-image requests](models/Qwen3.8-NEO-CODER-MAX/IMAGE_API.md).
Its MTP path is disabled. Qronos and Ornith serve **text only**, even though their
upstream architectures include vision components. See the
[native GGUF guide](models/Qwen3.8-NEO-CODER-MAX/NATIVE_GGUF.md) for that model's
specific support and arithmetic contract.

### Image generation

| Model & setup | Supported workflow | Measured GPU use | Measured advantage over stock |
| --- | --- | ---: | --- |
| [**FLUX.2 klein 4B**](models/FLUX.2-klein/README.md) | Text → image; 1024 × 1024, four steps; ComfyUI, web or CLI | ~14.6 GiB | [**36.7% less sampled GPU memory**; **16.2% lower generation latency**](models/FLUX.2-klein/BENCHMARKS.md) |

FLUX timings include text encoding, generation and image conversion; PNG writing
and UI transport are excluded. Image editing is not qualified in this package.

### Video generation

| Model & setup | Supported workflow | Measured GPU use | Measured advantage over stock |
| --- | --- | ---: | --- |
| [**FastWan FullAttn 5B**](models/FastWan/README.md) | Text → silent video; three denoiser evaluations, 480/720-class presets | 23.2–30.4 GiB | [**Up to 4.7% lower complete-clip latency**; 832 × 480, 49 frames](models/Wan2.2/BENCHMARKS.md#complete-measured-results) |
| [**Wan2.2 TI2V-5B**](models/Wan2.2/README.md) | Text + optional image → silent video | 23.0–24.4 GiB | [**1.1% lower latency for text input**; image cases **0.1–0.8% slower**, so stock is the default](models/Wan2.2/BENCHMARKS.md#complete-measured-results) |
| [**MiniMax H3**](models/MiniMax-H3/README.md) | Text + optional first/last images → video with native stereo audio | Up to 31.1 GiB | [**16.7% lower complete-clip latency**; continuous 15.08-second Turbo8 video](models/MiniMax-H3/BENCHMARKS.md#continuous-15-second-qualification) |

Wan and FastWan share one runtime and cache. FastWan qualifies text input only;
use Wan2.2 or MiniMax H3 when starting from an image. Video timings above include
file encoding; gains from distillation or fewer sampling steps are not counted
as Paiton acceleration.

### Meeting recordings

| Package & setup | Supported workflow | Measured advantage over stock |
| --- | --- | --- |
| [**Local meeting notes — review candidate**](models/Meeting/README.md) | Imported recording → transcript, anonymous speaker labels and partial notes with timestamps; CLI/container | [**2.2% lower complete processing time**; 291.03 → 284.50 s on a 39-minute meeting](models/Meeting/BENCHMARKS.md#final-candidate-complete-matched-comparison) |

The package combines Parakeet speech recognition, speaker diarization and a
compact Granite summary model. Notes require review against the recording;
coverage is incomplete. Live Teams capture and Studio integration are outside
this package's supported workflow.

## Quick start

Clone the library, then choose **one** model launcher:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
```

<details>
<summary><strong>Language and visual-chat models</strong></summary>

- **MiniCPM5:** `./models/MiniCPM5-2B/serve-docker.sh` — then `python3 models/MiniCPM5-2B/chat.py`; API port **8036**, model `minicpm5-2b`.
- **Qronos Qwen3.8:** `./models/Qwen3.8/serve-docker.sh` — once ready, `docker exec -it paiton-qwen38 paiton-chat`; API port **8000**, model `qwen38`.
- **NEO CODER MAX:** `./models/Qwen3.8-NEO-CODER-MAX/serve-docker.sh` — API port **8000**, model `qwen38-neo`; [chat and image examples](models/Qwen3.8-NEO-CODER-MAX/README.md#launch).
- **Ornith 1.5:** `./models/Ornith-1.5/serve-docker.sh --chat` — opens terminal chat after startup; API port **8000**, model `ornith`.
- **GPT-OSS-20B:** `./models/GPT-OSS-20B/serve-docker.sh` — then `python3 models/GPT-OSS-20B/chat.py`; API port **8020**, model `gpt-oss-20b`.
- **Qwen3-Coder:** `./models/Qwen3-Coder-30B/serve-docker.sh --chat` — terminal chat and coding API; port **8010**, model `qwen3-coder`.

The model guides document streaming, tool support and reasoning controls.
Thinking tokens use the same output budget as the final answer; choose the mode
explicitly when comparing responses or benchmarking.

</details>

<details>
<summary><strong>Image and video generation</strong></summary>

- **FLUX.2 klein:** `./models/FLUX.2-klein/launch.sh` — open [ComfyUI on port 8188](http://127.0.0.1:8188/?paiton=1).
- **FastWan:** `./models/FastWan/launch.sh` — open [ComfyUI on port 8192](http://127.0.0.1:8192/?paiton=1&preset=fast).
- **Wan2.2:** `./models/Wan2.2/launch.sh` — open [ComfyUI on port 8192](http://127.0.0.1:8192/?paiton=1&preset=base).
- **MiniMax H3:** `./models/MiniMax-H3/launch.sh` — open [ComfyUI on port 8190](http://127.0.0.1:8190/?paiton=1&studio=1).

The included workflows expose the supported prompts, inputs and generation
settings. Each model guide covers outputs, cache locations and server access.

</details>

<details>
<summary><strong>Meeting recordings</strong></summary>

Follow the [meeting setup guide](models/Meeting/REPRODUCE.md) to prepare the
models and pull the published image. Then, from `models/Meeting`:

```bash
./run-docker.sh --paiton /path/to/meeting.mp4 /path/to/new-result
```

Community-1 requires your own approved Hugging Face access for the initial model
download. Inference runs offline; the original recording is mounted read-only.

</details>

### Requirements and first launch

Use Linux with Docker and AMD GPU device access. The ComfyUI launchers also
require Docker Compose. Run one model at a time on the tested single-GPU setup.

The first launch downloads weights and may build or compile runtime components;
subsequent launches reuse persistent caches. Wait for the model's readiness
message before sending a request. Host RAM, disk space and preparation times
vary substantially—check the linked model guide before downloading.

The memory figures above are sampled driver VRAM at the reported settings,
including runtime overhead. They are not minimum-capacity guarantees. Smaller
GPUs have not been qualified, and context, concurrency and image/video resolution
can change memory requirements.

## Reading the results

Every advantage links to a report with the baseline, workload, sampling settings,
repetitions, quality checks and known limitations. Compare stock and Paiton
**within the same model and workload**; these rows are not a ranking across models.

- **Throughput:** `(Paiton tok/s ÷ baseline tok/s − 1) × 100`. MiniCPM's highlighted comparison is **127.6 → 197.0 tok/s**; Qwen3-Coder's C2 result is **101.61 → 172.82 tok/s**. These are aggregate output rates, not individual-stream decode rates.
- **Latency reduction:** `(baseline time − Paiton time) ÷ baseline time × 100`. A 50% latency reduction means twice the rate for equivalent fixed work, not a 50% throughput increase.
- **Baseline matters:** NEO is compared with **llama.cpp**, not stock vLLM. Its longer 128-output text workloads show 5.1% and 0.8% lower request latency; some prefill-only cases favor llama.cpp. GPT-OSS uses the fastest qualified **4.819-second stock reference** because stock timing varied between runs.
- **Optimization scope matters:** Ornith's result includes DFlash speculative decoding. Quantization, activation arithmetic and quality differences are described per model. MiniCPM's repeated timings show unresolved variability; meeting results also have substantial variation.

## About Paiton

This repository distributes the public integrations and compiled runtime
artifacts. **The Paiton compiler remains proprietary.** Model weights are
retrieved from their pinned publishers and retain their own licenses.

The vLLM plugin is [Apache-2.0 licensed](LICENSE). Bundled components retain
their applicable licenses, including those for ComfyUI and separate image tools.
See the [root notices](THIRD_PARTY_NOTICES.md) and each model's notices.

Paiton's broader work includes AMD Instinct accelerators and multi-GPU inference.
[Explore Paiton or discuss your workload →](https://eliovp.com/products/paiton)

[Release downloads](https://github.com/Eliovp-BV/paiton-vllm-plugin/releases) ·
[Containers](https://github.com/users/Eliovp/packages/container/package/paiton-vllm-plugin) ·
[Hugging Face](https://huggingface.co/EliovpAI) ·
[Report an issue](https://github.com/Eliovp-BV/paiton-vllm-plugin/issues)
