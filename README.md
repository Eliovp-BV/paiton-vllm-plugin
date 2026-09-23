<h1 align="center">More from your AMD GPU.</h1>

<p align="center">
  <strong>Chat. Code. Create images and video. Run locally.</strong><br>
  Optimized model packages, familiar runtimes and measured results.
</p>

<p align="center">
  <a href="#model-library"><img src="https://img.shields.io/badge/Tested_on-Radeon_AI_PRO_R9700-ED1C24?style=flat-square" alt="Tested on Radeon AI PRO R9700"></a>
  <a href="#model-library"><img src="https://img.shields.io/badge/Architecture-RDNA4_%C2%B7_32_GB-30363D?style=flat-square" alt="RDNA4 · 32 GB"></a>
  <a href="#chat-reasoning-coding-and-visual-understanding"><img src="https://img.shields.io/badge/LLM_API-OpenAI_compatible-30363D?style=flat-square" alt="OpenAI-compatible language-model API"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Plugin_license-Apache_2.0-30363D?style=flat-square" alt="Plugin license: Apache 2.0"></a>
</p>

<p align="center">
  <a href="#quick-start"><strong>Get started</strong></a> ·
  <a href="#model-library">Explore models</a> ·
  <a href="#native-cli">Native CLI</a> ·
  <a href="#reading-the-results">Benchmarks</a> ·
  <a href="https://github.com/Eliovp-BV/paiton-studio">Paiton Studio</a> ·
  <a href="https://github.com/Eliovp-BV/paiton-vllm-plugin/releases">Releases</a>
</p>

Paiton is a community model library for local AI on AMD GPUs. Run language
models, generate images and video, or process meeting recordings using
model-specific packages, compiled runtime artifacts and ready-to-run launchers.

Under the hood, Paiton combines compiler optimizations and native GPU kernels
with established runtimes: **vLLM** for language models, and **Diffusers and
ComfyUI** for image and video packages where appropriate. Every model has a setup
guide, a defined support scope and a reproducible benchmark report.

> [!TIP]
> Prefer a graphical workspace? [Paiton Studio](https://github.com/Eliovp-BV/paiton-studio)
> brings supported models together for local chat, writing, images and video.

## Performance at a glance

Measured on one Radeon AI PRO R9700. Three examples from the model library:

<table>
  <tr>
    <td align="center" valign="top" width="33%">
      <h3>+70.1%</h3>
      <strong>Output throughput</strong>
      <p>Qwen3-Coder 30B A3B<br>vs stock · two concurrent requests</p>
      <a href="models/Qwen3-Coder-30B/BENCHMARKS.md">See the benchmark →</a>
    </td>
    <td align="center" valign="top" width="34%">
      <h3>154.42 tok/s</h3>
      <strong>Weighted decode</strong>
      <p>Qwen3.8 27B MXFP4 + DFlash2<br>65K profile</p>
      <a href="models/Qwen3.8-MXFP4-DFlash2/README.md#current-benchmark-results">See the benchmark →</a>
    </td>
    <td align="center" valign="top" width="33%">
      <h3>36.7% less</h3>
      <strong>Sampled GPU memory</strong>
      <p>FLUX.2 klein 4B vs stock<br>1024 × 1024 · four steps</p>
      <a href="models/FLUX.2-klein/BENCHMARKS.md">See the benchmark →</a>
    </td>
  </tr>
</table>

These are different workloads, not a cross-model ranking or a universal
speedup claim. Baselines, settings and limitations are linked with each result.
[How to read the numbers →](#reading-the-results)

## Quick start

Choose the deployment path that fits your setup:

| Deployment | What you need | Start here |
| --- | --- | --- |
| **Native CLI** | An existing vLLM environment matching the model's [supported runtime](#native-serving-presets). | [Install and run `paiton serve NAME`](#run-in-your-vllm-environment) |
| **Managed containers** | Linux, Docker and AMD GPU device access. Model launchers supply their own inference environment. | [Run a model container](#run-with-a-managed-container) |

### Run in your vLLM environment

Activate MiniCPM5's [supported vLLM environment](models/MiniCPM5-2B/README.md#native-serving), then install and launch:

```bash
python -m pip install https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/v0.3.4/paiton_vllm_plugin-0.3.4-py3-none-any.whl
paiton serve minicpm5
```

Paiton downloads and verifies the matching native bundle automatically. Existing
weights in your Hugging Face cache are reused; missing checkpoint files are
downloaded from the pinned publisher. To reuse weights elsewhere, run
`paiton --model-dir /path/to/minicpm5 serve minicpm5`.

Paiton preserves your installed vLLM, Torch and ROCm versions. If you need a
complete inference environment, use a [managed container](#run-with-a-managed-container).

The first launch downloads weights and may build runtime components. Wait for
the readiness message, then send a request from another terminal:

```bash
curl http://127.0.0.1:8036/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minicpm5-2b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

See the [model guide](models/MiniCPM5-2B/README.md) for terminal chat and tools.
Run one model at a time on the tested single-GPU setup.

### Run with a managed container

MiniCPM5 is the smallest model download in this library. Check its
[model guide](models/MiniCPM5-2B/README.md) for host RAM and disk requirements:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
./models/MiniCPM5-2B/serve-docker.sh
```

Once ready, use the same API request above, or run
`python3 models/MiniCPM5-2B/chat.py` from another terminal.

[Choose another model →](#model-library) ·
[All container launchers →](#managed-containers) ·
[Use your existing vLLM environment →](#native-cli)

## Model weights and existing downloads

**Choose where your weights come from before launching a model.** Cloning this
repository downloads the launchers and documentation, not the model weights.
Creating a directory with `mkdir` does not download a checkpoint.

| Your situation | What to do |
| --- | --- |
| **First download** | Follow the model's **First download** instructions. Most launchers download their pinned weights automatically; Qwen3.8 MXFP4 + DFlash2 and Meeting require a preparation step. |
| **Already in a local folder** | Follow **Already in a local folder** and supply the absolute path to your complete checkpoint. Use the exact model and quantization listed in that guide. |
| **Already in the Hugging Face cache** | Follow **Already in the Hugging Face cache**. A download made with `hf download REPOSITORY` without `--local-dir` normally lives here. Containers need the cache explicitly mounted or prepared as described for that model. |

For the [native language-model CLI](#native-cli), a complete local MiniCPM5
checkpoint can be selected explicitly:

```bash
paiton --model-dir /absolute/path/to/minicpm5 serve minicpm5
```

Without `--model-dir`, the native CLI uses its remembered model location, or
resolves the pinned checkpoint from the Hugging Face cache and downloads missing
files. This behavior does not apply automatically to every Docker launcher.

The default Hub cache is `$HOME/.cache/huggingface/hub`. `HF_HOME`,
`HF_HUB_CACHE`, or `XDG_CACHE_HOME` can change it. A **Hub cache** contains
`models--OWNER--MODEL` directories; a **checkpoint folder** contains that model's
files. They are different inputs. See the [cache and path guide](docs/MODEL_WEIGHTS.md)
for locating your cache, exact revisions, and Docker snapshot links.

For example, to run the MiniCPM5 container with weights already downloaded to
your Hub cache, mount that cache read-only and keep runtime files in a separate
Docker volume:

```bash
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
# Or use: export HF_HUB_CACHE="/absolute/path/to/your/hub-cache"
docker run --rm --name paiton-minicpm5-cached \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -p 127.0.0.1:8036:8036 \
  --mount type=volume,src=paiton-minicpm5-cached-runtime,dst=/models/cache \
  --mount "type=bind,src=$HF_HUB_CACHE,dst=/models/cache/huggingface/hub,readonly" \
  ghcr.io/eliovp/paiton-vllm-plugin:minicpm5-2b-w4a16-rdna4-v1.0.0 --offline
```

This example needs the complete pinned MiniCPM5 snapshot. For other models or
a standalone model folder, use the matching guide below; container paths and
companion models differ. No model files are copied by this bind mount.

Every model guide has instructions for all three starting points:

| Model | Weight setup |
| --- | --- |
| MiniCPM5-2B | [Download or reuse weights](models/MiniCPM5-2B/README.md#model-weights-and-existing-downloads) |
| Qwen3.8 Qronos | [Download or reuse weights](models/Qwen3.8/README.md#model-weights-and-existing-downloads) |
| Qwen3.8 MXFP4 + DFlash2 | [Download or reuse target and draft](models/Qwen3.8-MXFP4-DFlash2/README.md#model-weights-and-existing-downloads) |
| Qwen3.8 NEO CODER MAX | [Download or reuse GGUF and projector](models/Qwen3.8-NEO-CODER-MAX/README.md#model-weights-and-existing-downloads) |
| Ornith 1.5 | [Download or reuse weights](models/Ornith-1.5/README.md#model-weights-and-existing-downloads) |
| GPT-OSS-20B | [Download or reuse weights](models/GPT-OSS-20B/README.md#model-weights-and-existing-downloads) |
| Qwen3-Coder 30B | [Download or reuse weights](models/Qwen3-Coder-30B/README.md#model-weights-and-existing-downloads) |
| FLUX.2 klein | [Download or reuse source and prepared tensors](models/FLUX.2-klein/README.md#model-weights-and-existing-downloads) |
| FastWan | [Download or reuse the denoiser and shared components](models/FastWan/README.md#model-weights-and-existing-downloads) |
| Wan2.2 | [Download or reuse the denoiser, encoder and VAE](models/Wan2.2/README.md#model-weights-and-existing-downloads) |
| MiniMax H3 | [Download or reuse the selected model components](models/MiniMax-H3/README.md#model-weights-and-existing-downloads) |
| Meeting | [Download or reuse the prepared speech and summary models](models/Meeting/README.md#model-weights-and-existing-downloads) |

## Model library

**Tested hardware: one Radeon AI PRO R9700 · 32 GB · RDNA4 (`gfx1201`).**
Choose a model by workflow, then follow its guide for the exact supported setup.

Context limits include input and generated tokens. GPU memory figures are
sampled driver VRAM, including runtime overhead—not minimum-capacity guarantees.
Smaller GPUs have not been qualified. **C1, C2 and C8** mean one, two and eight
concurrent requests.

### Chat, reasoning, coding and visual understanding

All seven packages expose an OpenAI-compatible vLLM API.

| Model & setup | Workflow / supported input | Context | Measured GPU use |
| --- | --- | ---: | ---: |
| [**MiniCPM5-2B**](models/MiniCPM5-2B/README.md)<br>W4A16 | Lightweight chat, coding and tools<br>Text | 8K | ~4.75 GiB |
| [**Qwen3.8 27B Qronos**](models/Qwen3.8/README.md) | General chat, coding and optional reasoning<br>Text | 8K | Not reported |
| [**Qwen3.8 27B MXFP4 + DFlash2**](models/Qwen3.8-MXFP4-DFlash2/README.md) | Chat, coding and tools<br>Text | 65K / 200K | Not reported |
| [**Qwen3.8 NEO CODER MAX 27B**](models/Qwen3.8-NEO-CODER-MAX/README.md)<br>Q4_K_M · native GGUF | Coding and visual chat<br>Text + one image | 8K | ~23.74 GiB |
| [**Ornith 1.5 35B A3B**](models/Ornith-1.5/README.md) | Chat and optional reasoning<br>Text | 8K | Not reported |
| [**GPT-OSS-20B**](models/GPT-OSS-20B/README.md) | Reasoning, coding, tools and JSON schemas<br>Text | 8K | ~17.0 GiB |
| [**Qwen3-Coder 30B A3B**](models/Qwen3-Coder-30B/README.md) | Code writing, review, testing and tools<br>Text | 4K | ~20.1 GiB |

**Small download, lightweight serving.** MiniCPM5 has a **2.11 GB** model download
and a measured **27.98-second** prepared-cache launch to a completed answer.
Direct-answer mode is the supported default; W4 thinking remains experimental.

**Long-context Qwen3.8.** Public ROCm 10 / vLLM 0.29.0 images support
65,536-token and 200,000-token profiles. The 200K profile is qualified for
one active request; the reported throughput figures below come from the
65K MXFP4 + DFlash2 profile, not the separate non-speculative native preset.
[Launch the current release →](models/Qwen3.8-MXFP4-DFlash2/README.md#run-the-current-release)

<details>
<summary><strong>Language-model benchmarks — results, baselines and workloads</strong></summary>

| Model | Measured result | Comparison / workload |
| --- | --- | --- |
| MiniCPM5-2B | [**+54.4% output tok/s**](models/MiniCPM5-2B/BENCHMARKS.md#sustained-generation-and-prefill) | vs stock · C1 |
| Qwen3.8 Qronos | [**+54.3% output tok/s**](https://eliovp.com/blog/paiton-qwen38-radeon-ai-pro-r9700) | vs stock · coding workload · C1 |
| Qwen3.8 MXFP4 + DFlash2 | [**154.42 tok/s weighted decode**<br>**421.20 tok/s aggregate at C8**](models/Qwen3.8-MXFP4-DFlash2/README.md#current-benchmark-results) | 65K profile · one R9700 at 300 W |
| Qwen3.8 NEO CODER MAX | [**6.4% lower request latency**](models/Qwen3.8-NEO-CODER-MAX/BENCHMARKS.md#matched-text-comparison) | vs llama.cpp · 128 input / 128 output · C1 |
| Ornith 1.5 | [**+27.0% output tok/s**](models/Ornith-1.5/BENCHMARKS.md) | vs stock · includes DFlash · C1 |
| GPT-OSS-20B | [**54.0% lower request latency**](models/GPT-OSS-20B/BENCHMARKS.md) | vs fastest qualified stock reference · 512 input / 256 output · C1 |
| Qwen3-Coder | [**+70.1% output tok/s at C2**<br>**+21.3% at C1**](models/Qwen3-Coder-30B/BENCHMARKS.md) | vs stock |

The Qwen3.8 65K profile also measured **218.1 tok/s median JSON decode**.
Its weighted decode, median JSON decode and aggregate throughput are distinct
metrics; consult the [full report](models/Qwen3.8-MXFP4-DFlash2/benchmarks/2026-09-20-combined/README.md)
for their workloads and settings.

</details>

<details>
<summary><strong>Vision, GGUF and reasoning support</strong></summary>

NEO supports the pinned author's mixed GGUF weights directly through Paiton's
vLLM integration, including [single-image requests](models/Qwen3.8-NEO-CODER-MAX/IMAGE_API.md).
Its MTP path is disabled. The [native GGUF guide](models/Qwen3.8-NEO-CODER-MAX/NATIVE_GGUF.md)
describes its specific support and arithmetic contract.

Qronos and Ornith are text-only in these packages, even though their upstream
architectures include vision components. MiniCPM5's W4 thinking mode remains
experimental; use the supported direct-answer default.

The model guides document streaming, tool support and reasoning controls.
Thinking tokens share the output budget with the final answer, so choose the
mode explicitly when comparing responses or benchmarking.

</details>

### Image generation

| Model & setup | Supported workflow | Measured GPU use | Measured result |
| --- | --- | ---: | --- |
| [**FLUX.2 klein 4B**](models/FLUX.2-klein/README.md) | Text → image<br>1024 × 1024 · four steps<br>ComfyUI, web or CLI | ~14.6 GiB | [**36.7% less sampled GPU memory**<br>**16.2% lower generation latency**](models/FLUX.2-klein/BENCHMARKS.md) |
| [**Qwen-Image-2.1 MXFP4**](models/Qwen-Image-2.1/README.md) | Text → image or transparent RGBA; image editing<br>2048 × 2048 generation, 1024 × 1024 editing · 40 steps<br>API or CLI | Up to 25.62 GiB | [**103.64 s warm complete-request median**<br>v1.0.2 default profile · matched pairs · 136.65 s bit-exact profile](models/Qwen-Image-2.1/BENCHMARKS.md#release-v102) |

FLUX timings include text encoding, generation and image conversion, but exclude
PNG writing and UI transport; its image editing path is not qualified.
Qwen-Image timings include the complete HTTP response and PNG serialization.
Qwen-Image v1.0.2 runs an int8 precision schedule after seven bit-exact steps by default
(136.65 → 103.64 s warm in matched pairs against its bit-exact profile, which itself measured
165.588 → 136.285 s against v1.0.1); `--precision-profile exact` restores the bit-exact path.

Qwen-Image checkpoints are available in two packages:
[the optimized RDNA4 package](https://huggingface.co/EliovpAI/Qwen_Image-2.1-MXFP4-Paiton-RDNA4)
used by this container, and
[the portable MXFP4 package](https://huggingface.co/EliovpAI/Qwen_Image-2.1-MXFP4)
for compatible framework runtimes beyond RDNA4. Follow the portable model card
for its setup and hardware measurements; those are separate from the R9700 results here.

Start the optimized Qwen-Image API with one command; model files download from
the pinned [RDNA4 Hugging Face checkpoint](https://huggingface.co/EliovpAI/Qwen_Image-2.1-MXFP4-Paiton-RDNA4)
on first use:

```sh
docker run --rm --name paiton-qwen-image21 --device /dev/kfd --device /dev/dri --ipc=host -p 127.0.0.1:8191:8191 -v paiton-qwen-image21-cache:/cache ghcr.io/eliovp/paiton-vllm-plugin:qwen-image21-mxfp4-rdna4-v1.0.2
```

[Generate, edit and save images →](models/Qwen-Image-2.1/README.md#generate-an-image)

### Video generation

| Model & setup | Supported workflow | Measured GPU use | Result vs stock |
| --- | --- | ---: | --- |
| [**FastWan FullAttn 5B**](models/FastWan/README.md) | Text → silent video<br>Three denoiser evaluations · 480/720-class presets | 23.2–30.4 GiB | [**Up to 4.7% lower complete-clip latency**<br>832 × 480 · 49 frames](models/Wan2.2/BENCHMARKS.md#complete-measured-results) |
| [**Wan2.2 TI2V-5B**](models/Wan2.2/README.md) | Text + optional image → silent video | 23.0–24.4 GiB | [**1.1% lower text-input latency**<br>Image cases 0.1–0.8% slower; stock is the default](models/Wan2.2/BENCHMARKS.md#complete-measured-results) |
| [**MiniMax H3**](models/MiniMax-H3/README.md) | Text + optional first/last images → video with native stereo audio | Up to 31.1 GiB | [**16.7% lower complete-clip latency**<br>Continuous 15.08-second Turbo8 video](models/MiniMax-H3/BENCHMARKS.md#continuous-15-second-qualification) |

**Starting from an image?** Use Wan2.2 or MiniMax H3. FastWan qualifies text input
only; Wan and FastWan share one runtime and cache.

Video timings include file encoding. Gains from distillation or fewer sampling
steps are not counted as Paiton acceleration.

### Meeting recordings

[**Local meeting notes · review candidate**](models/Meeting/README.md)

Turn an imported recording into a transcript, anonymous speaker labels and
partial notes with timestamps using the CLI or container. The package combines
Parakeet speech recognition, speaker diarization and a compact Granite summary model.

[**2.2% lower complete processing time**](models/Meeting/BENCHMARKS.md#final-candidate-complete-matched-comparison)
vs stock: 291.03 → 284.50 seconds for a 39-minute meeting.

> [!IMPORTANT]
> Notes have incomplete coverage and must be reviewed against the recording.
> Live Teams capture and Paiton Studio integration are outside this package's
> supported workflow. The measured processing times also show substantial variation.

## Native CLI

**Keep your supported vLLM environment. Install Paiton. Launch a model by name.**

Install the [v0.3.4 wheel](https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/tag/v0.3.4) using the command in
[Quick start](#run-in-your-vllm-environment). The CLI fetches the matching native
bundle on first use and verifies it against the installed catalogue.
Activate the model's supported runtime first; Paiton does not switch environments.

For language-model APIs—including NEO's image-capable chat API—the launch command
is `paiton serve NAME`. For example:

```bash
paiton serve minicpm5
```

Native serving needs neither the private compiler nor a repository checkout.
Qwen-Image, FLUX, MiniMax H3, Wan and FastWan use their [existing launchers](#managed-containers)
for image and video generation instead.

<details>
<summary><strong>One-time installation and preparation — MiniCPM5 example</strong></summary>

In MiniCPM5's supported environment, install Paiton and optionally prepare
your existing checkpoint before starting the server:

```bash
python -m pip install https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/v0.3.4/paiton_vllm_plugin-0.3.4-py3-none-any.whl
paiton doctor
paiton --model-dir /models/existing-minicpm5 \
  --prepare-only serve minicpm5
```

Preparation verifies and remembers the existing checkpoint path and caches the
native bundle. Subsequent launches need only `paiton serve minicpm5`. Once ready,
use the same API request shown in [Quick start](#quick-start).

Using an existing Hugging Face cache? Omit `--model-dir`; the preset resolves its
pinned repository and downloads only missing files. To prohibit downloads and
outbound runtime networking, put `--offline` before `serve`:

```bash
paiton --offline serve minicpm5
```

`paiton models` shows the installed catalogue and preparation status. Read the
[native execution guide](docs/NATIVE_EXECUTION.md) for checkpoint identities,
profile effects, preparation, offline use and lockfiles.

</details>

### Native serving presets

Each preset selects a specific checkpoint, bundle, runtime and supported profile.
First launch downloads and prepares any missing files automatically.
All entries target one 32 GB R9700. Check the linked model guide before preparing.

<details>
<summary><strong>View all seven presets, API names and required environments</strong></summary>

| Model guide | Launch command | API port / model name | Existing environment |
| --- | --- | --- | --- |
| [MiniCPM5](models/MiniCPM5-2B/README.md#native-serving) | `paiton serve minicpm5` | **8036** / `minicpm5-2b` | Python 3.14 · ROCm 7.14 · pinned vLLM 0.26.1 build |
| [Qwen3.8 Qronos](models/Qwen3.8/README.md#native-serving) | `paiton serve qwen38-qronos` | **8000** / `qwen38` | Python 3.12 · ROCm 7.14 · pinned vLLM 0.28.0 build |
| [Qwen3.8 NVFP4](models/Qwen3.8-MXFP4-DFlash2/README.md#native-serving) | `paiton serve qwen38-nvfp4` | **18982** / `Qwen3.8` | Python 3.12 · ROCm 10 · vLLM 0.29.0 |
| [Qwen3.8 NEO](models/Qwen3.8-NEO-CODER-MAX/README.md#native-serving) | `paiton serve qwen38-neo` | **8000** / `qwen38-neo` | Python 3.12 · ROCm 7.14 · pinned vLLM 0.28.0 build |
| [Ornith 1.5](models/Ornith-1.5/README.md#native-serving) | `paiton serve ornith` | **8000** / `ornith` | Python 3.12 · ROCm 7.14 · pinned vLLM 0.28.0 build |
| [GPT-OSS-20B](models/GPT-OSS-20B/README.md#native-serving) | `paiton serve gpt-oss-20b` | **8020** / `gpt-oss-20b` | Python 3.14 · ROCm 7.14 · pinned vLLM 0.26.1 build |
| [Qwen3-Coder](models/Qwen3-Coder-30B/README.md#native-serving) | `paiton serve qwen3-coder` | **8010** / `qwen3-coder` | Python 3.12 · ROCm 7.14 · pinned vLLM 0.28.0 build |

These runtime builds are separate existing environments. Installing Paiton
does not switch between them.

</details>

> [!NOTE]
> Native presets and published benchmark profiles are not interchangeable.
> In particular, the native NVFP4 preset does not use DFlash2. Published
> DFlash/DFlash2 throughput results describe their separate benchmark profiles.

<details>
<summary><strong>Preset behavior, quantization and supported inputs</strong></summary>

| Preset | Selected behavior |
| --- | --- |
| Qwen3.8 NVFP4 | Text-only · 65K · FP8 KV · no speculation. Explicit lossy NVFP4 → MXFP4 conversion in GPU memory; original weights remain unchanged. |
| Qwen3.8 Qronos | Uses the release's W4 LM head, with lossy quantization during loading. |
| Ornith | Text-only · 8K · non-speculative. Uses a cached lossless reshard. |
| NEO | Supports text plus one image. |
| MiniCPM5 | Explicitly disables thinking by default. |

Existing DFlash/DFlash2 benchmark launchers remain available. Follow each model's
benchmark profile rather than assuming its results apply to the native preset.

</details>

### Existing commands remain supported

`paiton vllm serve MODEL` remains an equivalent command form. Explicit profiles
also support local paths and repository IDs. Put Paiton options **before
`serve` or `vllm`**, and vLLM arguments after the model:

```bash
paiton serve minicpm5 --port 9000 --served-model-name local-mini
paiton --profile minicpm5-awq-text-8k vllm serve /models/existing-minicpm5 --port 9000
```

Installing Paiton does not activate it for ordinary `vllm serve MODEL` commands.
Paiton commands fail with an explanation when the runtime or payload does not match.

Existing model-specific shell/Python launchers, console commands and pinned
container images remain available. The paths, flags, ports and cache variables
used in existing blogs and Reddit instructions are retained.

## Managed containers

These model launchers supply their own inference environment. They are a
separate deployment path from the native CLI.

Clone the repository as shown in [Quick start](#quick-start),
then choose one launcher below. Run commands from the repository root unless
noted otherwise.

**Already downloaded the weights?** Set the model-specific paths in
[Model weights and existing downloads](#model-weights-and-existing-downloads)
before running these commands. Host cache variables are not automatically
forwarded into every container.

<details>
<summary><strong>Chat, coding and visual-chat launchers</strong></summary>

**MiniCPM5** · API port **8036** · model `minicpm5-2b`

```bash
./models/MiniCPM5-2B/serve-docker.sh
```

Once ready, run `python3 models/MiniCPM5-2B/chat.py` from another terminal.

**Qronos Qwen3.8** · API port **8000** · model `qwen38`

```bash
./models/Qwen3.8/serve-docker.sh
```

Once ready, run `docker exec -it paiton-qwen38 paiton-chat` from another terminal.

**Qwen3.8 MXFP4 + DFlash2** · API port **18982** · model `Qwen3.8`

First [choose a new download, existing local folders, or your Hugging Face cache](models/Qwen3.8-MXFP4-DFlash2/README.md#model-weights-and-existing-downloads)
and set the target, draft and runtime-cache paths. This launcher requires
prepared weights; creating empty directories is not sufficient. Then launch:

```bash
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10-65k.sh
```

Use [`run-rocm10-200k.sh`](models/Qwen3.8-MXFP4-DFlash2/run-rocm10-200k.sh) for the 200K profile.

**NEO CODER MAX** · API port **8000** · model `qwen38-neo`

```bash
./models/Qwen3.8-NEO-CODER-MAX/serve-docker.sh
```

See the [chat and image examples](models/Qwen3.8-NEO-CODER-MAX/README.md#launch).

**Ornith 1.5** · API port **8000** · model `ornith`

```bash
./models/Ornith-1.5/serve-docker.sh --chat
```

Opens terminal chat after startup.

**GPT-OSS-20B** · API port **8020** · model `gpt-oss-20b`

```bash
./models/GPT-OSS-20B/serve-docker.sh
```

Once ready, run `python3 models/GPT-OSS-20B/chat.py` from another terminal.

**Qwen3-Coder** · API port **8010** · model `qwen3-coder`

```bash
./models/Qwen3-Coder-30B/serve-docker.sh --chat
```

Provides terminal chat and a coding API.

</details>

<details>
<summary><strong>Image and video launchers</strong></summary>

The ComfyUI entries also require Docker Compose.

| Model | Launch command | Open when ready |
| --- | --- | --- |
| [FLUX.2 klein](models/FLUX.2-klein/README.md) | `./models/FLUX.2-klein/launch.sh` | [ComfyUI · 8188](http://127.0.0.1:8188/?paiton=1) |
| [Qwen-Image-2.1 MXFP4](models/Qwen-Image-2.1/README.md) | `./models/Qwen-Image-2.1/serve-docker.sh` | Image API · 8191 |
| [FastWan](models/FastWan/README.md) | `./models/FastWan/launch.sh` | [ComfyUI · 8192](http://127.0.0.1:8192/?paiton=1&preset=fast) |
| [Wan2.2](models/Wan2.2/README.md) | `./models/Wan2.2/launch.sh` | [ComfyUI · 8192](http://127.0.0.1:8192/?paiton=1&preset=base) |
| [MiniMax H3](models/MiniMax-H3/README.md) | `./models/MiniMax-H3/launch.sh` | [ComfyUI · 8190](http://127.0.0.1:8190/?paiton=1&studio=1) |

Included workflows expose the supported prompts, inputs and generation settings.
Each model guide covers outputs, cache locations and server access.

</details>

<details>
<summary><strong>Meeting-recording launcher</strong></summary>

Follow the [meeting setup guide](models/Meeting/README.md) to prepare the models
and pull the published image. Then, from `models/Meeting`:

```bash
./run-docker.sh --paiton /path/to/meeting.mp4 /path/to/new-result
```

Community-1 requires your own approved Hugging Face access for the initial model
download. Inference runs offline; the original recording is mounted read-only.

</details>

### Requirements and first launch

**Host setup.** Use Linux with Docker and AMD GPU device access. ComfyUI launchers
also need Docker Compose. Run one model at a time on the tested single-GPU setup.

**First launch.** Weights are downloaded, and runtime components may be built or
compiled. Later launches reuse persistent caches. Wait for the readiness message
before sending requests. Host RAM, disk space and preparation times vary
substantially; check the model guide before downloading.

**Memory planning.** Reported GPU use includes runtime overhead at the tested
settings. It is not a minimum-capacity guarantee. Smaller GPUs have not been
qualified; context length, concurrency and image/video resolution can change
memory requirements.

## Reading the results

Compare Paiton and the baseline within the same model and workload. Reports
document the baseline, sampling settings, repetitions, quality checks and known
limitations. The library is not a ranking across models.

<details>
<summary><strong>Throughput, latency and the baselines behind the numbers</strong></summary>

**Output throughput**

`Improvement = (Paiton tok/s ÷ baseline tok/s − 1) × 100`

MiniCPM5's highlighted comparison is **127.6 → 197.0 tok/s**. Qwen3-Coder's C2
comparison is **101.61 → 172.82 tok/s**. These are aggregate output rates, not
individual-stream decode rates.

**Latency reduction**

`Reduction = (baseline time − Paiton time) ÷ baseline time × 100`

A 50% latency reduction means twice the rate for equivalent fixed work,
not a 50% throughput increase.

**Baseline matters.** NEO is compared with llama.cpp, not stock vLLM. Its longer
128-output text workloads show 5.1% and 0.8% lower request latency; some
prefill-only cases favor llama.cpp. GPT-OSS uses the fastest qualified
4.819-second stock reference because stock timing varied between runs.

**Optimization scope matters.** Ornith's highlighted result includes DFlash
speculative decoding. Quantization, activation arithmetic and quality differences
are documented per model. MiniCPM5's repeated timings show unresolved variability;
meeting results also have substantial variation.

</details>

## About Paiton

**Public integrations. Compiled runtime artifacts. A proprietary compiler.**

This repository distributes Paiton's public integrations and compiled runtime
artifacts. The Paiton compiler remains proprietary. Model weights are fetched
from their pinned publishers and retain their own licenses.

The vLLM plugin is [Apache-2.0 licensed](LICENSE). Bundled components retain their
applicable licenses, including ComfyUI and separate image tools. See the
[root notices](THIRD_PARTY_NOTICES.md) and each model's notices.

Paiton's broader work also includes AMD Instinct accelerators and multi-GPU
inference. [Explore Paiton or discuss your workload →](https://eliovp.com/products/paiton)

<p align="center">
  <strong>Build locally. Get more from your GPU.</strong><br><br>
  <a href="https://github.com/Eliovp-BV/paiton-vllm-plugin/releases">Release downloads</a> ·
  <a href="https://github.com/users/Eliovp/packages/container/package/paiton-vllm-plugin">Containers</a> ·
  <a href="https://huggingface.co/EliovpAI">Hugging Face</a> ·
  <a href="https://github.com/Eliovp-BV/paiton-studio">Paiton Studio</a> ·
  <a href="https://github.com/Eliovp-BV/paiton-vllm-plugin/issues">Report an issue</a>
</p>
