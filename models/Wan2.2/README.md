# Wan2.2 TI2V-5B on Radeon AI PRO R9700

## Model weights and existing downloads

Run these examples from the repository root. The base denoiser, text encoder and VAE are pinned in [the checkpoint lock](checkpoints.lock.json).
A denoiser alone is not the complete pipeline. Wan2.2 and FastWan share the same
container and persistent data layout.

### First download

```bash
./models/Wan2.2/launch.sh
```

The launcher builds the local runtime image, downloads this preset and starts
ComfyUI. Its cache is under `PAITON_WAN_DATA/cache/hub`, not automatically your
host's default Hub cache.

### Already in a local folder

An existing prepared data directory can be selected with
`export PAITON_WAN_DATA="/absolute/path/to/wan-data"` before launching. It must
retain both `models/` and `cache/hub/`, so the preparation check can reuse files.

For models stored elsewhere, bind the complete ComfyUI-style model folder
read-only and use the Docker CLI below. It needs:

- `diffusion_models/wan2.2_ti2v_5B_fp16.safetensors`.
- `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`.
- `vae/wan2.2_vae.safetensors`.


Prepare the two local images once if they are not already built. These commands
build the runtime and do not download checkpoint weights:

```bash
docker build -t paiton-wan22-artifacts:local-v1 \
  -f models/Wan2.2/Dockerfile models/Wan2.2
docker build --build-arg PAITON_WAN_BASE_IMAGE=paiton-wan22-artifacts:local-v1 \
  -t paiton-wan22:local-v1 -f models/Wan2.2/Dockerfile.local models/Wan2.2
```

Then generate from your existing files:

```bash
export PAITON_MODEL_DIR="/absolute/path/to/comfy-models"
export PAITON_WAN_DATA="$HOME/.local/share/paiton/wan22-existing"
export PAITON_WAN_OUTPUTS="$HOME/paiton-videos"
mkdir -p "$PAITON_WAN_DATA" "$PAITON_WAN_OUTPUTS"
docker run --rm --init --device /dev/kfd --device /dev/dri \
  --group-add "$(stat -c '%g' /dev/kfd)" --user "$(id -u):$(id -g)" --shm-size 2g \
  --mount "type=bind,src=$PAITON_WAN_DATA,dst=/data" \
  --mount "type=bind,src=$PAITON_MODEL_DIR,dst=/data/models,readonly" \
  --mount "type=bind,src=$PAITON_WAN_OUTPUTS,dst=/outputs" \
  -e HF_HUB_OFFLINE=1 paiton-wan22:local-v1 \
  generate --preset base --engine stock --resolution 480 --duration 2 \
  --prompt 'A red fox walks through a sunlit woodland clearing.' --seed 1201
```

Use standalone files here. A model folder containing links outside the mounted
folder needs the linked storage mounted too; the following recipe does that
for Hub downloads.

### Already in the Hugging Face cache

After building the images above, mount the entire Hub cache for preparation
and generation. The download command runs in offline mode and reuses the exact
files in the checkpoint lock. It still creates model links and, for FastWan,
the converted denoiser in the writable data directory:

```bash
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
export PAITON_WAN_DATA="$HOME/.local/share/paiton/wan22-existing"
export PAITON_WAN_OUTPUTS="$HOME/paiton-videos"
mkdir -p "$PAITON_WAN_DATA" "$PAITON_WAN_OUTPUTS"
wan_cached=(docker run --rm --init --device /dev/kfd --device /dev/dri
  --group-add "$(stat -c '%g' /dev/kfd)" --user "$(id -u):$(id -g)" --shm-size 2g
  --mount "type=bind,src=$PAITON_WAN_DATA,dst=/data"
  --mount "type=bind,src=$HF_HUB_CACHE,dst=/data/cache/hub,readonly"
  --mount "type=bind,src=$PAITON_WAN_OUTPUTS,dst=/outputs"
  -e HF_HUB_OFFLINE=1 paiton-wan22:local-v1)
"${wan_cached[@]}" download --preset base --offline
"${wan_cached[@]}" generate --preset base --engine stock --resolution 480 --duration 2 \
  --prompt 'A red fox walks through a sunlit woodland clearing.' --seed 1201
```

The Bash array keeps the same mounts for both commands. Only run generation
when preparation succeeds. Set `HF_HUB_CACHE` to your cache drive's absolute
path if needed. Keep this cache mount on later runs because the prepared links
refer to it. These direct commands generate a clip; they do not start ComfyUI
or change the ordinary launcher's mounts. [Cache path guide](../../docs/MODEL_WEIGHTS.md).

## Serving interface

For this video generation package, keep using the existing launcher from the repository
root: `./models/Wan2.2/launch.sh`. Open [the included ComfyUI workflow](http://127.0.0.1:8192/?paiton=1&preset=base) after startup.
The setup, CLI alternatives, flags and cache locations documented below remain
available. `paiton serve` provides vLLM chat APIs, including NEO visual chat;
this package uses its existing generation workflow. See the
[main launch guide](../../README.md#quick-start) to choose an interface.


Local text-to-video and image-to-video with separate **stock** and **Paiton** choices. FastWan FullAttn 5B provides a three-evaluation text preview; original Wan2.2 TI2V-5B provides optional image input. Both produce silent video. [Hugging Face runtime package](https://huggingface.co/EliovpAI/Wan2.2-FastWan-5B-Paiton-RDNA4) · [FastWan text guide](../FastWan/README.md).

FastWan Paiton reduced complete-clip latency by **2.8–4.7%** in the fixed R9700 set. The base image cases did not improve end to end, so they default to stock. [See the measurements](BENCHMARKS.md) and [retained clips](QUALITY.md).

## Launch ComfyUI

Start the base text/image model:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git && cd paiton-vllm-plugin && ./models/Wan2.2/launch.sh
```

Open [FastWan text workflow](http://127.0.0.1:8192/?paiton=1&preset=fast) or [Wan image/text workflow](http://127.0.0.1:8192/?paiton=1&preset=base). The connected workflow opens on first visit. Choose `stock` or `paiton` in its loader, edit the prompt, set duration/resolution and click **Run**. The base workflow includes image upload; choose `(none)` for text-only generation. Use portrait orientation for the supplied portrait example. Native Wan image conditioning resizes and center-crops to the chosen canvas; it does not preserve every input edge.

ComfyUI listens on **0.0.0.0:8192**. From another machine, replace `127.0.0.1` with the server's address. This is a separate service and port from H3. Existing models, caches and outputs are not replaced.

The first launch builds the pinned local runtime and downloads the base preset. Subsequent launches reuse model and runtime caches. For FastWan text generation, use `./models/FastWan/launch.sh`; it adds the fast preset and opens its workflow. Both launchers share one service and cache. Set `PAITON_WAN_DOWNLOAD_PRESET=all` to download both. Workflow files are also in [comfyui/workflows](comfyui/workflows), with matching API examples.

```bash
./models/Wan2.2/launch.sh --logs
./models/Wan2.2/launch.sh --stop
```

![Connected FastWan workflow](assets/comfy-fast-workflow.png)

[Image-upload workflow screenshot](assets/comfy-base-upload.png)

## Settings

| Preset | Input | Resolution controls | Duration controls | Sampling |
| --- | --- | --- | --- | --- |
| `fast` | Text | 832×480 or 1280×704, landscape | 49 or 121 frames | Three DMD evaluations, guidance 1 |
| `base` | Text and optional image | 832×480 landscape or 480×832 portrait | 49 or 121 frames | 20 UniPC steps, simple schedule, shift 8, guidance 5 |

All outputs use 24 fps. The UI's “2 seconds” and “5 seconds” labels correspond to 49/121 frames: encoded duration is **2.0417/5.0417 seconds**, with first-to-last-frame timestamps spanning 2/5 seconds. Wan uses a `4n+1` frame grid. “720 class” here means **1280×704**, not 1280×720. Longer clips are native continuous generations, without stitching or interpolation. FastWan image input is not qualified; use `base`. Five-second base evidence here is image-conditioned; the base text benchmark uses 49 frames.

The text default is a short 720-class clip. Select 480 class for a faster preview, or five seconds for longer motion with higher generation time. The base image workflow defaults to 480 class and stock; FastWan defaults to Paiton. The short base image benchmark did not show a complete-clip Paiton gain. These choices do not inherit H3's duration limits or audio capabilities.

## Terminal generation and fair benchmarks

```bash
./models/Wan2.2/run.sh generate --preset fast --engine paiton \
  --resolution 720 --duration 5 --seed 1201 \
  --prompt 'A red fox walks through a sunlit woodland clearing and turns toward the camera.'

./models/Wan2.2/run.sh benchmark --preset fast --engine stock \
  --resolution 480 --duration 2 --runs 6 --warmups 2
./models/Wan2.2/run.sh benchmark --preset fast --engine paiton \
  --resolution 480 --duration 2 --runs 6 --warmups 2
```

Put an input image under the data directory's `input/` folder, then use its container path:

```bash
./models/Wan2.2/run.sh generate --preset base --engine stock \
  --portrait --duration 2 --image /data/input/example.jpg \
  --prompt 'The subject slowly turns toward the camera. Preserve the subject and composition.'
```

Outputs get unique directories under `~/paiton-videos`. Explicit `--output /outputs/my-run` must name a new or empty directory. Each benchmark retains commands/settings, clips, latents, component timings, actual denoiser calls and sampled telemetry. Warmups are retained and marked, not mixed into warm statistics. Fresh text conditioning runs every time, including repeated UI requests; loaded components may remain cached.

The terminal benchmark includes H.264 encoding (CRF 18, preset `fast`). The UI uses ComfyUI's native SaveVideo node at CRF 18; UI encoding behavior is not used as the terminal benchmark baseline. Run one generation process at a time.

## Requirements and persistent storage

- Linux, Docker with Compose, working AMD ROCm device access (`/dev/kfd`, `/dev/dri`), one **gfx1201 R9700 with 32 GB VRAM**. This artifact is hardware-specific.
- Qualification host: approximately 15 GiB usable RAM and 4 GiB swap. Leave headroom for the OS and loading; 32 GB system RAM is a more comfortable installation. Detailed measured peaks belong in [BENCHMARKS.md](BENCHMARKS.md).
- Model storage: about 18.15 GB for base, or 28.15 GB for fast including the retained 10 GB original and its lossless converted copy. Both presets use about **38.15 GB** together. Container layers, build caches, compiler caches and retained videos need additional space; allow **100 GB free disk** for a fresh build and both presets.
- First launch includes network downloads and container assembly. First generation includes model loading and compilation; it is substantially slower than warm generation. Download speed and build-cache state determine installation time, so warm clip time is not a first-run estimate.

Defaults:

| Variable | Default |
| --- | --- |
| `PAITON_WAN_DATA` | `${XDG_DATA_HOME:-$HOME/.local/share}/paiton/wan22` |
| `PAITON_WAN_OUTPUTS` | `$HOME/paiton-videos` |
| `PAITON_WAN_PORT` | `8192` |

Model originals live in `cache/hub`; model links stay relative to the data directory. Converted FastWan weights include a pinned provenance manifest. Container compiler caches use `cache/inductor-container`, separate from host execution caches because compiler cache entries can contain absolute paths. No model conversion overwrites originals.

## What Paiton changes

Paiton compiles and fuses Wan VAE channel normalization with the following SiLU for the measured BF16 shapes. Stock convolutions, denoiser, attention, conditioning, sampler and RNG behavior remain shared. Both engines use the qualified CK attention backend, native scaled-FP8 text encoder, BF16 denoiser/VAE, MIOpen and block-level `torch.compile`. Full-pipeline graph capture is disabled after an actual capture failure in the dynamic loader.

The VAE optimization produces small pixel-rounding differences; it does not change denoised latents in the paired checks. Its decode improvement is larger than its complete-clip improvement because sampling and file encoding remain substantial. See the actual measurements and examples rather than assuming a fixed percentage.

- [Complete benchmarks](BENCHMARKS.md)
- [Quality evidence and comparison clips](QUALITY.md)
- [Candidate selection and quantization assessment](CANDIDATES.md)
- [Pinned model/source revisions](checkpoints.lock.json)
- [Licenses and redistribution](THIRD_PARTY_NOTICES.md)

## Container and release layout

`Dockerfile` prepares a separate artifact image using the existing pinned community runtime. `Dockerfile.local` assembles pinned ComfyUI core/frontend locally, following the existing community packaging convention. The launch script builds local tags `paiton-wan22-artifacts:local-v1` and `paiton-wan22:local-v1`; it does not require an unpublished Wan image. Use `launch.sh --build` after modifying package code.

A proposed publication would need a **new Wan artifact image**, not an H3 retag. The combined ComfyUI image is local assembly only. No model weights or private compiler source are distributed in this directory. The [Hugging Face runtime package](https://huggingface.co/EliovpAI/Wan2.2-FastWan-5B-Paiton-RDNA4) provides both presets, compiled artifacts and retained evidence. The Wan GHCR image remains a proposed publication; launchers build locally from the pinned public base image.


## Measured warm clips

| Setting | Stock clip s | Paiton clip s | Paiton change |
| --- | ---: | ---: | ---: |
| FastWan 1280×704, 49f, person | 20.376 | 19.797 | +2.84% |
| FastWan 832×480, 49f, fox | 9.280 | 8.843 | +4.71% |
| FastWan 1280×704, 121f, fox | 51.986 | 50.109 | +3.61% |
| Base 480×832, 49f, image | 29.518 | 29.758 | -0.81% |

Complete times include fresh conditioning and H.264 file encoding. Each mean uses two measured runs after two warmups on one R9700. Positive change means faster; the base image case did not improve. See [all cases, ranges, memory and component times](BENCHMARKS.md).
