# FastWan FullAttn 5B on Radeon AI PRO R9700

## Model weights and existing downloads

Run these examples from the repository root. The FastWan denoiser and shared Wan text encoder/VAE are pinned in [the shared checkpoint lock](../Wan2.2/checkpoints.lock.json).
A denoiser alone is not the complete pipeline. Wan2.2 and FastWan share the same
container and persistent data layout.

### First download

```bash
./models/FastWan/launch.sh
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

- `diffusion_models/fastwan22_5b_fullattn_comfy_bf16.safetensors` and its `.conversion.json` manifest.
- `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`.
- `vae/wan2.2_vae.safetensors`.

The original FastWan Diffusers denoiser requires the package's lossless
conversion before generation. If you have that original download, use
**Already in the Hugging Face cache** below to perform preparation.

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
  generate --preset fast --engine paiton --resolution 480 --duration 2 \
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
"${wan_cached[@]}" download --preset fast --offline
"${wan_cached[@]}" generate --preset fast --engine paiton --resolution 480 --duration 2 \
  --prompt 'A red fox walks through a sunlit woodland clearing.' --seed 1201
```

The Bash array keeps the same mounts for both commands. Only run generation
when preparation succeeds. Set `HF_HUB_CACHE` to your cache drive's absolute
path if needed. Keep this cache mount on later runs because the prepared links
refer to it. These direct commands generate a clip; they do not start ComfyUI
or change the ordinary launcher's mounts. [Cache path guide](../../docs/MODEL_WEIGHTS.md).

## Serving interface

For this video generation package, keep using the existing launcher from the repository
root: `./models/FastWan/launch.sh`. Open [the included ComfyUI workflow](http://127.0.0.1:8192/?paiton=1&preset=fast) after startup.
The setup, CLI alternatives, flags and cache locations documented below remain
available. `paiton serve` provides vLLM chat APIs, including NEO visual chat;
this package uses its existing generation workflow. See the
[main launch guide](../../README.md#quick-start) to choose an interface.


Generate silent text-to-video locally in ComfyUI or the terminal. This three-evaluation distilled Wan2.2 setting defaults to Paiton, with a separate stock option. Image input is available through [Wan2.2 TI2V-5B](../Wan2.2/README.md).

[![Five-second FastWan example](../Wan2.2/assets/fast-720-5-paiton.jpg)](https://huggingface.co/EliovpAI/Wan2.2-FastWan-5B-Paiton-RDNA4/resolve/main/models/Wan2.2/assets/fast-720-5-paiton.mp4)

[Play the example](https://huggingface.co/EliovpAI/Wan2.2-FastWan-5B-Paiton-RDNA4/resolve/main/models/Wan2.2/assets/fast-720-5-paiton.mp4) · [Hugging Face runtime package](https://huggingface.co/EliovpAI/Wan2.2-FastWan-5B-Paiton-RDNA4)

## Quick start

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git && cd paiton-vllm-plugin && ./models/FastWan/launch.sh
```

Open [ComfyUI](http://127.0.0.1:8192/?paiton=1&preset=fast), edit the prompt and click **Run**. Choose stock/Paiton, 480/720 class and two/five seconds in the connected workflow. The default is Paiton, 720 class, two seconds. ComfyUI listens on **0.0.0.0:8192**; replace localhost with your server address for remote access.

You need Linux, Docker with Compose, AMD GPU device access and one **32 GB gfx1201 R9700**. The first launch builds the pinned runtime and downloads this preset; later launches reuse model and runtime caches. Model storage is about **28.15 GB**, including the preserved original and lossless converted checkpoint. Allow 100 GB free disk for containers, caches and outputs. Qualification used about 15 GiB usable host RAM and 4 GiB swap; 32 GB RAM gives more headroom.

Both Wan launchers share one service and cache. Run one generation at a time. Launch the [base model](../Wan2.2/README.md) to add image input. Outputs go to `~/paiton-videos`; cache and port controls are in the [shared runtime guide](../Wan2.2/README.md#requirements-and-persistent-storage).

```bash
./models/FastWan/launch.sh --logs
./models/FastWan/launch.sh --stop
./models/Wan2.2/run.sh generate --preset fast --engine paiton --resolution 480 --duration 2 --seed 1201 --prompt 'A red fox walks through a sunlit woodland clearing and turns toward the camera.'
```

## Measured performance

| Setting | Stock clip s | Paiton clip s | Lower latency |
| --- | ---: | ---: | ---: |
| 832×480, 49 frames | 9.280 | 8.843 | 4.71% |
| 1280×704, 49 frames, person | 20.376 | 19.797 | 2.84% |
| 1280×704, 121 frames | 51.986 | 50.109 | 3.61% |

Complete times include fresh conditioning and file encoding; each mean uses two measured runs after two warmups. Both engines use the same three DMD evaluations, adapters, schedule and runtime settings. Distillation gains are not compiler gains. Paiton fuses VAE normalization and SiLU; paired denoised latents were bitwise equal, with small decoded pixel differences. See [full benchmarks](../Wan2.2/BENCHMARKS.md), [quality limitations and paired clips](../Wan2.2/QUALITY.md) and [reproduction commands](../Wan2.2/README.md#terminal-generation-and-fair-benchmarks).

At 24 fps, 49/121 frames encode as 2.0417/5.0417 seconds. “720 class” means 1280×704. Clips are native continuous generation. This setting is qualified for text input only and generates no audio.

## Models and licenses

The pinned Apache-2.0 [FastWan FullAttn checkpoint](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers/tree/3e187042a324f6f5fb68fd22110a78725253de8f) is downloaded from its original publisher. The shared [checkpoint lock](../Wan2.2/checkpoints.lock.json), [license notices](../Wan2.2/THIRD_PARTY_NOTICES.md) and [runtime recipes](../Wan2.2/README.md#container-and-release-layout) cover reproducibility and redistribution. The Hugging Face package contains runtime code, compiled artifacts and evidence, not model weights or private compiler source.
