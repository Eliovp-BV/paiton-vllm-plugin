# Qwen-Image-2.1 on one R9700

Qwen-Image-2.1 makes images and transparent RGBA artwork from a text prompt, and edits an existing image, on one
Radeon AI PRO R9700 (32 GB). The container serves a JSON image API on port 8191.

With the 8-step Turbo model a 1024 × 1024 image takes **3.99 s** and a 2048 × 2048 image 22.05 s. Stock diffusers
needs 23.6 s for the same Turbo model at 1024 × 1024, with CPU offload because the bf16 model does not fit the card,
and runs out of memory at 2048 × 2048.

Start it from a checkout of this repository:

```sh
./models/Qwen-Image-2.1/serve-docker.sh --model turbo
```

Without `--model`, the launcher starts the original 40-step model, which is slower (19.7 s at 1024 × 1024, measured
on v1.0.3).

[Models](#models) · [Requirements](#requirements) · [Start](#start-with-one-command) ·
[Generate an image](#generate-an-image) · [API](#api) · [Precision profiles](#precision-profiles) ·
[Results](#results) · [Prompt guard](#minors-prompt-guard) · [Offline use](#download-offline-use-and-fallback)

<a id="qwen-image-21-turbo-8-steps-fp8--new-in-v110"></a>

## Models

One container holds three models. Pick one at launch with `--model`; only one is loaded at a time.

| `--model` | Model | Steps | First download | 1024 × 1024 | 2048 × 2048 |
| --- | --- | ---: | ---: | ---: | ---: |
| `turbo` | [Qwen-Image-2.1-Turbo, FP8 weights](https://huggingface.co/EliovpAI/Qwen-Image-2.1-Turbo-FP8-NearBF16) | 8 | 18.54 GB | 3.99 s | 22.05 s |
| `original` (default) | [Qwen-Image-2.1, MXFP4 weights](https://huggingface.co/EliovpAI/Qwen_Image-2.1-MXFP4-Paiton-RDNA4) | 40 | 9.33 GB | 19.7 s | 104.7 s |
| `uncensored` | [the abenzerps fine-tune converted to MXFP4 by Paiton](https://huggingface.co/EliovpAI/Qwen_Image-2.1-Uncensored-MXFP4-Paiton) | | | not measured on the R9700 | |

Times are warm complete requests on one R9700, PNG included. The `original` row was measured on v1.0.3.

```sh
./models/Qwen-Image-2.1/serve-docker.sh --model original
./models/Qwen-Image-2.1/serve-docker.sh --model uncensored
```

Restart the worker to switch models. Downloads are independently pinned and verified, and caches are separated by
HF repository and revision.

The uncensored conversion passed MI355 correctness, image-quality screening, RGBA, editing and A-B-A checks. It has
**not** been run on R9700 in this release's qualification. The R9700 timings and VRAM figures on this page describe
the original and Turbo models. The source author's "uncensored" label is not a guarantee of behavior. See
[the conversion measurements](measurements/uncensored-mi355.json).

For compatible framework runtimes beyond RDNA4, see the separate
[portable MXFP4 checkpoint](https://huggingface.co/EliovpAI/Qwen_Image-2.1-MXFP4). This container uses the RDNA4
package; portable setup and hardware measurements are documented in that model's card.

## Requirements

| Need | Detail |
| --- | --- |
| GPU | One Radeon AI PRO R9700, 32 GB. The worker requires at least 30 GiB free before loading. Run one model at a time. Smaller GPUs and other architectures are not qualified. |
| Host | Linux, Docker and a working AMD GPU driver. Qualification used a host with approximately 16 GiB RAM. |
| Disk | The container is about 10.4 GB before registry compression. Add the checkpoint (9.33 GB or 18.54 GB), runtime caches and your images. |
| Peak GPU memory | Turbo: 19.5 GiB at 1024 × 1024 and 26.1 GiB at 2048 × 2048. Original: 28.34 GiB in the latest container repeat. |

## Start with one command

```sh
./models/Qwen-Image-2.1/serve-docker.sh
```

This starts the original model. The first launch downloads **9.33 GB** of checkpoint files, verifies their SHA-256
hashes and loads the model. Wait for `READY http://0.0.0.0:8191`. The named volume keeps weights and runtime caches
for later launches. The container holds the inference environment and the compiled runtime libraries; the weights are
downloaded at launch.

For the Turbo model, add `--model turbo`. Its first launch downloads **18.54 GB** (pinned revision, SHA-256
verified).

The server listens on all interfaces (0.0.0.0) and the image API has no authentication. Use `PAITON_BIND=127.0.0.1`
(or `serve --host 127.0.0.1` without Docker) to keep it local, and use a firewall if the machine is reachable from
untrusted networks.

| Variable | Sets |
| --- | --- |
| `PAITON_PORT` | host port |
| `PAITON_BIND` | host address |
| `PAITON_CONTAINER` | container name |
| `PAITON_CACHE` | cache volume or path; use an absolute path for a host directory |
| `PAITON_IMAGE` | container image |

Without a repository checkout, the same container starts with:

```sh
docker run --rm --name paiton-qwen-image21 --device /dev/kfd --device /dev/dri --ipc=host -p 0.0.0.0:8191:8191 -v paiton-qwen-image21-cache:/cache ghcr.io/eliovp/paiton-vllm-plugin:qwen-image21-mxfp4-rdna4-v1.1.0
```

and for the Turbo model:

```sh
docker run --rm --name paiton-qwen-image21 --device /dev/kfd --device /dev/dri --ipc=host -p 0.0.0.0:8191:8191 -v paiton-qwen-image21-cache:/cache ghcr.io/eliovp/paiton-vllm-plugin:qwen-image21-mxfp4-rdna4-v1.1.0 --model turbo
```

## Generate an image

With the server running, use this standard-library client from another terminal:

```sh
python3 models/Qwen-Image-2.1/request.py --prompt 'A neon-lit street in Tokyo at night, rain reflections, cinematic photography' --output outputs/tokyo.png
```

For transparency, add `--mode rgba`. To edit a local image:

```sh
python3 models/Qwen-Image-2.1/request.py --mode edit --size 1024 --image outputs/tokyo.png --prompt 'Change the scene to a snowy winter evening' --output outputs/winter.png
```

To generate one image without keeping an API server running, stop the server container first and use:

```sh
./models/Qwen-Image-2.1/generate-docker.sh --prompt 'A ceramic teapot on a wooden table, soft daylight' --output /outputs/teapot.png
```

This saves the PNG and a timing JSON in `./outputs` on the host. Set `PAITON_OUTPUT_DIR` to choose another directory.
Existing output files are never overwritten. For direct edits, place the input in that directory and use
`--mode edit --size 1024 --image /outputs/input.png`.

## API

`GET /health` reports readiness and the active native fusions
(`native_fusions: active (bf16-regions, attention, normfuse, …)`; the default profile also lists its low-precision
kernels). `GET /v1/models` lists the supported tasks. Generation returns a PNG in `data[0].b64_json`:

```sh
curl http://127.0.0.1:8191/v1/images/generations -H 'Content-Type: application/json' -d '{"model":"paiton-image-2.1","prompt":"A glass dragon sticker","mode":"rgba","size":"2048x2048","seed":42,"n":1}' > response.json
```

The image endpoints use JSON. `/v1/images/edits` accepts the same fields with `size: "1024x1024"` and a base64 input
PNG in `image_b64`. It does not accept multipart uploads or remote image URLs. The included client handles encoding
and saving. This image engine uses Diffusers and its own API, separate from the repository's `paiton serve` vLLM
language-model presets.

Omit `model` to use the loaded checkpoint. For the uncensored selection the API model ID is
`paiton-image-2.1-uncensored`. `/health`, `/v1/models` and generation metrics identify the loaded model. The included
`request.py` client follows the server's loaded model unless `--model` is given.

What is qualified:

| Setting | Qualified scope |
| --- | --- |
| Text-to-image and RGBA | 1024 × 1024 or 2048 × 2048 |
| Editing | One input image, up to 4,194,304 pixels; 1024 × 1024 output |
| Sampling, original model | 40 steps, guidance 1.0, batch one |
| Sampling, Turbo model | 8 steps, guidance 1.0; a request that sets `steps` must use 8 |
| Requests | One active request; prompts up to 512 characters |
| Memory | Components resident; no CPU model offload or VAE tiling |
| Precision, original model | Balanced MXFP4 weight storage; BF16 activation arithmetic and intentionally BF16 components |
| Graphs | Whole-pipeline capture unsupported |

## Precision profiles

A precision profile sets how much of the arithmetic runs at lower precision. Each model has a default; add
`serve --precision-profile <name>` to the launcher to choose another, for example
`./models/Qwen-Image-2.1/serve-docker.sh serve --precision-profile exact`, or put it after the image name in the
`docker run` form. The checkpoint bytes are the same in every profile.

Turbo model. All profiles use the checkpoint's FP8 weights, with bf16 activations on steps 1-2 and FP8 activations
with power-of-two per-token scales on steps 3-8, as in the checkpoint's `paiton_quant.json`:

| Profile | Attention | Quality vs bf16 (73 images: LPIPS / DINOv2 / alpha IoU) |
|---|---|---:|
| `turbo-fp8-fast-rgba-exact` (default) | int8-QK / fp8-PV on the last four steps; RGBA requests keep exact attention | 0.0921 / 0.9839 / 0.942 |
| `turbo-fp8-fast` | int8-QK / fp8-PV on the last four steps, RGBA included | 0.0924 / 0.9839 / 0.939 |
| `turbo-fp8` | exact on every step | 0.0918 / 0.9841 / 0.942 |

Turbo profiles require `--model turbo` and the native fusions (`--no-native-fusions` is refused for this model).

Original model:

| Profile | What it does |
| --- | --- |
| `schedule-int8` (default) | Keeps the cached text prefix and the first seven denoising steps exact. Steps 8-40 run with int8 activations and int8/fp8 attention through native kernels. |
| `schedule-int8-11` | Keeps ten steps exact with int8-QK attention: more quality headroom, measured 112.7 s. |
| `exact` | Keeps every step bit-exact against the pinned BF16 arithmetic, with the exact native attention. |

The receipts record the profile, and the [measurements](measurements/low-precision-r9700.json) hold the matched
samples and quality grades. The `uncensored` model defaults to `exact`, which preserves BF16 activation arithmetic;
the low-precision profiles have not been graded on that fine-tune. An explicit `--precision-profile` overrides any
default.

## Results

Warm complete-request latency on one R9700 (PNG included):

| | 1024 × 1024 | 2048 × 2048 |
|---|---:|---:|
| Stock diffusers, Qwen-Image-2.1-Turbo bf16, 8 steps (model CPU offload: the bf16 model does not fit the 32 GB card) | 23.6 s | out of memory |
| Paiton v1.0.3, Qwen-Image-2.1 MXFP4, 40 steps (`schedule-int8`) | 19.7 s | 104.7 s |
| **Paiton v1.1.0, `--model turbo`** (default `turbo-fp8-fast-rgba-exact`) | **3.99 s** | **22.05 s** |
| Paiton v1.1.0, `--model turbo --precision-profile turbo-fp8` (exact attention) | 4.05 s | 23.68 s |

The v1.0.3 row is the previous release with the original 40-step model, for orientation; the Turbo model needs 8
steps. Turbo's first request after start takes 15.2 s at 1024 × 1024 (fresh cache). The first-launch download is
separate from these timings.

[Measurements, quality checks and limits](BENCHMARKS.md).

## Minors prompt guard

Every request of every model in this container (and of the FLUX.2 klein engine) passes a prompt guard before the prompt is
encoded. Prompts that combine sexual content with any indication of a minor are refused with HTTP 400
(`Prompt refused: sexual content involving minors is not allowed`); no image is generated. The engine does not start if the
guard is missing or fails its self-test. The guard is a prompt-level filter that deliberately over-blocks; it is not a
guarantee, and you remain responsible for how you use the models and for complying with the law and the model licenses.

## Download, offline use and fallback

Append `--download-only` to the container command or shell launcher to download and verify without initializing the
GPU. Append `--offline` to require the already populated cache. Network access is disabled in the model loader after
checkpoint preparation; it does not execute Python code downloaded from the Hub.

The original checkpoint is fixed at revision `b52373faeac233fb27c8bba2d334729208ee7b22`. The complete file allowlist
and hashes are in [checkpoint.lock.json](checkpoint.lock.json). To reuse a local published snapshot, mount it
read-only and pass `--model-dir /path/in/container`; it must pass the same verification. Weights are never
requantized at startup.

The uncensored selection uses its own [checkpoint.uncensored.lock.json](checkpoint.uncensored.lock.json). Passing an
original checkpoint directory with `--model uncensored`, or the reverse, fails verification. Both use the Paiton
packed format and the same native artifacts. The portable HF package has a different format label and loader and is
not a drop-in replacement for either pinned directory.

Fallback: append `--no-native-fusions` to use the existing native weight reconstruction with the original BF16
regions. The optimized regions check the pinned framework, upstream sources, artifact hashes, ABI and target before
use. Unsupported upstream contracts keep the original regions and report that state in `/health`. The managed image
pins the qualified environment and enables the optimized path by default.

## Existing environments and container build

The container is the reproducible entry point. A native launch requires CPython 3.12 and the exact pinned Torch/ROCm
and Python dependencies in `requirements-torch.txt`, `requirements-python.txt` and `requirements.txt`. Use an
isolated environment; these are image-runtime requirements, not additions to the vLLM plugin's dependencies.

```sh
PAITON_PYTHON=/path/to/qualified/python ./models/Qwen-Image-2.1/launch.sh serve --host 127.0.0.1
```

Build from the public runtime package only:

```sh
paiton_artifact_container=$(docker create ghcr.io/eliovp/paiton-vllm-plugin@sha256:c7ab0c5f900bf16c2b56fa5aa0f9dd7b8c32a0ea6106db954b43556c4cc07d97)
docker cp "$paiton_artifact_container:/app/artifacts/." models/Qwen-Image-2.1/artifacts/
docker rm "$paiton_artifact_container"
python3 models/Qwen-Image-2.1/verify_artifacts.py
docker build -t paiton-qwen-image21:local models/Qwen-Image-2.1
PAITON_IMAGE=paiton-qwen-image21:local ./models/Qwen-Image-2.1/serve-docker.sh
```

The native libraries are compiled and distributed in the published container. The commands above copy only those
existing artifacts from the pinned v1.0.2 image; they do not start a GPU process. v1.0.3 reuses the same 12
libraries, and the build verifies all their hashes. The Git checkout tracks their manifests; the proprietary compiler
and generated implementation source are not needed or included. The complete model still uses its pinned external
framework runtime.

## License and attribution

Built with Qwen. Model weights retain the upstream [Qwen Research License](LICENSE) and [NOTICE](NOTICE). The adapter
and distributed Paiton runtime binaries use the repository's Apache-2.0 runtime license. Dependency licenses remain
applicable. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Earlier releases

Results and notes for earlier versions are in the [history](HISTORY.md).
