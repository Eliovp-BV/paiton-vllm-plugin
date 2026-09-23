# Qwen-Image-2.1 MXFP4 on RDNA4

Generate images, transparent RGBA artwork, or edit an image on one **32 GB Radeon
AI PRO R9700**. The managed container enables the qualified Paiton optimizations
and downloads the balanced checkpoint from
[EliovpAI/Qwen_Image-2.1-MXFP4-Paiton-RDNA4](https://huggingface.co/EliovpAI/Qwen_Image-2.1-MXFP4-Paiton-RDNA4).

For compatible framework runtimes beyond RDNA4, see the separate
[portable MXFP4 checkpoint](https://huggingface.co/EliovpAI/Qwen_Image-2.1-MXFP4).
This container uses the RDNA4 package; portable setup and hardware measurements
are documented in that model's card.

At **2048 × 2048, 40 steps, guidance 1.0**, v1.0.2 measured **103.64 seconds** median warm
complete-request latency against **136.65 seconds** for its bit-exact profile in matched fresh-process
pairs (24.2% lower latency), and the bit-exact profile itself is 17.7% faster than v1.0.1
(165.588 → 136.285 seconds, denoiser tensors bit-exact). Three fresh container processes of the published image measured **103.29 seconds** warm with the default profile (one with `--precision-profile exact`: **133.74 seconds**).
Checkpoint bytes and settings are unchanged; the default profile's precision schedule is described below.
[Measurements, quality checks and limits](BENCHMARKS.md).

## Start with one command

Linux, Docker and a working AMD GPU driver are required. Run one model at a time
on the R9700; the worker requires at least 30 GiB free before loading. From a
repository checkout, start the image API like every other Paiton launcher:

```sh
./models/Qwen-Image-2.1/serve-docker.sh
```

The first launch downloads **9.33 GB** of checkpoint files, verifies their SHA-256
hashes, and loads the model. The named volume preserves weights and runtime caches.
Wait for `READY http://0.0.0.0:8191`. The download is separate from the reported
inference timings. The container contains the inference environment and compiled
runtime libraries; model weights are downloaded at launch.

`PAITON_PORT`, `PAITON_BIND`, `PAITON_CONTAINER`, `PAITON_CACHE` and `PAITON_IMAGE`
override the host port/address, container name, cache volume/path and image.
Use an absolute path for a host cache directory. The default API binds to localhost.

Without a repository checkout, the same container starts with:

```sh
docker run --rm --name paiton-qwen-image21 --device /dev/kfd --device /dev/dri --ipc=host -p 127.0.0.1:8191:8191 -v paiton-qwen-image21-cache:/cache ghcr.io/eliovp/paiton-vllm-plugin:qwen-image21-mxfp4-rdna4-v1.0.2
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

To generate one image directly without keeping an API server running, stop the
server container first and use:

```sh
./models/Qwen-Image-2.1/generate-docker.sh --prompt 'A ceramic teapot on a wooden table, soft daylight' --output /outputs/teapot.png
```

This saves the PNG and timing JSON in `./outputs` on the host. Set
`PAITON_OUTPUT_DIR` to choose another directory. Existing output files are never
overwritten. For direct edits, place the input in that directory and use
`--mode edit --size 1024 --image /outputs/input.png`.

## API

`GET /health` reports readiness and `native_fusions: active (bf16-regions, attention, normfuse)`; `GET /v1/models`
lists the supported tasks. Generation returns a PNG in `data[0].b64_json`:

```sh
curl http://127.0.0.1:8191/v1/images/generations -H 'Content-Type: application/json' -d '{"model":"paiton-image-2.1","prompt":"A glass dragon sticker","mode":"rgba","size":"2048x2048","seed":42,"n":1}' > response.json
```

The image endpoints use JSON. `/v1/images/edits` accepts the same fields with
`size: "1024x1024"` and a base64 input PNG in `image_b64`; it does not accept
multipart uploads or remote image URLs. The included client handles encoding
and saving. This image engine uses Diffusers and its own API, separate from the
repository's `paiton serve` vLLM language-model presets.

| Setting | Qualified scope |
| --- | --- |
| Text-to-image and RGBA | 1024 × 1024 or 2048 × 2048 |
| Editing | One input image, up to 4,194,304 pixels; 1024 × 1024 output |
| Sampling | 40 steps, guidance 1.0, batch one |
| Requests | One active request; prompts up to 512 characters |
| Memory | Components resident; no CPU model offload or VAE tiling |
| Precision | Balanced MXFP4 weight storage; BF16 activation arithmetic and intentionally BF16 components |
| Graphs | Whole-pipeline capture unsupported |

Maximum sampled complete-device memory in the latest container repeat was
**28.34 GiB**. Smaller GPUs and other architectures are not qualified. Loading
also uses host memory; qualification used a host with approximately 16 GiB RAM.
The container occupies about 10.4 GB before registry compression. Allow additional
disk space for the 9.33 GB checkpoint, runtime caches and generated images.

## Download, offline use and fallback

Append `--download-only` to the container command or shell launcher to download
and verify without initializing the GPU. Append `--offline` to require the
already populated cache. Network access is disabled in the model loader after
checkpoint preparation; it does not execute Python code downloaded from the Hub.

Add `serve --precision-profile exact` to the launcher
(`./models/Qwen-Image-2.1/serve-docker.sh serve --precision-profile exact`, or after the image
name in the `docker run` form) to keep every step bit-exact against the pinned BF16 arithmetic
(the v1.0.1 contract, with the exact native attention). The default profile `schedule-int8`
keeps the cached text prefix and the first seven denoising steps exact and runs steps 8–40 with
int8 activations and int8/fp8 attention through native kernels; the checkpoint bytes are
unchanged, the receipts record the profile, and [measurements](measurements/low-precision-r9700.json)
hold the matched samples and quality grades. `serve --precision-profile schedule-int8-11` keeps
ten steps exact with int8-QK attention (more quality headroom, measured 112.7 s). Append `--no-native-fusions` to use the
existing native weight reconstruction with the original BF16 regions. The optimized regions check the pinned framework,
upstream sources, artifact hashes, ABI and target before use. Unsupported upstream
contracts retain original regions and report that state in `/health`. The managed
image pins the qualified environment and enables the optimized path by default.

The checkpoint is fixed at revision
`b52373faeac233fb27c8bba2d334729208ee7b22`. The complete file allowlist and hashes
are in [checkpoint.lock.json](checkpoint.lock.json). To reuse a local published
snapshot, mount it read-only and pass `--model-dir /path/in/container`; it must
pass the same verification. Weights are never requantized at startup.

## Existing environments and container build

The container is the reproducible entry point. A native launch requires CPython
3.12 and the exact pinned Torch/ROCm and Python dependencies in
`requirements-torch.txt`, `requirements-python.txt` and `requirements.txt`.
Use an isolated environment; these are image-runtime requirements, not additions
to the vLLM plugin's dependencies.

```sh
PAITON_PYTHON=/path/to/qualified/python ./models/Qwen-Image-2.1/launch.sh serve --host 127.0.0.1
```

Build from the public runtime package only:

```sh
docker build -t paiton-qwen-image21:local models/Qwen-Image-2.1
PAITON_IMAGE=paiton-qwen-image21:local ./models/Qwen-Image-2.1/serve-docker.sh
```

The native libraries are already compiled and shipped under `artifacts/`; the
proprietary compiler and generated implementation source are not needed or included.
The complete model still uses its pinned external framework runtime.

## License and attribution

**Built with Qwen.** Model weights retain the upstream [Qwen Research License](LICENSE)
and [NOTICE](NOTICE). The adapter and distributed Paiton runtime binaries use the
repository's Apache-2.0 runtime license. Dependency licenses remain applicable.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Release v1.0.2

The managed image runs a **precision schedule** by default. The cached text-prefix pass and the
first seven denoising steps stay bit-exact against the pinned BF16 arithmetic; steps 8–40 run int8
activations (one scale per 256-element block, quantized inside the fused LayerNorm) through wave64
int8 WMMA GEMMs against an exact-weight int8 reconstruction of the unchanged MXFP4 checkpoint, with
an int8-QKᵀ / fp8-P·V attention on those steps. Exact steps use the wave64 exact attention kernel.
`--precision-profile exact` keeps every step bit-exact: that path is the exact native attention and
fused normalization qualified for this release, **165.588 → 136.285 seconds** warm
against v1.0.1 (17.70% lower latency, denoiser tensors bit-exact).

In two matched fresh-process pairs against that exact path, the default profile changed warm complete HTTP
median from **136.65 to 103.64 seconds** (24.2% lower latency) and the first request from
**148.50 to 115.37 seconds**, with whole-device use at 25.62 GiB. Against the exact images
the default grades 41.8 dB PSNR / 0.0008 LPIPS at 2048×2048 and 50.2 dB at
1024×1024 (gates: PSNR ≥ 35 dB, LPIPS ≤ 0.02, CLIP Δ ≥ −0.01). The RGBA, editing, 1024 and A-B-A checks passed (mode suite 50.2 / 35.0 / 52.0 / 50.2 / 48.9 dB, repeat byte-identical: True); the 2048² RGBA case sits at 35.02 dB against the 35 dB gate, so transparent-image workloads wanting more margin can use `schedule-int8-11` (38.5 dB there) or `exact`.

Fresh containers of the published image (three with the default profile, one with the exact profile; seed 42 first, seed 43 warm, isolated caches)
measured **103.29 seconds warm** and **113.95 seconds first request** with the default profile,
**133.74 / 147.83 seconds** with `--precision-profile exact`; startup-to-ready median
65.00 s, whole-device peak 25.55 GiB.

Compared with v1.0.1 (165.141 s warm in its container repeat), the default profile is
37% faster and the bit-exact profile 17% faster.
[Matched samples, quality and limits](BENCHMARKS.md#release-v102).

## Release v1.0.1

The managed image includes the qualified exact native weight reconstruction
improvement. In three matched fresh-process pairs, warm complete HTTP median
changed from **168.228 to 165.953 seconds** (1.35% lower latency) against
v1.0.0, with unchanged checkpoint, BF16 arithmetic and 2048/40/guidance-1 settings.
Native arithmetic/lifecycle tests, saved denoiser tensors, final image quality,
RGBA/editing and A-B-A checks passed. [Matched samples and quality](measurements/weight-reconstruction-r9700.json).

A separate three-process repeat of this same runtime measured **165.141 seconds
warm**, range **165.098–166.725 s**, sample SD **0.927 s**. Each process used seed
42 first and seed 43 warm, with isolated application caches. First-request median
was **179.319 s**; startup-to-ready median **75.050 s**; launch-to-first-image
median **256.170 s**. Whole-device request peak reached **28.336 GiB** with 5 ms
nominal sampling. The complete PNG delivery/client decoding is included; startup
and model download are separate. This repeat has no newly matched competing
backend and must not be pooled with the incremental qualification above.
[Individual container samples, settings, versions and hashes](measurements/container-followup-r9700.json).

The v1.0.1 container retains the qualified executable files and dependencies;
release metadata is updated. The original v1.0.0 tag remains available.
