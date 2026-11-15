# MiniMax H3 on one R9700

MiniMax H3 makes video with native stereo audio from a prompt and up to nine image, three video and three audio
references, through ComfyUI, on one Radeon AI PRO R9700 (32 GB). No hosted generation is used. Powered by MiniMax H3.

An 864×480 clip of 124 frames with audio takes **54.38 s** with Paiton in the four-step Turbo4 preset, against 64.15 s
with the stock engine. The eight-step Turbo8 preset takes 80.60 s against 95.83 s ([benchmarks](BENCHMARKS.md)).

Start it:

```sh
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
./models/MiniMax-H3/launch.sh
```

Then open ComfyUI at `http://127.0.0.1:8190`. The first launch builds the image and downloads the model components.

## v1.2: native runner (release candidate)

v1.2 runs a whole Ref2VA request on the R9700 with Paiton's own HIP kernels: text encoder, reference encoding, the
denoiser, the video and audio decoders. Python only prepares the inputs (NumPy, Pillow, tokenizers) and ffmpeg writes
the MP4. There is no PyTorch, Triton or ComfyUI in the container. It serves an HTTP API, and a small ComfyUI node sends
requests to it. v1.1.0 above stays available.

Time per request, Turbo4, one R9700 at 300 W, weights already loaded:

| Request | v1.2 | v1.1.0 |
|---|---:|---:|
| 864×480, 124 frames, 1 image reference | 30.4 s | 50.6 s |
| 864×480, 124 frames, 3 image references | 33.4 s | 85.3 s |
| 864×480, 124 frames, 9 image references | 44.3 s | 178.7 s |
| 1344×768, 124 frames, 9 image references at 512 px | 114.1 s | 258.4 s |
| 864×480, 124 frames, 1 video reference with its soundtrack | 203.8 s | not measured |

Turbo8 takes 51.4 s for the first row. How these were measured, the per-phase split and the quality comparison with
the BF16 model are in [BENCHMARKS.md](BENCHMARKS.md#v12-native-runner-release-candidate).

### Run it

```sh
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
PAITON_DETACH=1 ./models/MiniMax-H3/native/launch.sh http
docker logs -f paiton-minimax-h3-v12        # wait for "READY http://..."; the first start downloads the weights
curl -o panda.mp4 -F image=@panda.png \
  -F prompt='The red panda from <Picture 1> walks through a sunlit forest. Audio: birds and a stream.' \
  http://127.0.0.1:30000/v1/videos/generations
```

The server keeps the weights of one preset loaded (Turbo4 by default, `launch.sh http --preset turbo8` for Turbo8)
and runs one request at a time; more requests wait. It listens on `127.0.0.1:30000`. Set `PAITON_PORT` for another
port and `PAITON_BIND=0.0.0.0` to accept requests from other machines (it has no authentication). Stop it with
`docker stop paiton-minimax-h3-v12`.

### HTTP API

`POST /v1/videos/generations` takes `multipart/form-data` or JSON and answers with the MP4 (`video/mp4`, H.264 and
32 kHz stereo AAC). The header `X-Paiton-Metrics` holds the time per phase and the VRAM peak as JSON.

| Field | Default | Meaning |
|---|---|---|
| `prompt` | required | the text; refer to the references as `<Picture 1>` to `<Picture 9>` (and `<Video 1>`) |
| `image` | 1 to 9 | one file field per reference image, in prompt order (JSON: `images`, a list of base64 strings) |
| `video` | none | one video reference; its soundtrack must be 32 kHz (JSON: `video`, base64). Images and video together: at most 9 |
| `width`, `height` | 864, 480 | multiples of 32, at most 1344×768 |
| `seconds` or `num_frames` | 5 s (124 frames) | 24 fps; `num_frames` must be 17 n + 5; up to 15 s at 864×480 |
| `seed` | 0 | the same seed and inputs give the same clip |
| `preset` | the server's | must match the preset the server was started with |

A request that breaks these rules, or does not fit the card (about 82,000 tokens; see the limits below), gets a
400 reply with a JSON `error` before anything runs on the GPU. `GET /health` reports `ready`, `busy` and the preset.

The same with JSON:

```sh
python3 - <<'PY'
import base64, json, urllib.request
body = json.dumps({"prompt": "The red panda from <Picture 1> naps in the sun.", "seconds": 5, "seed": 1,
                   "images": [base64.b64encode(open("panda.png", "rb").read()).decode()]}).encode()
request = urllib.request.Request("http://127.0.0.1:30000/v1/videos/generations", body, {"Content-Type": "application/json"})
open("nap.mp4", "wb").write(urllib.request.urlopen(request).read())
PY
```

### ComfyUI

Copy `native/comfyui/paiton_h3v_client` into ComfyUI's `custom_nodes` folder and restart ComfyUI. The node
**MiniMax H3 Ref2VA (Paiton server)** takes an IMAGE (a batch counts as several references), up to eight more
optional images, the prompt, size, length, seed and the server address, and returns a VIDEO for `SaveVideo` or a
preview. All the work runs in the Paiton server, so ComfyUI can run on the CPU or on another machine.
`native/comfyui/workflows/ref2va-server.api.json` is an example with two references.

### Command line

`generate` runs one request without a server:

```sh
./models/MiniMax-H3/native/launch.sh prepare
./models/MiniMax-H3/native/launch.sh generate --prompt 'The red panda from <Picture 1> walks through a sunlit forest. Audio: birds and a stream.' \
  --ref panda.png --out panda.mp4
```

`launch.sh` runs the image `ghcr.io/eliovp/paiton-vllm-plugin:minimax-h3-rdna4-v1.2.0` and mounts the current
directory, so references and the output must be inside it. `prepare` downloads the weights for one preset from
[EliovpAI/MiniMax-H3-FP8-Paiton-RDNA4](https://huggingface.co/EliovpAI/MiniMax-H3-FP8-Paiton-RDNA4) (the text
encoder from [EliovpAI/MiniMax-H3-Image-FP8-Paiton-RDNA4](https://huggingface.co/EliovpAI/MiniMax-H3-Image-FP8-Paiton-RDNA4),
which uses the same files) and checks every file against the SHA-256 in `native/checkpoints.lock.json`: 50.9 GB for Turbo4, 20.5 GB more for Turbo8.
Each `generate` starts the container and loads the weights again: on our test host the first example took 46.6 s
from the command to the MP4, 30.3 s of it for the request itself. To keep the weights loaded, use `serve`, which reads
one JSON request per line and answers `DONE <file>` or `ERROR <message>`:

```sh
echo '{"prompt": "...", "refs": ["panda.png"], "width": 864, "height": 480, "num_frames": 124, "seed": 1, "out": "a.mp4"}' \
  | ./models/MiniMax-H3/native/launch.sh serve
```

Options of `generate`: `--preset turbo4|turbo8` (default `turbo4`), `--width`/`--height` (default 864×480; use
1344×768 for 768p), `--frames` (default 124, 24 fps), `--seed`, and `--ref` once per reference, in the order
`<Picture 1>`, `<Picture 2>`, ... of the prompt. A reference ending in `.mp4`, `.mov`, `.mkv` or `.webm` is a video
reference; its soundtrack is used as reference audio. Settings for `launch.sh`: `PAITON_CACHE` (host folder for the
weights, default `~/.cache/paiton/minimax-h3-v12`), `PAITON_RENDER` (one render node such as `/dev/dri/renderD128` on a
machine with several GPUs), `HF_TOKEN`.

### Requirements

- One Radeon AI PRO R9700 (gfx1201, 32 GB) with a ROCm kernel driver, Docker, and nothing else on that card. The
  runner keeps VRAM under a 27.5 GiB cap; the measured peak was 26.8 GiB.
- Host RAM: the text encoder's 50 layers (24.4 GB) are copied into pinned host memory only when 16 GiB of available
  memory is left after the copy, so about 41 GB available or more. With less, the runner streams the layers from the
  weight files on disk. Both modes give byte-identical output; on our test host the streamed text encoder took 2.8 s
  per request instead of 0.8 s, with the files in the page cache. We have not run it on a host with less than 64 GB.
- Disk: 51 GB for the Turbo4 weights, 71 GB for both presets, 2.9 GB for the image, plus room for the outputs.

### Limits of v1.2

- A request is limited to about 82,000 packed tokens, the most that fits under the VRAM cap with the FP8 denoiser.
  1344×768 with nine references fits up to about 9.4 s (226 frames). 15 s at 768p does not fit. A request that does
  not fit is refused before it allocates memory.
- No audio-only references (v1.1.0 accepts three). The soundtrack of a video reference must already be 32 kHz; other
  rates are refused.
- Turbo4 and Turbo8 only; Base50 stays on v1.1.0.
- The same seed gives a different clip than v1.1.0 or the diffusers pipeline, because the noise comes from a different
  random generator.
- The denoiser uses FP8 weights. With many references the clip can differ from the BF16 model's clip in layout (who
  stands where), while the prompt match stays the same. See [BENCHMARKS.md](BENCHMARKS.md#quality-against-the-bf16-model).

The v1.2 weights are converted from MiniMax H3 with the Turbo adapters merged in, and stay under the MiniMax H3
Community License (see [Licenses](#licenses)). The compiled libraries and the runner are only in the container image.

v1.1.0 with ComfyUI: [Start](#start-locally) · [Terminal generation](#terminal-generation-and-benchmarks) ·
[Performance and limits](#qualified-performance-and-limits) · [Requirements](#requirements-and-model-precision) ·
[Existing downloads](#model-weights-and-existing-downloads) · [Licenses](#licenses)

v1.1.0 uses the pruned Ref2VA W4A8 denoiser, the required Qwen3-VL-32B intermediate-state encoder, the public
eight-dimensional pruned AdaLN table, and the four/eight-step H3 Turbo adapters.

## Start locally

With Docker/Compose and a working ROCm driver, from the repository root:

```sh
./models/MiniMax-H3/launch.sh
```

This builds the local image, downloads SHA-256-verified pinned Turbo8 components once, and opens
ComfyUI at `http://127.0.0.1:8190`. Load a Ref2VA workflow, select **stock** or **paiton** in
**MiniMax H3 Engine**, edit the prompt and queue. Use `./models/MiniMax-H3/launch.sh --logs` or `--stop`. For four
steps, run `./models/MiniMax-H3/run.sh download --profile turbo4` and load `turbo4.json`. To install only the faster
preset initially: `PAITON_H3_PRESET=turbo4 ./models/MiniMax-H3/launch.sh`.

Models, verification stamps and runtime caches persist in `./paiton-h3-data`; outputs go to
`./paiton-videos`. Override `PAITON_H3_DATA` and `PAITON_H3_OUTPUTS` with existing absolute
directories. The UI keeps model objects with 2 GB host-RAM headroom. Identical cached UI requests are
not fresh-prompt benchmarks.

Reference inputs use `MiniMaxH3ReferenceToVideo` (the Paiton node `PaitonH3Ref2VA` forwards to it and
preflights the token cap): nine **separate** image inputs, three videos with optional soundtracks,
and three standalone audio inputs. `ref_image_size` is `match` (default) or `max`; reference videos
ignore it. Order is images → videos → audio and advances the shared RoPE clock.

## Terminal generation and benchmarks

```sh
./models/MiniMax-H3/launch.sh --stop
./models/MiniMax-H3/run.sh download --profile turbo4
./models/MiniMax-H3/run.sh generate --preset turbo4 --engine paiton \
  --prompt 'A red fox beside a stream. Audio: birds and flowing water.' \
  --seed 771 --ref-image /outputs/ref0.png --ref-image /outputs/ref1.png \
  --output /outputs/ref2va
./models/MiniMax-H3/run.sh benchmark --preset turbo4 --engine stock --output /outputs/stock4
./models/MiniMax-H3/run.sh benchmark --preset turbo4 --engine paiton --output /outputs/paiton4
./models/MiniMax-H3/run.sh report /outputs/paiton4
./models/MiniMax-H3/run.sh inspect /outputs/paiton4/clip-2.mp4 --output /outputs/paiton4/quality
```

For 768p, pass nine references pre-scaled to 512 short edge and `--width 1344 --height 768`.
Benchmark mode runs one warmup and two measured generations, encoding the prompt again each time. It
retains commands, timings, telemetry, clips and latent tensors. Use the same preset and settings for
both engines. User-visible timing includes H.264 encoding and AAC muxing. Run one GPU workload at a
time.

## Qualified performance and limits

**864×480, 124 actual frames at 24 fps: 5.1667 video seconds**, batch one, Euler/simple, shifts
video/audio 6/3, BasicGuider, and native 32 kHz stereo. Four and eight steps mean exactly four and
eight denoiser evaluations.

| Preset | 1 ref | 3 refs | 9 refs |
|---|---:|---:|---:|
| Turbo4 wall / sampling / per-step / peak | 50.6 / 22.0 / 5.50 s / 30.60 GiB | 85.3 / 25.2 / 6.31 s / 30.84 GiB | 178.7 / 45.3 / 11.32 s / 30.60 GiB |
| Turbo8 wall / sampling / per-step / peak | 109.2 / 56.9 / 7.11 s / 30.58 GiB | 119.2 / 58.1 / 7.26 s / 30.66 GiB | 221.5 / 95.7 / 11.96 s / 30.67 GiB |
| Base50 wall / sampling / per-step / peak | 393.7 / 360.0 / 7.20 s / 30.55 GiB | 417.2 / 388.7 / 7.77 s / 30.56 GiB | 595.2 / 565.6 / 11.31 s / 30.57 GiB |

**1344×768, 5 s, nine references at 512 short edge:**

| Preset | Wall | Sampling | Per-step | Peak |
|---|---:|---:|---:|---:|
| Turbo4 (warm) | 258.43 s | 118.0 s | 29.5 s | 30.46 GiB |
| Base50 | 1753.15 s (29.2 min) | 1602.6 s (26.7 min) | 32.05 s | 30.39 GiB |

At 768p, **25 steps is 15.9 min** (150.55 s fixed + 801.3 s sampling; misses the 15-min bar by
~0.9 min) and **50 steps is 29.2 min** (meets). At 864×480, 25 steps is **~5.2 min** (meets).

Paiton vs stock: latent tensors are bitwise identical at 1 reference (video `[1,24,37,30,54]`, audio
`[1,32,2,207]`, max_abs 0.0, 0 mismatches); decoded frames and PCM are bitwise identical at 1 and 9
references. The graph is exact at tensor level. Both 768p cells are under the 30.5 GiB gate (Turbo4 30.46 GiB, Base50 30.39 GiB); the 864×480 matrix
peaks at 30.55–30.84 GiB. The stock/Paiton sampling delta is about 20%
at 864×480; the four-vs-eight-step difference is a distillation tradeoff, not a compiler speedup.
Detailed records and methodology: [BENCHMARKS.md](BENCHMARKS.md).

**Token budget.** The cap is 49,152 packed tokens. 864×480 with 1/3/9 references is
**16,330 / 17,928 / 22,750** (9 refs at `max` with `ref768-*`: 30,218). 1344×768 with nine references at 512 short edge is **44,625** (37,296 video +
414 audio + 3,360 reference rows + 3,555 text/vision), leaving **4,527** tokens. `match` is 53,065
and is rejected fail-closed by `PaitonH3Ref2VA` (node 5) with
`H3 packed sequence 53065 exceeds the 49152-token cap`; `max` is 55,952 and also rejected. 15 s at
768p is unsupported; 864×480 supports 15 s without references, with at most four references at 512.

**Two R9700s = two independent workers.** One GPU is the product; two do not make one request faster.
Two workers behind a queue measured **1.99×** throughput (138.35 requests/hour) with unchanged
per-request latency (p50 51.70 s, p95 52.52 s). Each worker serves one request at a time. Host RAM:
32 GB → one worker; 48 GB → two at 864×480; 64 GB → two at 768p.

## Requirements and model precision

- 32 GB VRAM (31.9 GiB usable); sampled driver peaks 30.46–30.84 GiB. Most weights are outside
  Torch's allocator; Torch memory is not total VRAM.
- 16 GB system RAM tested for one worker using smart memory, Dynamic VRAM, two offload streams,
  `--fast-disk`, and a 2 GB GPU reserve. Do not use `--high-ram` on this host. Two workers need the
  host-RAM guidance above.
- 33.96 GB selected weights per preset, **35.92 GB** for both adapters with shared components.
  Allow **60 GB free disk** for the image, weights, caches and outputs; reference weights need more.
- Linux, `/dev/kfd`, `/dev/dri`, Docker/Compose. Pinned Torch **2.12.0+rocm7.14.0**, Triton
  **3.7.1 ROCm**, ComfyUI `efa6c8f804bff78b46a0fd458ebd2e47bba07a30`, kitchen **0.2.33**, AIMDO
  **0.5.2**. AUTO/COMPUTE was retained; clocks, power, fans and voltage were unchanged.

The denoiser's 200 main projections use W4 codebook storage, group 16, ConvRot 256, FP8 group scales,
INT8 activations, INT32 GEMMs and BF16 outputs. The Qwen3-VL encoder uses NVFP4/AWQ storage emulated
with FP16 computation and retains the required 50 decoder layers/intermediate output; it is not
replaced by a smaller model. Video VAE uses INT8 weights/BF16 compute; audio VAE remains explicitly
FP32. The pruned AdaLN fit is approximate, not an exact cache of original 33B modulation weights.

[checkpoints.lock.json](checkpoints.lock.json) pins upstream revisions, sizes and full SHA-256 values.
The downloader preserves bytes and hard-links cached files when possible.
`download --verify-only --rehash` checks complete files. The selected files need no local
conversion/calibration. Original full-precision quality parity is unproven; publisher calibration
claims are not independent H3 quality proof.

## Model weights and existing downloads

Run the examples from the repository root. The selected denoiser, text/vision encoder, VAEs and Turbo
adapters are pinned in [checkpoints.lock.json](checkpoints.lock.json). The launcher downloads and verifies
them on first use. To reuse an existing H3 data directory, set
`export PAITON_H3_DATA="/absolute/path/to/h3-data"`; the directory contains `models/` and `cache/`.
See [the model weights guide](../../docs/MODEL_WEIGHTS.md) for offline and Hugging Face cache mounts.

Container image:
`ghcr.io/eliovp/paiton-vllm-plugin:minimax-h3-rdna4-v1.1.0@sha256:bd8c49a2440139b8b65dce2041ff4877298fd39a5e34128f6d2b594ef450cbed`.
Hugging Face runtime package:
[`EliovpAI/MiniMax-H3-W4A8-Paiton-RDNA4@paiton-rdna4-v1.1.0`](https://huggingface.co/EliovpAI/MiniMax-H3-W4A8-Paiton-RDNA4/tree/paiton-rdna4-v1.1.0).

**Release package.** The combined runtime is distributed under the licenses in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). H3 regional authorization was confirmed for this
qualification. The compiled artifact ships inside the container image; its manifest and
`release.lock.json` record the hash.

## Licenses

The Paiton runtime bindings are Apache-2.0. The compiled artifact contains a modified comfy-kitchen
attention kernel (Apache-2.0; see [MODIFICATIONS.md](MODIFICATIONS.md)). MiniMax H3 is licensed under
the **MiniMax H3 Community License**, which excludes the European Union, the United Kingdom, the
Republic of Korea and the United States from its territory grant and has commercial-scale conditions.
Quantization, Turbo adapters and reference conditioning do not remove those terms. Weights are
downloaded from their publishers and are not relicensed or mirrored here. ComfyUI and AIMDO are
GPL-3.0 and are installed only in the user-built local image; the published artifact image does not
contain them. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the retained licenses in
`LICENSES/` before distributing anything built from this package.
