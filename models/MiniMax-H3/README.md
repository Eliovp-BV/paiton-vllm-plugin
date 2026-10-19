# MiniMax H3 with Paiton on Radeon

Powered by MiniMax H3.

Local **video with native stereo audio** and up to **nine image, three video and three audio
references** on one **32 GB gfx1201 R9700**. v1.1.0 uses the pruned Ref2VA W4A8
denoiser, the required Qwen3-VL-32B intermediate-state encoder, the public eight-dimensional pruned
AdaLN table, and the four/eight-step H3 Turbo adapters. No hosted generation is used.

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

- **32 GB VRAM** (31.9 GiB usable); sampled driver peaks 30.46–30.84 GiB. Most weights are outside
  Torch's allocator; Torch memory is not total VRAM.
- **16 GB system RAM** tested for one worker using smart memory, Dynamic VRAM, two offload streams,
  `--fast-disk`, and a 2 GB GPU reserve. Do not use `--high-ram` on this host. Two workers need the
  host-RAM guidance above.
- **33.96 GB** selected weights per preset, **35.92 GB** for both adapters with shared components.
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
