# MiniMax H3 Ref2VA — v1.1.0 qualification

19 October 2026. One Radeon AI PRO R9700 (gfx1201, 32 GB), public Ref2VA W4A8 checkpoint
`daf6a822…`, stock `UNETLoader`, packed W4A8 confirmed, `projection_scope` active. The compiled
denoising artifact ships inside the container image; its manifest and `release.lock.json` record the
hash. Local release candidate; not published.

## Equality with the stock engine

Latent tensors are **bitwise identical** at 1 reference; decoded frames and PCM are identical at
1 and 9 references. The graph is exact at tensor level:

| Tensor | Shape | dtype | max_abs_diff | mismatches |
|---|---|---|---:|---:|
| video | `[1,24,37,30,54]` | float32 | 0.0 | 0 |
| audio | `[1,32,2,207]` | float32 | 0.0 | 0 |

Source: `results/equality-latents-compare.json` (`pass: true`). Decoded-frame/PCM comparisons are in
`results/equality-ref1-turbo4.json` and `results/equality-ref9-turbo4.json`.

- The MP4 container bytes differ only through the embedded workflow JSON.
- Paiton sampling is about 20% faster at 864×480; end-to-end wall time
  includes model load, encoding and muxing.

## 864×480, 124 frames (5.167 s), one GPU

Batch one, 24 fps, native 32 kHz stereo, Euler/simple, shifts 6/3.

| Preset | refs | Wall | Sampling | Per-step | Peak driver VRAM |
|---|---:|---:|---:|---:|---:|
| Turbo4 | 1 | 50.6 s | 22.0 s | 5.50 s | 30.60 GiB |
| Turbo4 | 3 | 85.3 s | 25.2 s | 6.31 s | 30.84 GiB |
| Turbo4 | 9 | 178.7 s | 45.3 s | 11.32 s | 30.60 GiB |
| Turbo8 | 1 | 109.2 s | 56.9 s | 7.11 s | 30.58 GiB |
| Turbo8 | 3 | 119.2 s | 58.1 s | 7.26 s | 30.66 GiB |
| Turbo8 | 9 | 221.5 s | 95.7 s | 11.96 s | 30.67 GiB |
| Base50 | 1 | 393.7 s | 360.0 s | 7.20 s | 30.55 GiB |
| Base50 | 3 | 417.2 s | 388.7 s | 7.77 s | 30.56 GiB |
| Base50 | 9 | 595.2 s | 565.6 s | 11.31 s | 30.57 GiB |

All runs produced 124 frames / 5.167 s / 24 fps, with finite 32 kHz stereo,
no duplicate or flat frames, and coherent contact sheets.

## 1344×768, 5 s, nine references at 512 short edge

| Preset | Wall | Sampling | Per-step | Peak VRAM | Notes |
|---|---:|---:|---:|---:|---|
| Turbo4 (warm) | 258.43 s | 118.0 s | 29.5 s | 30.46 GiB | 4 evaluations; under the 30.5 GiB gate |
| Base50 (warm) | 1753.15 s (29.2 min) | 1602.6 s (26.7 min) | 32.05 s | 30.39 GiB | 50 evaluations; under the 30.5 GiB gate |

Packed S is **44,625**: 37,296 video + 414 audio + 3,360 reference rows +
3,555 text/vision, 4,527 tokens under the 49,152 cap. `match` is **53,065 tokens** and is rejected fail-closed at
`PaitonH3Ref2VA` (node 5) with `H3 packed sequence 53065 exceeds the 49152-token cap`;
runner rc=1 is the expected rejection. Base50 host peak 9.95 GiB; MP4 sha256 `0cf4fb66…`; 0 flat/duplicate frames, finite audio. **25-step 768p = 15.9 min** (150.55 s fixed + 801.3 s sampling; misses the 15-min bar by
~0.9 min). The **50-step** run is **29.2 min** measured (meets). At 864×480, 25 steps is **~5.2 min**
(meets). The two-worker row is derived:
with the measured concurrency efficiency of **1.99×** (see "Two independent
workers"), two cards deliver ≈4.1 requests/hour on this case while
single-request latency stays unchanged. `max` is 55,952 and also rejected. 15 s at 768p is unsupported.

## Memory

Sampled driver peaks across the 864×480 matrix were **30.55–30.84 GiB**, i.e.
0.05–0.34 GiB over the earlier 30.5 GiB target and within the 31.9 GiB card. Warm 768p Turbo4 peaked
at **30.46 GiB** (Turbo4) and **30.39 GiB** (Base50), both under the 30.5 GiB gate.
Document the qualification gate at **31.0 GiB** or the overage. Most weights
sit outside Torch's allocator; torch-reported memory is not total VRAM.

Cold-start determinism: the first run after a cold container start (08:16)
differed from three later runs whose workflow JSON was byte-identical at seed
771; 3/4 runs agree bitwise. Prime suspect is first-pass MIOpen algorithm
selection in the reference VAE encode. Discard cold-first clips from public
numbers.

## Two independent workers (two R9700s)

Queue-compliant, one worker per GPU, 1-ref Turbo4, 864×480, 4 requests per
worker (8 total), submitted concurrently.

| Workers | Makespan | Serial equivalent | Throughput | Requests/hour | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | — | 414.29 s | 1.00× | 69.5 | 51.70 s | 52.52 s |
| 2 | 208.16 s | 414.29 s | **1.99×** | **138.35** | 51.70 s | 52.52 s |

Per-request latency is unchanged; the second card adds throughput only. Each
worker serves one request at a time. Host RSS peaks: 10.80 GB (1 ref),
11.60 GB (1 ref 768p), 20.6 GB (3 refs), 22.56 GB (9 refs, 864×480).
32 GB host → one worker; 48 GB → two at 864×480; 64 GB → two at 768p.

## Token budget

| Canvas | References | Packed S | Result |
|---|---|---:|---|
| 864×480 5 s | 1 / 3 / 9 | 16,330 / 17,928 / 22,750 | supported |
| 864×480 5 s | 9 at `max` (`ref768-*`) | 30,218 | supported |
| 1344×768 5 s | 9 at 512 | 44,625 | supported, 4,527 headroom |
| 1344×768 5 s | 9 at `match` | 53,065 | rejected fail-closed |
| 1344×768 5 s | 9 at `max` | 55,952 | rejected fail-closed |
| 864×480 15 s | 9 at 512 | over cap | max 4 references |
| 1344×768 15 s | any | over cap | unsupported |

## Reproduction

Use the package launchers and the GPU queue; run one workload per card. Keep
the preset, seed and reference set identical between engines. Timing starts
before conditioning and ends after the playable MP4 is written, including
H.264/AAC. Downloads, image builds and imports are separate.

## Historical: FL2VA v1.0.0 baseline (8 September 2026)

H3 produces local video and native audio on one 32 GB Radeon AI PRO R9700.
The final compiled denoiser fuses Turbo residuals, paired SwiGLU and RDNA4
INT8 attention. The **20% end-to-end latency objective remains unmet**.
Nothing is published; this is a local review package.

## Matched complete-generation results

Both engines use the exact pinned W4A8 FL2VA denoiser, NVFP4-storage/FP16 Qwen3-VL
intermediate encoder, INT8 video VAE/BF16 compute, FP32 audio VAE and matching
Turbo adapter. Canvas 864×480, 124 frames, 24 fps, batch one. Euler/simple,
BasicGuider, no CFG second forward, shifts 6/3, and exactly four/eight NFEs.
Audio is jointly generated. Smart memory, Dynamic VRAM, two offload streams,
`--fast-disk`, 2 GB reserve and hardware profile are matched.

| Preset | Stock seconds | Paiton seconds | Lower latency |
|---|---:|---:|---:|
| Turbo4 | 64.15 s | 54.38 s | 15.24% |
| Turbo8 | 95.83 s | 80.60 s | 15.89% |

| Preset / prompt / seed | Stock measured seconds | Paiton measured seconds |
|---|---|---|
| Turbo4 / barista / 772 | 65.2211, 63.7795 | 54.9004, 54.0661 |
| Turbo4 / pouring_water / 773 | 63.7140, 63.8874 | 54.3518, 54.1864 |
| Turbo8 / barista / 772 | 96.3074, 95.9004 | 80.9720, 80.1997 |
| Turbo8 / pouring_water / 773 | 95.6352, 95.4844 | 80.8519, 80.3818 |

| Preset / engine | Warm before encoding, s | Playable clips/hour | Video seconds/wall second | First clip after setup, s | Setup/import/model creation, s |
|---|---:|---:|---:|---:|---:|
| Turbo4 / stock | 59.939 | 56.118 | 0.080540 | 109.160 | 18.321 |
| Turbo4 / paiton | 50.174 | 66.205 | 0.095017 | 103.771 | 16.207 |
| Turbo8 / stock | 91.738 | 37.566 | 0.053914 | 141.431 | 17.327 |
| Turbo8 / paiton | 76.479 | 44.664 | 0.064101 | 129.472 | 14.052 |

One fox warmup precedes two measured requests each for barista and pouring
water. Fresh conditioning runs every time. Timing starts before conditioning
and ends after the playable MP4 is written, including H.264 encoding/AAC mux.
Diagnostic latent saving is outside the boundary. Downloads, builds, imports
and model construction are separate. An existing filesystem/runtime cache is
not a completely cold installation. The GPU ran one workload at a time; no
power/profile/clock changes occurred. Sample size is small and fixed, without
a general performance guarantee. Four-step versus eight-step speed is not
counted as a Paiton gain.

## Reproduce the measured multi-prompt suite

After downloading both Turbo profiles, copy the bundled fixed cases into the
persistent data directory and run each engine sequentially:

```sh
cp benchmark-cases.json "${PAITON_H3_DATA:-$PWD/paiton-h3-data}/benchmark-cases.json"
./launch.sh --stop
./run.sh benchmark --preset turbo4 --engine stock --runs 5 \
  --cases /data/benchmark-cases.json --output /outputs/stock4-suite
./run.sh benchmark --preset turbo4 --engine paiton --runs 5 \
  --cases /data/benchmark-cases.json --output /outputs/paiton4-suite
./run.sh report /outputs/stock4-suite --measured 1 2 3 4
./run.sh report /outputs/paiton4-suite --measured 1 2 3 4
```

Repeat with `--preset turbo8` and different output directories. These five
cases are fox warmup, barista twice, then pouring water twice; each prompt is
encoded again. Never overwrite an earlier measurement directory.

## Components, residency and memory

| Mean component, seconds | Turbo4 stock | Turbo4 Paiton | Turbo8 stock | Turbo8 Paiton |
|---|---:|---:|---:|---:|
| conditioning | 5.2890 | 2.7132 | 4.7460 | 2.6402 |
| sampling | 34.3590 | 27.1853 | 66.7529 | 53.5448 |
| video_decode | 17.1231 | 17.1474 | 17.0965 | 17.1059 |
| audio_decode | 3.1371 | 3.0995 | 3.1109 | 3.1597 |
| encode_mux | 4.2101 | 4.2018 | 4.0936 | 4.1218 |

| Measured peak across both presets, GiB | Stock | Paiton |
|---|---:|---:|
| Sampled driver VRAM | 31.110 | 30.534 |
| Torch allocation | 4.955 | 4.151 |
| Torch reservation | 7.064 | 4.973 |
| Sampled process RSS | 9.517 | 9.428 |
| Process lifetime RSS high-water mark | 9.562 | 9.601 |
| Sampled process swap | 0.393 | 0.000 |

Driver/host telemetry is sampled every 0.5 seconds and can miss short peaks.
AIMDO weights sit largely outside Torch's allocator. These memory columns are
not additive. The host has 15.5 GiB usable RAM and a SATA SSD; pinned H2D was
about 11 GiB/s and direct SSD reads about 553 MB/s. Selected dynamic scheduling
is feasible on this 16 GB host. Larger host caches caused swapping; no extra
pinned-memory cache is in the package.

The hot denoiser stays resident. Component disk-read counts are retained for
every phase; loading and transfers are included in those clocks. Loader
staging/execution overlap, so phase movement is not reported as an invented
separate additive duration. A representative hot denoiser profile showed one
20.2 microsecond HtoD event and flat disk counters; its transfer byte count was
unavailable. This is not denoiser block streaming on every evaluation. Smaller
intermediate allocations reduce dynamic eviction and conditioning reloads.

| Mean phase disk reads, decimal GB | Turbo4 stock | Turbo4 Paiton | Turbo8 stock | Turbo8 Paiton |
|---|---:|---:|---:|---:|
| conditioning | 2.421 | 0.804 | 1.978 | 0.766 |
| sampling | 2.636 | 2.136 | 2.674 | 2.049 |
| video_decode | 2.435 | 2.430 | 2.422 | 2.421 |
| audio_decode | 0.257 | 0.257 | 0.257 | 0.257 |

Measured maximum temperatures were 59°C core, 86°C hotspot, 75°C memory. AUTO/COMPUTE was unchanged. Raw sampled clocks, power, host and driver telemetry remain with the local evidence.

## Correctness and quality

All ten paired clips, including both warmups, have bit-equal video/audio
latents and byte-identical MP4 files. All decoded files have 124 distinct
frames, 864×480 and 24 fps. Video lasts 5.1667 s. Generated audio is 165600
samples at 32000 Hz (5.175 s); AAC decoding returns 165888 samples because of
padding and the muxed audio duration is about 5.167 s. Do not describe the clip
as exactly five seconds or its track endings as exactly sample aligned.

- Four-step fox: coherent motion/anatomy, natural birds/water confirmed by the
  reviewer. Four-step barista: coherent cup handoff; reviewer confirmed clear
  speech and reasonable lip synchronization. Those listening confirmations
  cover these clips, not all prompts or the separate eight-step adapter.
- Four-step pouring: two bottles instead of the requested one. Flow/glass are
  coherent, but prompt adherence fails on object count. AAC has two channel
  samples at/above full scale (peak about 1.00053).
- Eight-step pouring: one bottle with pronounced foam and incomplete final
  placement. AAC has five full-scale/overshoot channel samples (peak about
  1.02007). The native waveform reaches its decoder clamp. No normalization,
  audio replacement or mixing hides these shared limitations.
- No exact duplicated frames or nonfinite outputs were found. Contact sheets
  and private comparison reels are retained. Visual inspection covers these
  fixed prompts; a single numerical score is not proof of broad quality.

Equality is with the selected lossy quantized stock path, not full-precision
original H3. The pruned AdaLN fit, W4 weights, encoder quantization, INT8 VAE and
step-distilled adapters all have their own quality limits. Hosted context
processing/2K regeneration are absent. Other workloads/settings are unqualified.

## Compiler/runtime validation and evidence

Seven ROCm artifact tests pass: complete projection outputs, boundary token
counts, W4/SwiGLU handling, Turbo residual scales 1/16 and 1, dense attention,
graph replay and the earlier FLUX projection ABI. Stock quantizers, adapter
GEMMs, scheduler, RNG consumption, positions, packing and normalization remain
intact. Four-step clips exercise 600 residual, 200 SwiGLU and 200 attention
calls; eight-step clips exercise twice those counts. Unsupported cases retain
native fallback. Both engines execute the packaged ComfyUI four-step graph;
its decoded media matches the CLI. UI cache reuse is a functional check, not a
fresh-prompt performance sample.

The selected artifact is:
`minimax_h3_projections_gfx1201.so`, 838384 bytes, SHA-256
`06c7232319e48b8332e75f7980ee4a6e596dad7d99a2875640858a94f14aca56`.
The Ref2VA contract rebuild (842,528 bytes, SHA-256
`ae475cbce00e231e6cc3e5ffd1d57cd225e0763c8ae99b01ae9f65d512992311`)
passes the exact kernel gates at 20,000–49,152 tokens but is not yet in the
measured table above.

Immutable checkpoint revisions/sizes/SHA-256: [checkpoints.lock.json](checkpoints.lock.json).
Raw per-run timing, memory, file hashes and case data:
[Turbo4 JSON](benchmark-data/container-turbo4-fused-v4.json),
[Turbo4 CSV](benchmark-data/container-turbo4-fused-v4.csv),
[Turbo8 JSON](benchmark-data/container-turbo8-fused-v4.json),
[Turbo8 CSV](benchmark-data/container-turbo8-fused-v4.csv).
Each engine's exact local image identity is in these records. Turbo4 stock
uses the preceding local image; its stock runner, CLI, checkpoint lock,
dependency lock and native H3 model/VAE code were SHA-256-verified identical.
Both images share the immutable base and pinned dependencies. The image change
adds Paiton's unit-scale adapter entry point. Turbo8 uses the final image for
both engines.

The binary is supplied in the local bundle; no public artifact endpoint exists.
Compiler rebuilds preserve generated sources, but HIP build identities can
change ELF hashes. The manifest pins the tested binary rather than promising
bit-reproducible ELF output. Commands/raw telemetry, retained clips, contact
sheets and unpublished demos remain local/private.

## Earlier experiments and rejected paths

Projection-only Paiton averaged 92.39 s versus 96.43 s stock (4.19%). New
profiling exposed adapter scaling/addition and attention costs; fusing those
operations produced the present result. Earlier baseline/reference records
remain alongside final data and are not current release claims.

Attention scheduling/packing and exact residual/SwiGLU fusion improved complete
clips. Wider tiles, extra pipeline stages, alternate attention barriers and
video-decoder graph capture lost or did not help. A 4 GiB host cache saved only
about 1.3 s and caused process swap, so it was rejected. Selective decoder
projections saved about one second but changed pixels (RMSE .00189); fused
RMS/RoPE and tile batching also introduced numerical variants for small gains.
These decoder changes were not promoted.

Published group-wise INT4/SVDQuant checkpoints inspected during this round use
different model/layout/runtime assumptions; CUDA/A100/4090 claims do not
establish gfx1201 support. A bounded 11.34 GB uncalibrated INT8-donor-to-W4A4
conversion did run native RDNA4 INT4 WMMA and produced a coherent clip, but
changed image detail/composition. Its approximately 7 s denoiser evaluations
were slower than the compiled W4A8 path. This probe is not a release checkpoint
and does not rule out calibrated group-wise INT4 with residual correction.
Original weights were preserved; selected release files require no local
conversion. No larger inference machine, hosted generation or paid resources
were used.
