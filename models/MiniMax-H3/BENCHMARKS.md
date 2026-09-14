# Continuous 15-second qualification

| Matched Turbo8, 15.08-second clip | Stock | Paiton |
| --- | ---: | ---: |
| Complete playable MP4 | 399.40 s | **332.85 s** |
| Pipeline before file encoding | 385.60 s | 318.98 s |
| Fixed-length clips per hour | 9.01 | **10.82** |
| Generated video seconds per wall second | 0.0378 | 0.0453 |


16.66% lower latency, 19.99% higher throughput. One fox warmup and two measured fresh requests at seed 771 per engine. The 362-frame clip is a single generation, not the earlier three-clip comparison reel. All video/audio latents and paired MP4s are identical.

| Component, measured mean seconds | Stock | Paiton |
| --- | ---: | ---: |
| conditioning | 21.718 | 10.821 |
| sampling | 317.987 | 262.296 |
| video decode | 42.521 | 42.473 |
| audio decode | 3.345 | 3.370 |
| encode mux | 13.800 | 13.873 |

One **32 GB R9700** is required. Long-clip sampled driver memory reached **30.34 GiB**. We tested **16 GB system RAM with 4 GB swap**; lifetime process RSS reached 12.09 GiB, and up to 1.24 GiB of process swap was sampled. **24 GB or more host RAM is recommended** for the browser, longer requests and other applications. Allow **60 GB free disk** for the approximately 6 GB image, 33.96 GB selected weights, caches and outputs. Both adapters share the large components and total 35.92 GB of checkpoint files. Use SSD storage. Filesystems without hard-link support can require another 34 GB for duplicate cache copies.

| Peak memory across these runs, GiB | Stock | Paiton |
| --- | ---: | ---: |
| Torch peak allocation | 10.653 | 7.805 |
| Torch peak reservation | 16.418 | 10.197 |
| Sampled driver VRAM | 30.283 | 30.338 |
| Sampled host RSS | 11.616 | 11.457 |
| Process lifetime RSS high-water mark | 12.091 | 11.752 |
| Sampled process swap | 0.488 | 1.239 |

Driver telemetry is sampled every 0.5 seconds and may miss brief peaks. Torch allocation, reservation and driver measurements overlap. No GPU clocks or profile settings were changed. See [long raw timings](assets/long15-timings.csv) and [complete public measurement data](assets/long15-results.json), including temperatures, per-phase disk reads and sampled GPU energy. Device energy is not wall energy. Phase loading costs remain inside the clock; disk reads are not a direct measurement of PCIe transfer time.

The 20% end-to-end latency engineering target remains unmet; the 19.99% throughput gain is a different metric.

The stock baseline uses the prior v4 image; the release image uses the extended v5 artifact. A file-hash and dependency-version comparison confirms identical stock implementation, model/VAE code, CLI, checkpoint lock and dependency lock. The artifact extends the accepted token count and retains existing arithmetic. Image identities and parity evidence are retained with this release.

---

# MiniMax H3 qualification — 8 September 2026

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
