# Qwen-Image-2.1 on RDNA4: history

Results and notes of earlier releases, moved here from the [guide](README.md) on 10 November 2026. The current
release and its numbers are in the guide.

## v1.0.2 summary

At **2048 × 2048, 40 steps, guidance 1.0**, v1.0.2 measured **103.64 seconds** median warm
complete-request latency against **136.65 seconds** for its bit-exact profile in matched fresh-process
pairs (24.2% lower latency), and the bit-exact profile itself is 17.7% faster than v1.0.1
(165.588 → 136.285 seconds, denoiser tensors bit-exact). Three fresh container processes of the published image measured **103.29 seconds** warm with the default profile (one with `--precision-profile exact`: **133.74 seconds**).
Checkpoint bytes and settings are unchanged; the default profile's precision schedule is described below.

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

## v1.0.3 and v1.1.0

v1.0.3 added the `--model` selector (`original`, `uncensored`) to the existing runtime and native libraries.
`original` stayed the default with its `schedule-int8` precision profile. v1.1.0 added the 8-step Turbo model
(`--model turbo`).
