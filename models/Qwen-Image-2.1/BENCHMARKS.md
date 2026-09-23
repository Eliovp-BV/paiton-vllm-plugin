# Qwen-Image-2.1 R9700 measurements

Current release: **v1.0.2**. Matched fresh-process pairs measured **136.65 → 103.64 seconds** warm for
its default precision schedule against its bit-exact profile, and **165.588 → 136.285 seconds**
for the bit-exact profile against v1.0.1. Three fresh container processes measured **103.29 seconds warm** (default profile); one with `exact` measured **133.74 seconds**.
See [v1.0.2 results](#release-v102). The v1.0.1 and v1.0.0 sections below are retained as historical
qualification.

One Radeon AI PRO R9700, 32 GB, gfx1201, unchanged 300 W power cap. Balanced
MXFP4 checkpoint, batch one, 2048 × 2048, 40 steps, guidance 1.0, seed 42 and
the same neon-street prompt for both paths. All model components remain resident;
no CPU model offload, VAE tiling or approximate caching. The default profile's precision schedule is
documented below; `--precision-profile exact` restores the bit-exact path.

Exact prompt: `A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement`.

## Release v1.0.2

Built from this checkout and private compiler commits `ab6affcf`, `b52d9314` and `234968af`. Two things changed
against v1.0.1: every step now runs the exact native attention and fused normalization (the `exact` profile,
bit-exact denoiser tensors), and the default profile `schedule-int8` adds a precision schedule on top of it.
Checkpoint bytes, residency, the 2048/40/guidance-1 workload and the allocation policy are unchanged.

### Bit-exact profile: exact native attention and fused normalization

Released as the `exact` profile of v1.0.2. Built from private compiler commit `ab6affcf`, it replaces
the framework attention of the dense text-to-image and cached decode steps with an exact native HIP kernel
that reproduces the pinned framework arithmetic bit for bit, reads the cached text prefix and the
token-major projections in place (no concatenation or relayout copies), and fuses each block's
LayerNorm with the preceding gated residual and the following modulation. Checkpoint bytes, BF16
arithmetic, residency, the 2048/40/guidance-1 workload and the allocation policy are unchanged.

| Measurement | v1.0.1 control | v1.0.2 exact profile | Latency reduction |
| --- | ---: | ---: | ---: |
| First request median | 176.981 s | 147.428 s | 16.70% |
| Warm request median | 165.588 s | 136.285 s | 17.70% |
| Warm range | 165.540–165.660 s | 136.248–136.296 s | |
| Warm sample standard deviation | 0.060 s | 0.025 s | |

| Pair | v1.0.1 warm | Exact profile warm | Saving |
| --- | ---: | ---: | ---: |
| 1 | 165.588 s | 136.285 s | 29.304 s |
| 2 | 165.540 s | 136.296 s | 29.245 s |
| 3 | 165.660 s | 136.248 s | 29.411 s |

Three fresh server processes per path (`-m paiton_image21 serve --native-fusions`, the control from the
v1.0.1 checkout), alternating order, one first and one warm request each, isolated application caches,
whole-device VRAM sampled every 5 ms (maximum **25.582 GiB** across all timed requests). Complete HTTP
timings include text encoding, denoising, VAE, PNG encoding and receipt of the JSON/base64 response;
startup was separate. This is a bounded matched reproducibility check on one R9700, not a population
confidence interval.

Saved denoiser outputs at steps 1, 20 and 40 of the exact profile are **bit-exact** against the v1.0.1 run
of the same request. Complete images differ from fresh-process v1.0.1 images only at the documented
cross-process VAE/post-processing variability level: over 10 graded pairs (neon workload, warm
benchmark pairs, and the RGBA/edit/repeat suite) RGB PSNR ranged **54.55–61.71 dB**,
maximum LPIPS **0.0002177**, minimum CLIPscore delta **-0.001065**, maximum alpha MAE
**0.02912**; every predefined gate passed. Each process repeated its first request
byte-identically after intervening RGBA and edit requests, with no prefix-cache objects retained.

The native attention serves calls without a mask or with an all-valid key mask; the segmented prefill
step, any other mask, or unsupported shapes keep the previous framework paths (counted per request in
the response metrics). Both companion libraries passed independent execution without Torch/Triton,
bounds/canary checks, nondefault streams, event handoffs, poisoned changed-input graph replays and A-B-A;
the attention library is bit-exact against the pinned framework on synthetic, one-hot and captured
denoiser calls in both memory layouts. Records:
[native-attention-r9700.json](measurements/native-attention-r9700.json).

### Default profile: precision schedule

The cached text-prefix pass and the first seven denoising steps run the bit-exact path. Steps 8–40 run int8
activations, one fp32 scale per 256-element block, quantized inside the fused LayerNorm/modulation kernel and
multiplied through wave64 int8 WMMA GEMMs against an int8 reconstruction of the unchanged MXFP4 weights (exact for
99.6% of channels, the rest rounded and counted per request). On those steps the attention computes QKᵀ from
per-row int8 Q and K (int32 scores, bf16 softmax) and P·V from fp8 probabilities and per-channel fp8 values;
exact steps use the wave64 exact attention (bit-exact, 45/45 reference cases). The switch point is counted per
transformer forward.

| Profile (two matched pairs each) | Exact control warm | Candidate warm | Warm reduction | First request |
| --- | ---: | ---: | ---: | ---: |
| `schedule-int8` (default): 7 exact steps, int8-QK + fp8-PV attention | 136.65 s | 103.64 s | 24.2% | 148.50 → 115.37 s |
| 7 exact steps, int8-QK attention (bf16 P·V) | 136.58 s | 112.56 s | 17.6% | 149.17 → 124.13 s |
| `schedule-int8-11`: 10 exact steps, int8-QK attention | 136.24 s | 112.72 s | 17.3% | 147.97 → 124.26 s |
| 10 exact steps, exact attention on every step | 136.51 s | 118.53 s | 13.2% | 147.74 → 129.82 s |
| wave64 exact attention only (bit-exact) | 136.31 s | 133.56 s | 2.0% | 147.85 → 145.25 s |

Default-profile warm samples: 103.79 / 103.48 s against 136.49 / 136.82 s; peak whole-device VRAM
25.62 GiB. Two alternating fresh server processes per path, one first and one warm request each, isolated caches,
whole-device VRAM sampled every 5 ms; complete HTTP timings to PNG receipt; startup separate. Bounded matched checks on one R9700.

Quality against the exact pipeline's images (gates: PSNR ≥ 35 dB, LPIPS ≤ 0.02, CLIP Δ ≥ −0.01):

| Image | `schedule-int8` (default) | 7 exact, int8-QK | `schedule-int8-11` | 10 exact, exact attention |
| --- | --- | --- | --- | --- |
| 2048² neon prompt (PSNR / LPIPS) | 41.8 dB / 0.0008 | 43.0 dB / 0.0010 | 42.1 dB / 0.0007 | 44.6 dB / 0.0006 |
| 1024² teapot prompt | 50.2 dB / 0.0005 | 50.8 dB / 0.0005 | 51.2 dB / 0.0005 | 51.6 dB / 0.0004 |
| Mode suite (1024 text, 2048 RGBA, 1024 edit, A-B-A, 2048 text) | 50.2 / 35.0 / 52.0 / 50.2 / 48.9 dB; repeat byte-identical: True |

The 2048² RGBA case is the most sensitive mode: the default grades 35.02 dB PSNR / 0.0088 LPIPS there against the 35 dB gate
(the ten-exact-step profile `schedule-int8-11` scores 38.5 dB on the same case); transparent-image workloads wanting more margin can
select that profile or `exact`.

The same formats applied to all 40 steps fail the gate (30–33 dB): the early steps decide the composition, and
fewer than seven exact steps costs quality quickly (prefix-only 36.7 dB). Simulated per-GEMM error does not predict
the image gate; the schedule was chosen on graded images.

Qualification of the released default (two 2048² requests in one fresh process, step-1 denoiser tensor bit-exact:
True; relative L2 at steps 20 / 40: 0.0093 / 0.0554) graded 41.8 dB, 41.8 dB against the exact reference images.

All eleven companion libraries passed independent execution without Torch/Triton, bounds/canary checks, nondefault
streams, event handoffs and A-B-A; the int8 and fp8 GEMMs are checked against fp32 references and the exact-weight
reconstruction against the BF16 path per channel. The int8 quantization exactness claim covers weights only; activations
are rounded on the low-precision steps by design. Records: [low-precision-r9700.json](measurements/low-precision-r9700.json).

### Container validation

Fresh containers of the published image `ghcr.io/eliovp/paiton-vllm-plugin:qwen-image21-mxfp4-rdna4-v1.0.2` (`sha256:c7ab0c5f900bf16c2b56fa5aa0f9dd7b8c32a0ea6106db954b43556c4cc07d97`): three with the default profile and one with `--precision-profile exact`, alternating, one first (seed 42) and one warm (seed 43)
request each, isolated caches, whole-device VRAM sampled every 5 ms, complete HTTP timing to PNG receipt; `/health` reported
the native regions active and every receipt carries the precision profile and the kernel hashes.

| Container | Startup to ready | First request | Warm request | Peak whole-device VRAM | Profile |
| --- | ---: | ---: | ---: | ---: | --- |
| lpimage-1-default | 69.1 s | 113.66 s | 103.35 s | 25.51 GiB | schedule-int8 |
| lpimage-1-exact | 85.1 s | 147.83 s | 133.74 s | 25.33 GiB | exact |
| lpimage-2-default | 60.0 s | 114.52 s | 103.24 s | 25.47 GiB | schedule-int8 |
| lpimage-3-default | 65.0 s | 113.95 s | 103.29 s | 25.55 GiB | schedule-int8 |

Default profile medians: warm **103.29 s**, first **113.95 s**, startup 65.00 s. Exact profile medians:
warm **133.74 s**, first **147.83 s**. [Container samples](measurements/low-precision-r9700.json).

## Original v1.0.0 complete-request qualification

The control uses Quark HIP weight reconstruction with an exact contiguous
attention layout. It is faster than the original Quark path. The candidate uses
the Paiton reconstruction binary and qualified BF16 regions. Both paths use the
same allocation policy, checkpoint bytes and activation/rounding contract.

| Measurement | Matched Quark HIP | Paiton optimized | Latency reduction |
| --- | ---: | ---: | ---: |
| First request median | 200.287 s | 179.702 s | 10.28% |
| Warm request median | 188.790 s | 168.101 s | 10.96% |
| Warm range | 188.707–188.882 s | 168.066–168.125 s | |
| Warm sample standard deviation | 0.087 s | 0.030 s | |

Three independent fresh processes per path each supplied one first request and
one warm request. Complete HTTP timings include text encoding, denoising, VAE,
PNG encoding and receipt of the JSON/base64 response. Additional instrumented
requests are excluded. Processes alternated, with isolated framework/library
caches and no concurrent GPU work. OS page caches were not dropped and the GPU
was not reset. Startup was separate: approximately **32 seconds** to readiness,
excluding model download. This is a bounded reproducibility check, not a
population confidence interval.

Historical Quark first/warm results were 196.6204/192.8927 seconds. The refreshed
original Quark warm result was 192.9299 seconds. The stronger matched control in
the table is used for the speedup claim. Portable-model results are
separate and are not R9700 controls. The original unpack-only Paiton path had not
shown a convincing complete-request improvement.

The integrated optimized CLI separately measured **179.590/168.020 seconds** for
its first/warm requests. The fallback passed its matched image check.

## Memory and quality

Maximum measured complete-device memory across the primary, RGBA/editing/cache
and integrated-plugin checks was **25.939 GiB**. Primary control/candidate peaks
were 25.348/25.347 GiB. Physical VRAM was sampled independently at a nominal 5 ms
interval, alongside the separate HIP sampler and allocator counters. Finite
sampling cannot exclude arbitrarily shorter transients. Both paths use expandable
allocator segments and a 26 GiB allocator budget; this is not a hard device quota.

Seven candidate image pairs passed the predefined pixel, LPIPS, CLIPscore and
alpha checks. RGB PSNR ranged from **54.29 to 61.66 dB**; maximum LPIPS was
**0.0002343** and minimum CLIPscore delta was **−0.000385**. Two control/control
comparisons recorded process variation. Captured denoiser outputs at steps 1, 20
and 40 on two prompts had **zero unequal elements**. Each backend repeated A
byte-identically after intervening RGBA and edit requests, with no prefix-cache
objects retained between requests. Exact per-request prefix caching is preserved.

The bounded suite does not establish equivalence for arbitrary prompts. These
checks compare the optimization to the same balanced checkpoint; they do not
remove the checkpoint's existing quantization loss against unquantized BF16.
The original weight-quality screen is available in the
[Hugging Face model card](https://huggingface.co/EliovpAI/Qwen_Image-2.1-MXFP4-Paiton-RDNA4).

Native libraries also passed independent execution without Torch/Triton,
nondefault streams, explicit event handoffs, changed-input graph replay, poisoned
outputs and A-B-A checks. These cover native regions with initialized fixed
buffers/shapes. Whole-engine graphs, concurrent/batched requests, other targets
or framework revisions and 2048 editing remain unsupported or unqualified.

## Reproduction and identities

The exact measurement distributions are in
[bf16-regions-r9700.json](measurements/bf16-regions-r9700.json), with the image
screen in [optimization-quality.json](measurements/optimization-quality.json).
Individual complete-request samples, startup times, memory and output hashes are
in [complete-requests.json](measurements/complete-requests.json).

Runtime: Torch `2.15.0.dev20260907+rocm10.0`, HIP `7.15.26333`, Diffusers commit
`7263f3317f6b392d62f41e9d75ed9d7e21fc5a5c`, Transformers `5.17.0`. The Dockerfile
and requirement files pin the managed runtime. Checkpoint revision and all file
hashes are in `checkpoint.lock.json`; runtime artifact hashes are in the
`artifacts/**/manifest.json` files (twelve libraries in v1.0.2). The optimized-region SHA-256 is
`96ca54713c96ec15068e25d252281aee225a57236c2627f3886d1b9a138817f5`.

Use the [launch and request commands](README.md) for the optimized path. For a
fresh first/warm pair, start a fresh worker with a new runtime cache and submit
the identical neon prompt and seed twice. Time submission through complete body
receipt; separate process startup and model download. A stock framework decoder
or unpack-only fallback does not reproduce the Quark HIP control in the table.

The standard-library client records both requests and their PNG hashes:

```sh
python3 models/Qwen-Image-2.1/benchmark.py --output benchmarks/qwen-image21-run1 --suite
```

`--suite` adds RGBA, editing and mixed-request A-B-A checks. Use a new output
directory for each run. This client records the API's HIP/allocator memory counters;
independent physical VRAM monitoring is additionally required for a device-memory
qualification like the one reported above.


## Original v1.0.0 managed container validation

The managed image was built only from public, pinned runtime dependencies and the
allowlisted adapter/binaries. An anonymous first launch downloaded and SHA-256
verified all 76 model files. A separate network-disabled, GPU-free preparation
check reused the cache successfully.

The real container API measured **180.327 seconds first request**
and **167.415 seconds warm**, with native regions active.
Cached startup, including complete file verification and loading, was
**57.065 seconds**; download is additional. This is one
packaging-verification process, separate from the three-process matched experiment
above. Its six requests passed repeatability, RGBA, editing and mixed-mode A-B-A
checks, with a maximum independently sampled device peak of
**25.818 GiB**.

The direct generation launcher and unpack-only fallback each completed an
additional matched 1024 request. Seventeen CPU contract/launcher tests passed;
seven container image pairs passed the unchanged quality screen. The standalone
native stream/event/graph executable also passed inside the container without
framework imports. See [container-validation.json](measurements/container-validation.json).

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
