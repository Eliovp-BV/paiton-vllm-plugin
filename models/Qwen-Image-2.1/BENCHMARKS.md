# Qwen-Image-2.1 R9700 measurements

One Radeon AI PRO R9700, 32 GB, gfx1201, unchanged 300 W power cap. Balanced
MXFP4 checkpoint, batch one, 2048 × 2048, 40 steps, guidance 1.0, seed 42 and
the same neon-street prompt for both paths. All model components remain resident;
no CPU model offload, VAE tiling, precision changes or approximate caching.

Exact prompt: `A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement`.

## Complete requests

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
hashes are in `checkpoint.lock.json`; runtime artifact hashes are in the two
`artifacts/**/manifest.json` files. The optimized-region SHA-256 is
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


## Managed container validation

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


## Qualified local follow-up

This checkout includes a further exact native weight reconstruction improvement.
In three matched fresh-process pairs, warm complete HTTP median changed from
**168.228 to 165.953 seconds** (1.35% lower latency), with the
same balanced checkpoint, BF16 arithmetic, 2048 resolution, 40 steps and guidance
1.0. This is an incremental comparison against the already optimized v1.0.0
runtime. Native arithmetic/lifecycle tests, saved denoiser tensors, image-quality
and RGBA/editing/A-B-A checks passed.

The published Docker command above still uses v1.0.0. This follow-up is available
in the local review checkout and has not been published in a new container.
[Samples and qualification limits](measurements/weight-reconstruction-r9700.json).
