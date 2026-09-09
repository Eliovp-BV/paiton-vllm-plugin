# R9700 benchmark protocol and results

Fourteen container configurations completed, with 56 retained clips including warmups. Raw records are in [benchmark-data](benchmark-data); no slow measured runs were removed. This is a local, unpublished release candidate.

## Complete measured results
Positive change means lower complete-clip latency. These are means of two measured runs after two retained warmups, at 24 fps. Small differences should be read with the raw ranges.

| Setting | Stock clip s | Paiton clip s | Paiton change |
| --- | ---: | ---: | ---: |
| FastWan 832×480, 49f, fox | 9.280 | 8.843 | +4.71% |
| FastWan 1280×704, 49f, person | 20.376 | 19.797 | +2.84% |
| FastWan 1280×704, 121f, fox | 51.986 | 50.109 | +3.61% |
| FastWan 832×480, 121f, fox | 22.031 | 21.185 | +3.84% |
| Base 480×832, 49f, image | 29.518 | 29.758 | -0.81% |
| Base 480×832, 121f, image | 79.648 | 79.757 | -0.14% |
| Base 832×480, 49f, text | 30.127 | 29.783 | +1.14% |

FastWan improves 2.84–4.71% in this batch. Base text improves 1.14%; the two base image cases are slightly slower end to end despite faster pipeline time before encoding. Stock is therefore the base workflow default. No 20% claim is made.

### Pipeline time and video throughput
| Case / engine | Complete s [range] | Pipeline s | Clips/hour | Video s / wall s |
| --- | ---: | ---: | ---: | ---: |
| [fast-480-2 / stock](benchmark-data/fast-480-2-stock/results.json) | 9.280 [9.227, 9.334] | 7.853 | 387.91 | 0.22000 |
| [fast-480-2 / paiton](benchmark-data/fast-480-2-paiton/results.json) | 8.843 [8.717, 8.969] | 7.641 | 407.11 | 0.23088 |
| [fast-720-2-person / stock](benchmark-data/fast-720-2-person-stock/results.json) | 20.376 [20.256, 20.495] | 17.595 | 176.68 | 0.10020 |
| [fast-720-2-person / paiton](benchmark-data/fast-720-2-person-paiton/results.json) | 19.797 [19.784, 19.810] | 17.065 | 181.85 | 0.10313 |
| [fast-720-5 / stock](benchmark-data/fast-720-5-stock/results.json) | 51.986 [51.388, 52.584] | 46.758 | 69.25 | 0.09698 |
| [fast-720-5 / paiton](benchmark-data/fast-720-5-paiton/results.json) | 50.109 [50.035, 50.184] | 44.852 | 71.84 | 0.10061 |
| [fast-480-5 / stock](benchmark-data/fast-480-5-stock/results.json) | 22.031 [21.904, 22.158] | 19.239 | 163.41 | 0.22885 |
| [fast-480-5 / paiton](benchmark-data/fast-480-5-paiton/results.json) | 21.185 [21.101, 21.269] | 18.695 | 169.93 | 0.23798 |
| [base-480-2-image / stock](benchmark-data/base-480-2-image-stock/results.json) | 29.518 [29.445, 29.592] | 28.694 | 121.96 | 0.06917 |
| [base-480-2-image / paiton](benchmark-data/base-480-2-image-paiton/results.json) | 29.758 [29.701, 29.816] | 28.506 | 120.97 | 0.06861 |
| [base-480-5-image / stock](benchmark-data/base-480-5-image-stock/results.json) | 79.648 [79.611, 79.686] | 76.900 | 45.20 | 0.06330 |
| [base-480-5-image / paiton](benchmark-data/base-480-5-image-paiton/results.json) | 79.757 [79.689, 79.826] | 76.534 | 45.14 | 0.06321 |
| [base-480-2 / stock](benchmark-data/base-480-2-stock/results.json) | 30.127 [30.042, 30.212] | 28.628 | 119.49 | 0.06777 |
| [base-480-2 / paiton](benchmark-data/base-480-2-paiton/results.json) | 29.783 [29.730, 29.836] | 28.416 | 120.87 | 0.06855 |

### Component wall times
Seconds, averaged over the same measured runs. Base calls contain both CFG branches in a batch of two; FastWan uses batch one. Latent/video batch size is one in both cases.

| Case / engine | Conditioning | Denoising | VAE decode | Encoding | Model calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast-480-2 / stock | 0.524 | 1.730 | 5.590 | 1.425 | 3 × B1 |
| fast-480-2 / paiton | 0.541 | 1.732 | 5.358 | 1.200 | 3 × B1 |
| fast-720-2-person / stock | 0.531 | 4.478 | 12.578 | 2.778 | 3 × B1 |
| fast-720-2-person / paiton | 0.533 | 4.490 | 12.034 | 2.729 | 3 × B1 |
| fast-720-5 / stock | 1.062 | 14.510 | 31.176 | 5.225 | 3 × B1 |
| fast-720-5 / paiton | 0.530 | 14.522 | 29.790 | 5.254 | 3 × B1 |
| fast-480-5 / stock | 0.554 | 4.797 | 13.879 | 2.788 | 3 × B1 |
| fast-480-5 / paiton | 0.543 | 4.809 | 13.333 | 2.488 | 3 × B1 |
| base-480-2-image / stock | 1.090 | 21.995 | 5.602 | 0.822 | 20 × B2 |
| base-480-2-image / paiton | 1.121 | 22.003 | 5.373 | 1.249 | 20 × B2 |
| base-480-5-image / stock | 1.093 | 61.915 | 13.885 | 2.746 | 20 × B2 |
| base-480-5-image / paiton | 1.091 | 62.120 | 13.316 | 3.221 | 20 × B2 |
| base-480-2 / stock | 1.075 | 21.937 | 5.608 | 1.497 | 20 × B2 |
| base-480-2 / paiton | 1.075 | 21.954 | 5.379 | 1.364 | 20 × B2 |

### Complete process memory peaks
GiB; peaks include startup and warmups. Driver figures are sampled; allocation/reservation come from PyTorch and omit memory managed outside its allocator. Process swap was zero in every primary case.

| Case / engine | GPU allocated | GPU reserved | Driver VRAM | Host RSS | Process swap |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast-480-2 / stock | 3.810 | 5.598 | 23.053 | 4.361 | 0.000 |
| fast-480-2 / paiton | 3.810 | 5.668 | 23.192 | 4.397 | 0.000 |
| fast-720-2-person / stock | 8.490 | 12.773 | 30.229 | 5.330 | 0.000 |
| fast-720-2-person / paiton | 8.490 | 12.615 | 30.377 | 5.276 | 0.000 |
| fast-720-5 / stock | 8.864 | 13.453 | 30.285 | 7.548 | 0.000 |
| fast-720-5 / paiton | 8.864 | 13.770 | 30.357 | 7.450 | 0.000 |
| fast-480-5 / stock | 3.978 | 5.455 | 22.911 | 5.495 | 0.000 |
| fast-480-5 / paiton | 3.976 | 6.098 | 23.622 | 5.413 | 0.000 |
| base-480-2-image / stock | 4.006 | 5.568 | 23.424 | 5.077 | 0.000 |
| base-480-2-image / paiton | 3.924 | 5.449 | 23.372 | 4.696 | 0.000 |
| base-480-5-image / stock | 4.171 | 6.566 | 24.422 | 5.762 | 0.000 |
| base-480-5-image / paiton | 4.090 | 6.221 | 24.142 | 5.674 | 0.000 |
| base-480-2 / stock | 4.006 | 5.930 | 23.533 | 4.755 | 0.000 |
| base-480-2 / paiton | 3.923 | 5.361 | 23.032 | 4.674 | 0.000 |

### Startup and first generation
First-generation seconds include lazy GPU loading and JIT compilation. The second warmup is also excluded from the table above. Downloads/builds are separate. The final Paiton artifact build took 9.26 seconds locally; users receive that compiled artifact.

| Case / engine | Runtime import s | Component setup s | First generation s | Second warmup s |
| --- | ---: | ---: | ---: | ---: |
| fast-480-2 / stock | 15.162 | 1.032 | 56.965 | 9.484 |
| fast-480-2 / paiton | 13.878 | 1.048 | 55.318 | 9.333 |
| fast-720-2-person / stock | 14.240 | 1.041 | 83.690 | 21.005 |
| fast-720-2-person / paiton | 14.013 | 1.037 | 65.507 | 20.275 |
| fast-720-5 / stock | 14.373 | 1.050 | 110.834 | 52.806 |
| fast-720-5 / paiton | 13.983 | 1.038 | 94.728 | 52.072 |
| fast-480-5 / stock | 14.325 | 1.040 | 81.836 | 22.795 |
| fast-480-5 / paiton | 13.910 | 1.037 | 66.919 | 21.527 |
| base-480-2-image / stock | 14.204 | 1.052 | 84.558 | 29.662 |
| base-480-2-image / paiton | 13.930 | 1.037 | 77.453 | 30.035 |
| base-480-5-image / stock | 14.150 | 1.033 | 141.719 | 79.879 |
| base-480-5-image / paiton | 13.938 | 1.039 | 126.383 | 79.698 |
| base-480-2 / stock | 14.175 | 1.038 | 88.860 | 30.304 |
| base-480-2 / paiton | 14.016 | 1.039 | 77.425 | 29.673 |

Maximum sampled hotspot temperature across these runs: 92 °C. Loaded clocks and power are retained in each telemetry file; the GPU profile was unchanged. The physical host had approximately 15 GiB usable RAM and 4 GiB swap, with pre-existing system swap usage distinct from generation-process swap.

## Matched comparison

One AMD Radeon AI PRO R9700, 32 GB, gfx1201, 64 CU / 32 WGP. Host CPU: Intel Core i5-8400, six cores/six threads; the terminal runner uses four PyTorch CPU threads. H.264 encoding runs on this CPU. The existing GPU profile was inspected and left unchanged (automatic clocks, compute profile). No power, clock, fan or voltage changes. Other generation workloads were absent during the batch; this task's idle ComfyUI instance was stopped for the terminal measurements. No container build or video postprocessing ran alongside primary measurements.

The two engines share model revisions, prompts, input image, seeds, geometry, frame rate, actual frame count, sampling schedules, adapters, batch size, attention backend, dtype, offload policy and compiler-cache behavior. Both use fresh conditioning. There is no cached-conditioning headline scenario.

- **FastWan:** three DMD evaluations, guidance 1, timesteps [1000,757,522], training table shift 8, BF16 CPU RNG in BTCHW order. No LoRAs.
- **Base:** 20 UniPC steps, simple schedule, shift 8, guidance 5; positive and negative conditioning freshly encoded. Actual evaluation counts and batch shapes are recorded, rather than equating scheduler steps with assumed model calls.
- **Both:** native scaled-FP8 UMT5 encoder; BF16 denoiser and VAE; CK INT8 attention; MIOpen enabled; block `torch.compile(fullgraph=False,dynamic=False)` with one compiler worker. Stock includes all these qualified runtime improvements.
- **Paiton difference:** compiled VAE normalization/SiLU only. Denoiser fusion experiments are not enabled in the release default. Distillation and quantization are not credited to the compiler.

Each setting starts a fresh process, retains two warmup clips, then measures two complete clips. Downloads occur separately. Runtime import and component setup are recorded separately from generation. The first generation includes lazy GPU loading and compilation together; its time is not presented as pure compilation time. Warm generation is reported separately. Persistent caches are reused by both engines; the two warmups allow residency and shape compilation to settle. The raw range matters when interpreting a small gain.

## Timing and memory definitions

`end_to_end_seconds` starts before fresh conditioning and ends after the H.264 container is closed. It includes VAE decoding, device-to-host pixel transfer, pixel conversion and file encoding (CRF 18, preset `fast`, 24 fps). It excludes process startup, downloads and post-measurement latent saving. The input image file is decoded once before the timed repeats; image VAE conditioning is recomputed inside every timed base image run. `pipeline_seconds` stops before file encoding. The CLI's one-shot `generate` command includes a cold first run; it should not be expected to match warm timings.

Clips/hour = 3600 / mean complete clip seconds. Generated-video-seconds/wall-second = actual encoded duration / mean complete clip seconds. These are video throughput measures, not text-token or cost metrics.

Component phases are synchronized wall times. Per-evaluation GPU events are retained separately. The gap between denoising wall time and summed model events includes scheduler work, transfers and host overhead; it must not be mislabeled as pure PCIe transfer time. Dedicated transfer profiling is a separate diagnostic and is excluded from headline measurements.

Memory records include PyTorch allocation/reservation peaks per phase, sampled driver VRAM, process peak RSS and sampled process swap, system available RAM/swap, loaded temperatures, clocks and power. Driver sampling is every 0.5 seconds, so its peak is a sampled lower bound. PyTorch allocation does not include all dynamic-loader allocations; driver VRAM is needed to understand the complete footprint. Persistent reservation/cache high-water marks are not live activation size.

The 5B hot denoiser is kept resident where the dynamic loader permits; the package does not deliberately stream every denoiser block for every evaluation. Text encoder and VAE residency are managed separately by the pinned loader. Memory includes the text encoder, denoiser, VAE, image conditioning when present, latents, activations, attention workspaces and compilation/cache pools. There is no vision encoder in this selected base TI2V image path; image conditioning uses the VAE. The package allocates no explicit full-pipeline graph pool because capture failed qualification; runtime cache/pool allocations remain included in the measured reservation and driver footprint. Untiled VAE is the qualified default.

## Reproduce

Copy [the attributed example image](assets/wan-official-example.jpg) to the data directory's `input/wan-official-example.jpg`, download both pinned presets, and run:

```bash
./models/Wan2.2/run.sh download --preset all
./models/Wan2.2/benchmark.sh
```

The fixed prompts/seeds are in [benchmark-cases.json](benchmark-cases.json). Exact resolved arguments and schedules accompany every case. [release.lock.json](release.lock.json) records runtime and artifact identity; [checkpoints.lock.json](checkpoints.lock.json) records source and model pins. H3 benchmark files and published measurements are unchanged.

See [EXPERIMENTS.md](EXPERIMENTS.md) for rejected optimizations, attention/quantization comparisons, VAE tiling, graph capture and numerical checks. A14B exploratory timings are a separate candidate comparison with a different 16-fps duration, not matched Paiton speedups.

For a separate transfer diagnostic, use `benchmark --runs 3 --warmups 2 --transfer-profile` with an explicit resolution/duration. This profiles each phase of the third run and saves memcpy event records. Profiling changes execution overhead: do not add those timings to the unprofiled benchmark table.


## Separate transfer diagnostic

A third warm 832×480×49 FastWan generation was profiled phase by phase with each engine. These profiled runs are excluded from all tables above. [Raw memcpy events and summary](benchmark-data/transfers/summary.json) are retained.

| Reported device copy activity | Stock ms | Paiton ms |
| --- | ---: | ---: |
| Conditioning D→H, one copy | 0.746 | 0.726 |
| Denoising H→D, four copies | 0.976 | 0.886 |
| Decode D→H, four copies | 9.819 | 9.874 |

The warm trace does not show repeated full denoiser-block transfers. These are profiler-reported copy activities, not a universal transfer bound for every preset or loader state. CPU `hipMemcpyWithStream` wall time is much larger—for example 5.119/4.958 seconds during stock/Paiton decode—because the call also waits for preceding GPU work. Calling that entire interval PCIe transfer time would misidentify the bottleneck. File encoding operates on the decoded CPU frames in this path.

In the separate A14B experiments, both experts stayed resident across the switch. Fresh conditioning plus encoder offload took about 2.06 seconds with dequantized BF16 and 2.42 seconds with W4A8 in the final exploratory runs; those combined phase times are not pure transfer measurements.
