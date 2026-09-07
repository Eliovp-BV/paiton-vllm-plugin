# FLUX.2 klein 4B benchmark record

On September 7, 2026, one Radeon AI PRO R9700 (`gfx1201`, 32 GB) averaged
**1.053758 seconds per image with Paiton** and **1.257996 seconds with the
strongest qualified stock configuration**. This is **16.235% lower latency**
and **19.382% more potential images per hour**. The 20% latency objective and
sub-second complete generation were not achieved.

| Metric | Stock | Paiton |
| --- | ---: | ---: |
| Mean seconds per image | 1.257996 | 1.053758 |
| Projected images per hour | 2,861.7 | 3,416.3 |
| Text encoder, seconds | 0.03765 | 0.03700 |
| Four transformer steps, seconds | 0.98283 | 0.88192 |
| VAE decode, seconds | 0.20897 | 0.12164 |
| Peak Torch allocation, GiB | 19.33 | 12.87 |
| Peak Torch reservation, GiB | 22.10 | 14.07 |
| Maximum sampled driver VRAM, GiB | 23.04 | 14.58 |

The headline uses wall-clock prompt-to-PIL time, including text encoding,
denoising, VAE decoding and image conversion. Component intervals use GPU events
and are diagnostic. PNG encoding/writing, UI transport, download, model loading
and initial compilation are outside warm generation time. Images per hour is
`3600 / mean seconds`, not an hour-long throughput test.

## Protocol

Both paths use the same pinned SDNQ checkpoint revision
`45e9cc76cb70f84473ce5c6c2e2282d0ef3c6ecd`, equivalent prepared INT8 transformer
weights, expanded BF16 text-encoder values and BF16 VAE. Transformer attention
uses INT8 QK, BF16 PV and FP32 softmax. Decoder attention remains BF16. Both use
1024 × 1024, four steps, guidance 1.0, batch one, 512 text tokens, the checkpoint's
FlowMatchEuler scheduler and full GPU residency.

The fixed cases are a wildlife fox photograph (seed 42), teal cup/lemon product
photography (31415), and a watercolor bookshop with a red-raincoat cyclist (2026).
There are two warmups and two measured runs per case, six measured generations
per backend. The harness retains complete prompts and all settings. GPU runs
are sequential. The host has 16 GB RAM. AUTO with COMPUTE was unchanged; no
clock, power, voltage or fan limits were modified. Sampled loaded junction
temperatures were 58–82°C for stock and 63–77°C for Paiton. Automatic clocks vary,
and the small sample does not establish a universal speed guarantee.

Stock qualification covered native and SDNQ attention, quantized and expanded
matrix execution, graph capture, whole-module compilation, immutable prepared
weights, weight freezing, channels-last decoding and vendor convolution tuning.
The selected stock configuration compiles transformer, text encoder and VAE,
uses the prepared-weight helper, and leaves weight freezing and convolution
autotuning disabled. The text-encoder output elimination is offered identically
to stock and Paiton. Pipeline and dependency versions are retained in the data.

## Quality

All retained final latents are finite. Within each benchmark process, the two
measured PNGs match exactly for every fixed prompt and seed. Stock and Paiton
images differ. Separately compiled processes can also differ; neither seed
identity nor one perceptual score proves equivalent quality.

| Prompt | RGB SSIM | PSNR, dB | Latent relative RMSE |
| --- | ---: | ---: | ---: |
| Fox | 0.9752 | 32.45 | 0.1456 |
| Product | 0.9739 | 30.10 | 0.2425 |
| Bookshop | 0.8594 | 21.38 | 0.2894 |

Visual inspection retains the fox's pose, woodland lighting and fur; the teal
cup, whole lemon and readable PAITON card; and the watercolor style, wet street,
bookshop and person with a bicycle. Fur, reflections, shelves, masonry and
lettering differ. The cyclist stands beside the bicycle in both outputs.
This three-prompt set does not establish quality across all subjects or styles.

## ComfyUI workflow check

Two measured Paiton workflow runs after two warmups took 1.408 and 1.404 seconds
from ComfyUI execution start to successful SaveImage completion. This includes
local engine transport, image handling and saving, and excludes browser display.
The 1.406-second mean is separate from the prompt-to-PIL comparison. Engine
switching, offline restart and browser reload were checked; only the selected
worker remained loaded after each image. See the [UI validation record](benchmark-data/comfyui-validation.json).

## Reproduce

Build or pull the release images and prepare the model using the README. Stop
other generation services, then run:

```bash
./launch.sh --stop
./run.sh benchmark --backend stock --suite --output /outputs/stock
./run.sh benchmark --backend paiton --suite --output /outputs/paiton
```

[Every measured timing](benchmark-data/raw-timings.csv) and
[settings, memory, versions and quality](benchmark-data/results.json) are included.
The [article](https://eliovp.com/blog/paiton-flux2-klein-radeon-ai-pro-r9700)
provides charts, full-resolution quality pairs, example media, startup details
and the complete evidence archive.
