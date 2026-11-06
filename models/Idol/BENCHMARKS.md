# Idol benchmark record

One Radeon AI PRO R9700 (`gfx1201`, 32 GB) at its stock 300 W limit, no clock, power or fan changes. One image to
262,144 Gaussians, 50 steps, cfg 3.0, rescale_t 3.0. Request time = image in to `.ply` written (idol-sdk's
`IdolPipeline.__call__` plus `save_ply`), median of 8 images after one warm-up request. Speed across GPUs, prices and
performance per dollar: [README.md](README.md#speed-across-gpus).

## Setup

Stock Idol: idol-sdk `4702a7f` with its default optimized sampling (compiled denoiser, guided and unguided passes in
one batch), PyTorch 2.15 nightly on ROCm 10.0. cuDNN attention does not exist on ROCm, so attention uses PyTorch's
default fused attention. Paiton: the release image, the same pipeline with the denoiser replaced by the native library.

Images: 8 objects (figures, food, furniture, toys) matching the gallery of the Idol model card, `--mode synthetic`,
one fixed seed per image. Request time hardly depends on the image (spread under 0.1 s).

| One R9700, median of 8 | Stock | Paiton `--exact` | Paiton default |
| --- | ---: | ---: | ---: |
| Request | 39.22 s | 19.45 s | 16.44 s |
| Denoiser, 50 steps | 38.28 s | 18.51 s | 15.41 s |
| Everything else (background removal, DINOv3, VAE, decoders, `.ply`) | 0.94 s | 0.93 s | 0.93 s |
| Peak GPU memory reserved by PyTorch | 10.7 GiB | 9.6 GiB | 9.6 GiB |

Request times for Paiton come from the release image. The denoiser / rest split comes from the same code with
per-stage timers (harness runs).

Per-step denoiser time: stock 766 ms, Paiton `--exact` 370 ms, Paiton default 298 ms on steps 7-49 (attention in int8
QK + fp8 PV; matrix products fp16 throughout).

## Quality

Each `.ply` is rendered at the predicted input camera and 7 new views (512 × 512), and compared with stock Idol's
render of the same image and seed: PSNR, SSIM, LPIPS (AlexNet), alpha IoU, 8 images × 8 views. The noise floor is
stock Idol in its non-optimized mode (same model, other kernels) against stock optimized.

| Against stock optimized | PSNR | LPIPS | SSIM | Alpha IoU | Worst image PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock, non-optimized (noise floor) | 39.54 | 0.0026 | 0.9739 | 0.9984 | 37.06 |
| Paiton `--exact` (release image) | 39.54 | 0.0027 | 0.9740 | 0.9984 | 37.09 |
| Paiton default (release image) | 37.97 | 0.0039 | 0.9713 | 0.9955 | 34.63 |

Release rule for the default: mean LPIPS at most twice the noise floor (0.0052) and worst image PSNR at least 33 dB.

MIOpen chooses convolution kernels for background removal, DINOv3 and the VAE by timing them on first use. On a
fresh cache, a different choice changes the denoiser's input slightly (here 0.2 % of input pixels by up to 3 of 255
levels); over 50 steps that alone moved the exact path from LPIPS 0.0027 to 0.0049 against stock in our test, more than the int8/fp8 attention does. The package therefore
seeds the MIOpen cache with the database the release was checked with, so a new install picks the same kernels; the
table above was measured from an empty cache.
