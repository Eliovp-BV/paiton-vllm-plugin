# Idol with Paiton on Radeon

[Idol](https://huggingface.co/speridlabs/idol) turns one picture of an object into a 3D Gaussian splat (`.ply`).
This package runs the stock Idol pipeline with the Paiton native denoiser on one Radeon AI PRO R9700 (`gfx1201`).

<p align="center"><img src="assets/turntable-farmer.gif" width="640" alt="Input image and a 360-degree turntable of the 3D Gaussians generated from it"></p>

Left: the input image. Right: the 3D Gaussians this package generated from it on one R9700 (default settings, 262,144
Gaussians), rendered all the way around, including the back the picture never showed. The input image was generated
with [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Apache-2.0).
[Treasure chest](assets/turntable-treasure-chest.webp) and [lighthouse](assets/turntable-lighthouse.webp) examples.

| One R9700, 50 steps, cfg 3, 262,144 Gaussians | Request time |
| --- | ---: |
| Stock Idol (PyTorch, its default optimized mode) | 39.2 s |
| Paiton, exact (`--exact`) | 19.4 s |
| **Paiton, default** | **16.4 s** |

Same model, same settings as stock: 50 steps, guidance 3.0, no retraining and no change to the weights.

## Speed across GPUs

Request time = one image in, `.ply` written, 50 steps, 262,144 Gaussians. Every row except the R9700 rows is copied
from the [speridlabs/idol model card](https://huggingface.co/speridlabs/idol) ("Across GPUs": median over 8 images,
same code and settings on every GPU). R9700 rows: ours (release image), median of 8 images after one warm-up, see
[BENCHMARKS.md](BENCHMARKS.md).

| GPU | VRAM | Request time | Current price per GPU, USD (approximate) |
| --- | ---: | ---: | ---: |
| B300 | 269 GB | 3.4 s | n/a [1] |
| B200 | 179 GB | 3.6 s | n/a [1] |
| H200 (NVL) | 140 GB | 6.6 s | ~33,700 [2] |
| H100 | 80 GB | 6.7 s | ~32,400 [3] |
| RTX 5090 | 32 GB | 14.1 s | ~7,550 [4] |
| A100 | 80 GB | 16.3 s | ~17,800 [5] |
| **Radeon AI PRO R9700, Paiton** | 32 GB | **16.4 s** | ~2,000 [6] |
| L40S | 45 GB | 16.6 s | ~8,100 [7] |
| RTX 4090 | 24 GB | 19.1 s | ~4,800 [8] |
| Radeon AI PRO R9700, Paiton `--exact` | 32 GB | 19.4 s | ~2,000 [6] |
| Radeon AI PRO R9700, stock Idol | 32 GB | 39.2 s | ~2,000 [6] |
| RTX 3080 | 10 GB | 46.2 s | ~710 [9] |
| L4 | 22 GB | 54.6 s | ~3,580 [10] |

How the prices were picked: the lowest in-stock new price on a US price tracker or retailer listing in early November
2026, one card per row; no launch prices, used or refurbished cards. Prices are approximate and change daily.

## Performance per dollar

3D assets per hour per $1,000 of GPU = 3600 / request time / current price × 1000, one GPU, sorted best first.

| GPU | Request time | Current price used, USD | 3D assets per hour per $1,000 |
| --- | ---: | ---: | ---: |
| RTX 3080 | 46.2 s | 710 | 109.7 |
| **Radeon AI PRO R9700, Paiton** | 16.4 s | 2,000 | **109.5** |
| Radeon AI PRO R9700, Paiton `--exact` | 19.4 s | 2,000 | 92.5 |
| Radeon AI PRO R9700, stock Idol | 39.2 s | 2,000 | 45.9 |
| RTX 4090 | 19.1 s | 4,800 | 39.3 |
| RTX 5090 | 14.1 s | 7,550 | 33.8 |
| L40S | 16.6 s | 8,096 | 26.8 |
| L4 | 54.6 s | 3,584 | 18.4 |
| H100 | 6.7 s | 32,412 | 16.6 |
| H200 (NVL) | 6.6 s | 33,732 | 16.2 |
| A100 | 16.3 s | 17,789 | 12.4 |
| B300, B200 | 3.4 s, 3.6 s | n/a | n/a |

The RTX 3080 and the R9700 with Paiton are level per dollar; the R9700 finishes each asset 2.8× sooner and has 32 GB.

Sources (price, seller or tracker, date):

1. No public single-GPU listing found (B300 and B200 sell in 8-GPU systems), checked 6 Nov 2026.
2. 33,731.99, CDW listing, PNY H200 NVL (4-6 week lead time), 21 Sep 2026; ServerSupply lists 35,995.
3. 32,411.99, CDW listing, PNY H100 80 GB PCIe (marked discontinued), 21 Sep 2026; Newegg 32,879 on 2 Sep 2026.
4. 7,550, lowest in-stock new (Amazon, ASUS TUF OC), https://videocardprices.com/card/nvidia-rtx-5090/, 4 Nov 2026.
5. 17,789, Newegg listing, PNY A100 80 GB PCIe, https://www.newegg.com/p/N82E16814133879, Nov 2026.
6. 2,000, lowest in-stock new (B&H, ASUS Turbo), https://gpuprix.com/us/gpus/radeon-ai-pro-r9700, 4 Nov 2026.
7. 8,095.99, Newegg listing, PNY L40S, https://www.newegg.com/p/pl?d=l40s, Nov 2026.
8. 4,800, lowest in-stock new (Amazon, ASUS ROG Strix OC), https://gpuprix.com/us/gpus/geforce-rtx-4090, 4 Nov 2026.
9. 709.67, lowest in-stock new (Amazon, ASUS TUF RTX 3080 V2 OC), https://videocardprices.com/card/nvidia-rtx-3080/, 5 Nov 2026.
10. 3,583.99, CDW listing, PNY L4 24 GB, https://www.cdw.com/product/pny-nvidia-l4-graphic-card-24-gb-low-profile/7549280, Nov 2026.

## Model weights

Nothing is converted or re-uploaded. On the first run the package downloads the stock files, pinned by revision,
with your own Hugging Face account:

| Part | Repository | Revision | License |
| --- | --- | --- | --- |
| Denoiser, octree and Gaussian decoders, BiRefNet | `speridlabs/idol` | `2731c229` | Apache-2.0 |
| DINOv3 ViT-H+/16 image encoder | `timm/vit_huge_plus_patch16_dinov3.lvd1689m` | `dd0addc0` | DINOv3 License |
| FLUX.2 VAE (encoder) | `black-forest-labs/FLUX.2-klein-4B`, `vae/` only | `e7b7dc27` | Apache-2.0 |

DINOv3 is gated: accept its license on Hugging Face first, then pass your token. The timm copy is rearranged into
the layout idol-sdk loads (key names only, values unchanged). The klein-4B VAE is the same autoencoder Idol uses,
stored in bf16, which is the precision Idol runs it in. The native denoiser reads the stock fp32
`denoiser.safetensors` itself and keeps fp16 copies in GPU memory.

Download size: about 3 GB for Idol, 3.4 GB for DINOv3 and 0.2 GB for the VAE.

## Run

```bash
docker pull ghcr.io/eliovp/paiton-vllm-plugin:idol-rdna4-v1.0.0
```

```bash
export HF_TOKEN=hf_...            # account that accepted the DINOv3 License
mkdir -p "$HOME/paiton-idol/models" "$HOME/paiton-idol/outputs"
docker run --rm --device /dev/kfd --device /dev/dri --group-add "$(stat -c '%g' /dev/kfd)" \
  -e HF_TOKEN -v "$HOME/paiton-idol/models:/models" -v "$HOME/paiton-idol/outputs:/outputs" -v "$PWD:/inputs:ro" \
  ghcr.io/eliovp/paiton-vllm-plugin:idol-rdna4-v1.0.0 generate /inputs/photo.png --mode real
```

The first run downloads the weights into `$HOME/paiton-idol/models`; later runs start from there. Each image writes
`<name>.ply` (open it in any Gaussian splat viewer), `<name>_input.png` (the framed input) and `<name>_camera.json`
to `$HOME/paiton-idol/outputs`. Peak GPU memory: about 9.6 GiB.

| Option | Default | |
| --- | --- | --- |
| `--mode synthetic` / `real` | `synthetic` | renders and generated images / photos (wider crop) |
| `--exact` | off | fp16 attention on every step; output as stock fp16 |
| `--gaussians N` | 262,144 | 32,768 to 1,048,576 |
| `--seed N`, `--steps N`, `--cfg X` | 42, 50, 3.0 | as idol-sdk |

`generate` takes files or folders (`.png`, `.jpg`, `.jpeg`, `.webp`). `benchmark FOLDER` prints the median request
time after one warm-up. `download` only fetches the weights.

## What runs where

- **Denoiser** (50 guided steps, about 94 % of the request): `artifacts/libidol_denoiser_gfx1201.so`, native HIP/C++
  with hipBLASLt and rocBLAS. It does not use PyTorch or Triton.
- **Everything else** (background removal, DINOv3, VAE encoder, both decoders, `.ply` writing): the unmodified
  idol-sdk (`4702a7f`) on PyTorch, about 0.9 s per request.

The default runs int8 QK and fp8 PV attention from step 7 of 50, with fp16 matrix products on every step. Our render
check against stock compares 8 images from 8 views each. The default stays within twice the stock implementation-noise
LPIPS, and its worst image stays at or above 33 dB PSNR. `--exact` matches stock within that noise. The native
library picks its matrix-product kernels by timing them at load, so two processes can differ slightly (at the level
of stock's own optimized vs non-optimized difference); results are not bit-reproducible across runs.

## Licenses

Package code and the native runtime: Apache-2.0 (`LICENSE`). The proprietary compiler that produced the runtime is
not included. Model weights keep their upstream licenses (table above); see `NOTICE` and `THIRD_PARTY_NOTICES.md`.
