# Candidate and quantization assessment

This release targets one Radeon AI PRO R9700, 32 GB, gfx1201. Selection combines actual local clips, complete generation time, memory and a usable image-input path. NVIDIA throughput is not evidence for this GPU.

## Selected settings

**FastWan FullAttn 5B** supplies fast general text-to-video. **Wan2.2 TI2V-5B** supplies general text/image-to-video. Neither generates audio. Their checkpoints and samplers differ; distillation gains are not Paiton gains. Exact download and source revisions are in [checkpoints.lock.json](checkpoints.lock.json).

The dense 5B transformer has 30 blocks, width 3072, 24 heads of 128, FFN 14336 and 48 latent channels. Its VAE compresses time by 4 and space by 16; transformer patches are 1×2×2. The original [Wan2.2 release](https://github.com/Wan-Video/Wan2.2) was July 28, 2025. Native defaults use 50 steps, shift 5 and guidance 5. This package's base setting uses the separately qualified Comfy configuration: 20 UniPC steps, simple schedule, shift 8, guidance 5.

[FastWan FullAttn](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers) is an Apache-2.0 DMD distilled model. Its repository was created August 2, 2025; the series was announced August 4. It uses full attention, without a sparse attention requirement. The package qualifies text input only despite TI2V in the checkpoint name. The pinned [FastVideo implementation](https://github.com/hao-ai-lab/FastVideo/tree/a943220c115228ade5d57b3bab9a6a87fd600a10) defines three DMD evaluations at [1000, 757, 522], a 1000-point training table with shift 8, FP64 clean-prediction intermediates, and BF16 CPU noise in BTCHW order. This is not a three-step Euler sampler. Local reference checks matched clean prediction, re-noising and final RNG state exactly on four fixtures.

## Runtime choice

ComfyUI is the qualified runtime for the selected 5B paths: its actual Wan image-latent node, scaled-FP8 text loader, AMD CK attention and bounded-memory loader all ran on the R9700. Native Wan and FastVideo supplied architecture/sampling references; the DMD port was checked against the pinned FastVideo functions. Diffusers was used for the A14B SDNQ experiments. The text-serving vLLM plugin is the community repository, not the video runtime. This is the strongest qualified stock configuration in these experiments, not a claim that every possible runtime has been benchmarked.

## Alternatives investigated

| Candidate | Actual configuration and local evidence | Decision |
| --- | --- | --- |
| Base TI2V-5B | BF16 execution of the pinned FP16 repack; native scaled-FP8 UMT5 encoder; supports optional image input | Retain for image input and base text generation |
| FastWan FullAttn 5B | BF16 transformer, three DMD evaluations; useful walking/head-turn output | Default text preview; separately qualified longer/720-class settings |
| T2V-A14B SDNQ UINT4/SVD32 + Seko V1.1 | Both experts resident, separate unmerged rank-64 adapters; four Euler evaluations; 832×480×49 at 16 fps | Better natural color/head-turn in the fox example, but ~59–60 s warm with dequantized BF16 and ~53.6–54.0 s with W4A8; not the practical default |
| I2V-A14B quantized | Separate image model, not interchangeable with T2V expert weights | Architecture/loader assessed; no additional large download after 5B image path succeeded |
| GGUF 5B | Loader dequantizes stored blocks; format alone does not establish native gfx1201 low-bit GEMM | No second-format download without a demonstrated speed benefit |
| LightX2V current acceleration paths | ROCm AITER INT8 CK paths and newer NVFP4/sparse paths have different hardware requirements | No inferred R9700 support from NVIDIA or gfx950 results |
| 5B Turbo | Source license CC BY-NC-SA-4.0, unlike the Apache base | Not the unrestricted community default |
| Wan S2V / Animate | Speech-driven video / specialized character animation and replacement | Different tasks; not general TI2V substitutes |

Each A14B expert has 40 blocks, width 5120, 40 heads of 128 and FFN width 13824. It uses 16 latent channels and a VAE with temporal compression 4 and spatial compression 8. At the same canvas/frame count its transformer sees four times as many spatial tokens as the 5B architecture, so active parameter count alone is a poor latency predictor.

A14B means approximately 14B active parameters **per denoising evaluation**, not 14B total checkpoint parameters. There are two approximately 14B experts, high-noise and low-noise, about 28B total. The tested SDNQ files are 8.305 GB per expert, plus 8.147 GB text encoder, 0.508 GB VAE and two 1.227 GB adapters. The tested Seko schedule has sigmas [1, .9375, .8333333, .625, 0], boundary .9, guidance 1, two high then two low calls. Both experts stay resident; fresh text conditioning includes encoder transfers. Peak sampled driver VRAM was 25.07 GiB, host RSS 10.18 GiB and process swap 0.50 GiB. Its 49-frame clip at 16 fps is 3.0625 seconds, not the same duration as 5B's 49 frames at 24 fps.

Original A14B defaults differ: T2V uses shift 12, boundary .875 and low/high guidance 3/4; I2V uses shift 5, boundary .9 and guidance 3.5. Do not replace these with the distilled schedule unless the corresponding adapters are installed.

## Quantization is a runtime property

- The tested SDNQ UINT4/SVD32 loader normally dequantizes to BF16. Its optional quantized matmul unpacks to INT8 and quantizes activations, with additional buffers/metadata and first-run compilation. The W4A8 complete-clip improvement above belongs to this runtime setting, not Paiton. Both experts and their LoRAs were exercised across the switch.
- Native FP8 computation works on this R9700. The shared scaled-FP8 text encoder is useful. An on-load FP8 denoiser experiment did not beat BF16: warm denoising ~1.75 s versus ~1.73 s for compiled BF16 at 832×480×49. Its fox clip stayed coherent but differed substantially (encoded PSNR 18.89 dB). Keep BF16; this single clip does not establish general FP8 quality loss.
- Generic INT8 projection probes incurred activation quantization overhead. Neither an INT8 format label nor smaller stored weights guarantees lower complete-generation latency.
- TorchAO's installed extension compatibility warnings and absence of a qualified Wan/gfx1201 path prevented promotion. Quark was investigated for loader/kernel/calibration support; no supported end-to-end configuration was established. No speculative conversion was performed.
- FastWan conversion only renames safetensors keys. All 825 tensor payloads are preserved, with no calibration and no excluded tensors. Originals and conversion hashes remain in the persistent cache. Base FP16 weights are cast to BF16 during loading; this is not the original FP32 Diffusers checkpoint.

## Pinned alternative sources

- [Wan-AI model organization](https://huggingface.co/Wan-AI); native TI2V revision `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`, T2V-A14B `c8c270b13ee05bfa474194ac9fb07a5868a97cea`, I2V-A14B `206a9ee1b7bfaaf8f7e4d81335650533490646a3`.
- [SDNQ A14B](https://huggingface.co/Disty0/Wan2.2-T2V-A14B-SDNQ-uint4-svd-r32/tree/0c380902f2694211e02d21743e2e91010a4a9515), Apache-2.0; [Lightning/Seko adapters](https://huggingface.co/lightx2v/Wan2.2-Lightning/tree/18bccf8884ec0a078eed79785eb4ef13ea16ce1e).
- [LightX2V](https://github.com/ModelTC/LightX2V/tree/95e9b86bff996954420133d5acfd53cd3b7c00cc); DistillModels revision `db93455b9e85c4d8a3ff9297fcfa189d213cfe29`; AITER `a7d3bf8cd47afbaf6a6133c1f12e3b01d2c27b0e`.
- QuantStack GGUF 5B revision `57437632ddd08bdcbd1508c866aa22e126ed51d2`; quanhaol 5B Turbo model `c7cbd600f8ffa4a59bee2a3e742bdf58277a926c`, code `77768551236ad110d68299cf1a7151c0c728bddb` (August 6, 2025).
- Wan S2V was released August 26, 2025 and consumes externally supplied audio; optional separate CosyVoice TTS does not make it joint audio/video generation. Animate was released September 19, 2025 for specialized animation/replacement.

See [third-party notices](THIRD_PARTY_NOTICES.md) for redistribution boundaries. Checkpoints download from their pinned publishers; no model weights are embedded in the artifact image.
