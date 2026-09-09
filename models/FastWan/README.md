# FastWan FullAttn 5B on Radeon AI PRO R9700

Generate silent text-to-video locally in ComfyUI or the terminal. This three-evaluation distilled Wan2.2 setting defaults to Paiton, with a separate stock option. Image input is available through [Wan2.2 TI2V-5B](../Wan2.2/README.md).

[![Five-second FastWan example](../Wan2.2/assets/fast-720-5-paiton.jpg)](https://huggingface.co/EliovpAI/Wan2.2-FastWan-5B-Paiton-RDNA4/resolve/main/models/Wan2.2/assets/fast-720-5-paiton.mp4)

[Play the example](https://huggingface.co/EliovpAI/Wan2.2-FastWan-5B-Paiton-RDNA4/resolve/main/models/Wan2.2/assets/fast-720-5-paiton.mp4) · [Hugging Face runtime package](https://huggingface.co/EliovpAI/Wan2.2-FastWan-5B-Paiton-RDNA4)

## Quick start

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git && cd paiton-vllm-plugin && ./models/FastWan/launch.sh
```

Open [ComfyUI](http://127.0.0.1:8192/?paiton=1&preset=fast), edit the prompt and click **Run**. Choose stock/Paiton, 480/720 class and two/five seconds in the connected workflow. The default is Paiton, 720 class, two seconds. ComfyUI listens on **0.0.0.0:8192**; replace localhost with your server address for remote access.

You need Linux, Docker with Compose, AMD GPU device access and one **32 GB gfx1201 R9700**. The first launch builds the pinned runtime and downloads this preset; later launches reuse model and runtime caches. Model storage is about **28.15 GB**, including the preserved original and lossless converted checkpoint. Allow 100 GB free disk for containers, caches and outputs. Qualification used about 15 GiB usable host RAM and 4 GiB swap; 32 GB RAM gives more headroom.

Both Wan launchers share one service and cache. Run one generation at a time. Launch the [base model](../Wan2.2/README.md) to add image input. Outputs go to `~/paiton-videos`; cache and port controls are in the [shared runtime guide](../Wan2.2/README.md#requirements-and-persistent-storage).

```bash
./models/FastWan/launch.sh --logs
./models/FastWan/launch.sh --stop
./models/Wan2.2/run.sh generate --preset fast --engine paiton --resolution 480 --duration 2 --seed 1201 --prompt 'A red fox walks through a sunlit woodland clearing and turns toward the camera.'
```

## Measured performance

| Setting | Stock clip s | Paiton clip s | Lower latency |
| --- | ---: | ---: | ---: |
| 832×480, 49 frames | 9.280 | 8.843 | 4.71% |
| 1280×704, 49 frames, person | 20.376 | 19.797 | 2.84% |
| 1280×704, 121 frames | 51.986 | 50.109 | 3.61% |

Complete times include fresh conditioning and file encoding; each mean uses two measured runs after two warmups. Both engines use the same three DMD evaluations, adapters, schedule and runtime settings. Distillation gains are not compiler gains. Paiton fuses VAE normalization and SiLU; paired denoised latents were bitwise equal, with small decoded pixel differences. See [full benchmarks](../Wan2.2/BENCHMARKS.md), [quality limitations and paired clips](../Wan2.2/QUALITY.md) and [reproduction commands](../Wan2.2/README.md#terminal-generation-and-fair-benchmarks).

At 24 fps, 49/121 frames encode as 2.0417/5.0417 seconds. “720 class” means 1280×704. Clips are native continuous generation. This setting is qualified for text input only and generates no audio.

## Models and licenses

The pinned Apache-2.0 [FastWan FullAttn checkpoint](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers/tree/3e187042a324f6f5fb68fd22110a78725253de8f) is downloaded from its original publisher. The shared [checkpoint lock](../Wan2.2/checkpoints.lock.json), [license notices](../Wan2.2/THIRD_PARTY_NOTICES.md) and [runtime recipes](../Wan2.2/README.md#container-and-release-layout) cover reproducibility and redistribution. The Hugging Face package contains runtime code, compiled artifacts and evidence, not model weights or private compiler source.
