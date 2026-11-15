# Third-party notices

The container image holds the compiled Paiton libraries, the native runner, a Python server, ffmpeg and the pinned
Python and ROCm runtime packages listed below. It holds no model weights, no PyTorch, no Triton and no compiler source.

| Component | License and source |
|---|---|
| LTX-2.5 weights converted for Paiton (FP8 transformer, W8A8 text encoder, modulation tables) and the unchanged VAE, upsampler and audio VAE files | LTX-2.x Community License Agreement, copy in `LICENSES/LTX-2.x-Community-License.txt`; downloaded on first start from [EliovpAI/LTX-2.5-FP8-Paiton-RDNA4](https://huggingface.co/EliovpAI/LTX-2.5-FP8-Paiton-RDNA4), which lists the modified files in `MODIFICATIONS.md`; original files in [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) |
| Paiton server, client and scripts | Apache-2.0, `LICENSE` and `NOTICE` |
| LTX-2.5 native runner and libraries (`lib/ltx_run`, `lib/ltx25_*.so`) | Apache-2.0 for ElioVP's code, `LICENSE` and `NOTICE`. They implement the LTX-2.x architecture, which Section 1.5 (ii) of the LTX-2.x Community License Agreement counts as a Derivative, so that agreement applies to them too |
| ROCm 10.0 runtime and math libraries (`rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-device-gfx1201` wheels) | AMD license files inside the wheels (`/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core` and `_rocm_sdk_libraries`) |
| tokenizers 0.23.2 | Apache-2.0, license file in the package's `dist-info` |
| FFmpeg 6.1.1 (Ubuntu package `ffmpeg`, with libx264) | GPL-2.0-or-later as built by Ubuntu; the runner and the server call the `ffmpeg` and `ffprobe` programs and do not link to them. Source: `apt-get source ffmpeg` on Ubuntu 24.04; license files under `/usr/share/doc` |
| Ubuntu 24.04 base image | Package licenses under `/usr/share/doc` |

## LTX-2.x Community License

Read the full agreement before you download or use the weights. The points that most often matter:

- Revenue. An entity with annual revenue of US$10 million or more (counted together with its affiliates) needs a paid
  commercial license from Lightricks for any use other than the non-commercial uses in Section 2.2 (Section 2.1).
- Use restrictions. Attachment A and Lightricks' Acceptable Use Policy apply to every use, to the outputs and to
  derivatives. Anyone you let use the model must be bound by them (Sections 3.1 and 4).
- Machine-generated content. Attachment A item 5 asks you to state clearly that a generated video is machine
  generated when you publish it.
- Markings. Section 6 forbids removing watermarks, provenance data or other disclosure features that Lightricks puts
  into the model or its outputs. The Lightricks reference pipeline adds a watermark only when you pass it a marking
  pack (`--marker-root-path`); it is off by default and this server does not add one.
- Passing the weights or this image on. Include the full agreement, keep Lightricks' notices, mark modified files, and tell the
  recipient that the weights and the native runtime are derivatives of LTX-2.x under that agreement. You may not
  pass them to a company with US$10 million or more in revenue unless it already has a paid license from Lightricks (Section 3.5).
