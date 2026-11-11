# Third-party notices

The container image holds compiled Paiton libraries, the native runner, four numpy weight converters, a Python server
and the pinned Python and ROCm runtime packages listed below. It holds no model weights, no PyTorch and no Triton.

| Component | License and source |
|---|---|
| MiniMax-H3-Image weights, tokenizer and configs | MiniMax-H3-Image Community License Agreement (10 Nov 2026), copy in `LICENSES/MiniMax-H3-Image-Community-License.txt`; downloaded from [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/tree/00dd081ebdd33315adf2c4d84f64aa327ddf3039) on first start |
| Paiton libraries, runner, converters, server and scripts | Apache-2.0, `LICENSE` and `NOTICE` |
| ROCm 10.0 runtime (`rocm-sdk-core` wheel) | AMD license files inside the wheel (`/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core`) |
| numpy 2.4.2, Pillow 12.1.0, tokenizers 0.23.2 | numpy: BSD-3-Clause with 0BSD, MIT, Zlib and CC0-1.0 parts; Pillow: MIT-CMU; tokenizers: Apache-2.0; license files in each package's `dist-info` |
| Ubuntu 24.04 base image | Package licenses under `/usr/share/doc` |

## MiniMax-H3-Image Community License

Read the full agreement before you download or use the weights. The points that most often matter:

- Territory. The grant covers the world except the European Union, the United Kingdom, the Republic of Korea and the
  United States (Sections I.3 and I.5). People in those regions need a separate license from MiniMax.
- Attribution. A commercial product or service that uses MiniMax-H3-Image must show "MiniMax-H3-Image" prominently in
  its user interface (Section IV.2).
- Commercial scale and hosting. You need prior written authorization from MiniMax if you and your affiliates have more
  than US$1 million in annualized revenue, or if you offer the model to third parties as a hosted service (Section
  IV.1). Running this server for other people over a network counts as a hosted service.
- Users. Anyone you give access to must be bound by terms at least as protective as Section V and the Acceptable Use
  Policy in Exhibit A (Section V.2).
- Distribution. If you pass on the weights or files converted from them, include the agreement and the NOTICE text, and
  mark modified files (Section III).

The converted bundle that the container writes to `/cache/native-bundle` is a modified copy of the weights: FP8
transformer and text encoder projections, an int8 VAE decoder and precomputed AdaLN tables. This package does not
redistribute it.
