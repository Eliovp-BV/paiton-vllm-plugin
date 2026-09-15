# Attribution and component terms

This distribution combines native Paiton runtime artifacts with techniques
adapted from the projects below. Attribution remains applicable to the adapted
implementation; changes to interfaces or integration do not erase that origin.
No blanket Apache-2.0, MIT, or other license is asserted for all artifacts in
this repository.

| Component | Upstream project and revision | Relationship |
|---|---|---|
| Packed matrix techniques | [magiccodingman/vllm-radiance](https://github.com/magiccodingman/vllm-radiance/tree/e9c80e21c7b1c16993677f6a514156ddbd392a3b), `e9c80e21c7b1c16993677f6a514156ddbd392a3b` | Adapted native HIP implementation and Paiton runtime integration |
| Attention, recurrent prefill, and convolution techniques | [StillDeadcode/libr4d](https://codeberg.org/StillDeadcode/libr4d/src/commit/e8de4bc1f3dbd608dcb8d3ffceb6b48acdf83bb7), `e8de4bc1f3dbd608dcb8d3ffceb6b48acdf83bb7` | Adapted native HIP implementation and Paiton runtime integration |
| Serving framework and DFlash2 scheduling | [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/2cf0a6915ce544dc493a0990f2ea38d81601128a), `2cf0a6915ce544dc493a0990f2ea38d81601128a` | External dependency supplied by the qualified official vLLM image |
| Target checkpoint | [AMD Qwen3.8-27B-Quark-AWQ-MXFP4](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4/tree/5233554c5fa56afda40150556b95573c2d7d29c0) | Downloaded unchanged from its original repository; weights are not redistributed here |
| DFlash2 draft checkpoint | [tcclaviger/Qwen3.8-27B-DFlash2-FP8](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8/tree/ee0cb26a8279b7910cc28d82a8a3e15e4728d56f) | Downloaded unchanged from its original repository; weights are not redistributed here |
| Benchmark corpus | [GGZ14/BetterBench](https://github.com/GGZ14/BetterBench/tree/575cc3925bac922d6ad4a39e62502673799979d9), `575cc3925bac922d6ad4a39e62502673799979d9` | Benchmark methodology and corpus; no serving dependency |

The maintainer reports direct author permission for the Radiance/libr4d
adaptation. This statement records the permission reported to the project; it
does not create a new general license or replace an upstream author's terms.
Upstream model and framework terms remain applicable to separately downloaded
components. The proprietary compiler and generated implementation source are
not included in this runtime distribution.
