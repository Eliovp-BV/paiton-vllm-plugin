# MiniCPM5-2B release notices

The checkpoint is OpenBMB's `MiniCPM5-2B-GPTQ` revision
`6c1ee6fa521aa53f47cfb32696e6d8ef5b0db805`, under Apache-2.0.
The name says GPTQ; its actual stored tensors use asymmetric AWQ GEMM packing,
4-bit weights, group size 128. We redistribute a pinned downloader, not weights
inside the container. The original checkpoint is unchanged. Runtime preparation
losslessly transposes packed nibbles, scales and zero points in GPU memory.
No calibration or additional quantization is performed. The publisher does not
establish an exact BF16 source revision or calibration dataset in this checkpoint's
metadata; we do not claim to reproduce their quantization procedure.

The checkpoint license is retained in `LICENSES/MiniCPM-Apache-2.0.txt`, from
OpenBMB/MiniCPM commit `f7988734fb95d43091dfda2d3c618ed2ebf0a9d0`.
Retain applicable copyright notices, license and modification attribution when
redistributing. This license permits commercial redistribution subject to its terms.

The plugin and retained vLLM/AITemplate components use Apache-2.0. ROCm,
Composable Kernel and other runtime components retain their respective notices
and licenses in the image and repository's top-level `LICENSES` and
`THIRD_PARTY_NOTICES.md`. The lazy FlashAttention import patch modifies the pinned
vLLM runtime; its original SPDX notice is retained. Paiton compiled artifacts
contain generated runtime code; private compiler source is not included.
