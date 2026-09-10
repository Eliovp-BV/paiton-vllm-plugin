# Representation decision

The selected base is new in September 2026. Its quantized repository is a
September 7 derivative, not a separate base-model release. All comparisons retain
revision pins and preserve original files.

| Representation | Actual compatibility and evidence | Decision |
|---|---|---|
| BF16 original | Native Llama loader, ROCm attention and skinny GEMM execute on gfx1201; 5.034 GB weights; 15/20 direct-mode quality | Correct baseline. Compiled BF16 projections/fused gate did not improve full serving. |
| Same original, FP16 activation/weight casting | Native loader casts the original BF16 checkpoint; 15/20 quality; faster generation than tested BF16 | Strong stock reference, retains roughly 5 GB weights. |
| Official `MiniCPM5-2B-GPTQ` | Actual config is AWQ GEMM asymmetric W4A16, G128. Native vLLM AutoAWQ runs Triton GEMM for small token counts and full dequantization plus FP16 matmul at 256+ tokens. 2.100 GB weights, 14/20 quality | Selected after compiled decode qualification, not simply because it is INT4. |
| GPTQ/Marlin as a format name | The selected repository does not store that layout. AutoAWQ selects the ROCm Triton path, not CUDA Marlin | Do not relabel the actual runtime or claim native INT4 matrix arithmetic. |
| INT8, FP8, W4A8, MXFP4, Quark | No additional compatible pinned checkpoint was established for this exact selected model. Existing repository kernels or a format name alone do not establish loader, calibration and gfx1201 execution compatibility | Not downloaded/converted or qualified. No performance claim for these paths. |

The original W4 file contains 990,904,320 bytes of packed projection weights,
30,965,760 bytes of FP16 scales, 7,741,440 bytes of packed zero points,
534,773,760 bytes each for embeddings and output head, 348,160 bytes of other
unquantized tensors and a 108,680-byte safetensors header. These components explain
why a nominal 4-bit checkpoint is not simply total_parameters / 2 bytes.

Paiton losslessly transposes nibbles into a row layout, transposes FP16 scales and
expands zero-point metadata to uint8. Original tensors remain available for stock
prefill. The extra GPU layout is roughly 0.97 GiB and has no extra checkpoint disk
cost. Kernel unpacking/dequantization rounds weights to FP16 before FP32 FMA
accumulation. There is no startup calibration or quantization search.

Publisher metadata identifies GPTQModel 7.3.5 / GPTAQ and reencoding into AWQ
packing without AWQ scale search. It does not establish a calibration dataset or
exact BF16 source revision sufficient to reproduce the publisher's quantization.
Our reproducible operation begins at the pinned W4 inputs and is bit-exact layout
preparation; we do not claim to recreate the publisher's training or quantizer.

KV precision is independent: this release uses **FP16 KV**, not INT4 KV. Context
and KV capacity remain 8K / 1 GiB in stock and Paiton comparisons.

The BF16/FP16 comparison is between the two pinned public checkpoints. Because the publisher does not pin the quantizer's exact BF16 input revision, the one-task quality delta is an observed checkpoint difference, not a causally isolated measurement of quantization alone.
