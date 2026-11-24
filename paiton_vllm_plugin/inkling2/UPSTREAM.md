# Upstream sources of this package and what changed

The files below come from vLLM (Apache-2.0) and keep their licence headers. Sources:

- vLLM main, commit `c4f6d8c7` (24 Nov 2026): `vllm/models/inkling2/`
- vLLM draft pull request 67685, head `800f72e` (`[ROCm][Model] Add Inkling 2 Small support on ROCm`):
  `vllm/models/inkling2/amd/`

Target: vLLM 0.32.0, which has the first Inkling generation only.

## Unchanged files

| file here | upstream file |
|---|---|
| `configs.py`, `processing.py` | main, same path |
| `amd/__init__.py`, `amd/dspark.py`, `amd/moe.py`, `amd/vision.py` | PR, same path |
| `amd/ops/{__init__,attention,dspark,linear,mxfp8,norm,triton_bmm,vision}.py` | PR, same path |

## Changed files

| file here | upstream file | change | reason |
|---|---|---|---|
| `quant_config.py` | main | `from .configs import` instead of `from vllm.models.inkling2.configs import` | the package lives in the plugin |
| `amd/ops/sconv.py` | PR | `from .norm import` instead of `from vllm.models.inkling2.amd.ops.norm import` | same |
| `amd/ops/moe.py` | PR | `combine` sums in blocks of 256 columns when the hidden size is not a multiple of 512 | Mini's hidden size is 2304; the PR was written for Small |
| `amd/ops/activation.py` | PR | `swiglu_quantize` uses a program per 128 columns when the intermediate size is not a multiple of 256 | Mini's intermediate size is 1152 |
| `amd/ops/loop.py` | main `nvidia/ops/loop.py` | `tl.rsqrt` instead of the CUDA inline-assembly `rsqrt` of `nvidia/ops/norm.py` | the two controller kernels are plain Triton apart from that |
| `amd/ops/attn_res.py` | PR | `APPLY_NORM` option: fp32 mixes when no norm weight is given | the loop blends messages into unnormalized mixes; this is the option of main's `nvidia/ops/attn_res.py`, which the PR left out |
| `amd/attention.py` | PR | `get_kv_cache_spec` returning `SlidingWindowSpec` / `FullAttentionSpec` instead of `get_kv_cache_config` / `KVCacheLayerConfig` | vLLM 0.32 API |
| | | `Inkling2AttentionPassCache`, `pass_conv_states`, `pass_idx`: one KV cache and conv state per loop pass | looped model (from main's `nvidia/attention.py`) |
| | | local layers accept separate K and V inputs | loop passes after the first |
| | | `process_weights_after_loading(self, act_dtype=None)` | 0.32 passes the model dtype to attention hooks |
| `amd/sconv.py` | PR | `owner` argument of `Inkling2ShortConv.forward` | conv state per loop pass |
| `amd/model.py` | PR | `num_passes` and per-pass conv states in `Inkling2DecoderLayer`; `_make_layer`; `_log_scaling_tau`; `Inkling2LoopModel` for looped checkpoints; the "not supported on ROCm" error for looped checkpoints is gone | looped model |
| | | NVFP4 layers use `Inkling2BringupMoE` instead of raising; `PAITON_INK2_EXPERTS` | bring-up, see `amd/experts_bringup.py` |
| | | model-level `process_weights_after_loading` that runs the package's other post-load hooks | 0.32 runs only this hook and the attention layers' (main runs every module's) |
| | | `PAITON_INK2_TAP_DIR` tap in `forward` | bring-up check against the reference |

The gfx950 check of the PR (`Inkling2 on ROCm requires gfx950`) is unchanged: every Triton kernel in `amd/ops`
was written for that architecture.

## New files (not from upstream)

- `amd/loop.py`: port of main's `nvidia/loop.py` to the AMD operators. Each sublayer's short convolution and the
  next AttnRes mix run as two kernels (the AMD path has no fused `conv_mix`); the controller GEMMs are `torch.bmm`.
- `amd/experts_bringup.py`: PyTorch experts on dequantized weights with the trained input quantization. Bring-up
  only, not a serving path.
- `__init__.py`: registration of the architectures, config classes and the `inkling2` quantization method.

## Not ported yet

`speculator.py`, `speculator_config.py` (DSpark drafter config), `nvidia/lora.py`, and the differences of main's
`vllm/renderers/inkling*.py` and `vllm/parser/inkling.py` against 0.32.
