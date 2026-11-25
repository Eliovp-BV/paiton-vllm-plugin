# Upstream sources of this package and what changed

The files below come from vLLM (Apache-2.0) and keep their licence headers. Sources:

- vLLM main, commit `c4f6d8c7` (24 Nov 2026): `vllm/models/inkling2/`, `vllm/parser/inkling.py`
- vLLM draft pull request 67685, head `800f72e` (`[ROCm][Model] Add Inkling 2 Small support on ROCm`):
  `vllm/models/inkling2/amd/`

Target: vLLM 0.32.0, which has the first Inkling generation only.

## Unchanged files

| file here | upstream file |
|---|---|
| `configs.py`, `processing.py` | main, same path |
| `amd/__init__.py` | PR, same path |

## Changed files

| file here | upstream file | change | reason |
|---|---|---|---|
| `quant_config.py` | main | `from .configs import` instead of `from vllm.models.inkling2.configs import` | the package lives in the plugin |
| `amd/attention.py` | PR | `get_kv_cache_spec` returning `SlidingWindowSpec` / `FullAttentionSpec` instead of `get_kv_cache_config` / `KVCacheLayerConfig` | vLLM 0.32 API |
| | | `Inkling2AttentionPassCache`, `pass_conv_states`, `pass_idx`: one KV cache and conv state per loop pass | looped model (from main's `nvidia/attention.py`) |
| | | local layers accept separate K and V inputs | loop passes after the first |
| | | `process_weights_after_loading(self, act_dtype=None)` | 0.32 passes the model dtype to attention hooks |
| `amd/sconv.py` | PR | `owner` argument of `Inkling2ShortConv.forward` | conv state per loop pass |
| `amd/model.py` | PR | `num_passes` and per-pass conv states in `Inkling2DecoderLayer`; `_make_layer`; `_log_scaling_tau`; `Inkling2LoopModel` for looped checkpoints; the "not supported on ROCm" error for looped checkpoints is gone | looped model |
| | | model-level `process_weights_after_loading` that runs the package's other post-load hooks | 0.32 runs only this hook and the attention layers' (main runs every module's) |
| | | `PAITON_INK2_TAP_DIR` tap in `forward` | bring-up check against the reference |
| | | `ops_dump.record` around `embed_input_ids` and `compute_logits` | per-operator fixtures for the native operators (`amd/ops_dump.py`), off unless `PAITON_INK2_OPS_DUMP` is set |
| `parser.py` | main `vllm/parser/inkling.py` | only `_inkling2_is_reasoning_end` is taken, unchanged, as `is_reasoning_end` of a subclass of 0.32's `InklingParser` | 0.32's parser has the same block grammar as main's (the config is identical), so the rest is reused. Its rule for "has reasoning ended" reads the text block of the last user message of an Inkling 2 prompt as the end of reasoning |
| | | the adapters and `Inkling2EngineToolParser` are built as in main's `registered_adapters.py` and `tool_parsers/inkling_tool_parser.py` | registered as `inkling2` through `register_lazy_module` |
| | | `adjust_request` is left out | it frames JSON response formats with xgrammar's `TokenFormat`, which no 0.32 code uses (0.32 pins xgrammar 0.2.8), and with `_parser_structured_outputs`, which 0.32's Responses request does not have |

## Removed files

The pull request's Triton path is not in this package: `amd/ops/` (all its kernels), `amd/moe.py`, `amd/dspark.py`
and `amd/vision.py`, and the bring-up layer `Inkling2BringupMoE` that used `amd/moe.py`. On the R9700 every operator
runs in Paiton's native libraries (`amd/native_*.py`, `amd/experts_native.py`, `amd/draft.py`), and
`PAITON_INK2_OPS=native` is required.

## New files (not from upstream)

- `amd/loop.py`: port of main's `nvidia/loop.py` to the AMD operators. Each sublayer's short convolution and the
  next AttnRes mix run as two kernels (the AMD path has no fused `conv_mix`); the controller GEMMs are `torch.bmm`.
- `amd/ops_dump.py`: recorder of per-operator fixtures (inputs, outputs, cache blocks before and after) for the
  native operators. Bring-up tool, off unless `PAITON_INK2_OPS_DUMP` is set. `amd/loop.py` sets the visit tag and
  records the controller GEMMs through it.
- `amd/experts_bringup.py`: PyTorch experts on dequantized weights with the trained input quantization. Bring-up
  only, not a serving path.
- `__init__.py`: registration of the architectures, config classes and the `inkling2` quantization method.

## Not ported yet

`speculator.py`, `speculator_config.py` (DSpark drafter config), `nvidia/lora.py`, and `adjust_request` of main's
parser (table above).

### Prompt rendering

Main renders Inkling 2 prompts with its own renderer (`vllm/renderers/inkling.py`, `inkling_encoding.py`). It is not
ported. Prompts come from the checkpoint's `chat_template.jinja` through 0.32's default renderer (tokenizer mode
`hf`, picked automatically). On the four fixture prompts that gives the reference's token ids, with tools and with
`reasoning_effort`. 0.32 hands the template tool-call arguments as objects and an earlier reply's reasoning as
`reasoning_content`, which is what the template reads.

What main's renderer does and the template does not, from reading both:

- `tool_choice` `required` or a named function, and `parallel_tool_calls: false`, add one instruction sentence to
  the prompt, and a named choice declares only that tool. The template declares every tool and adds nothing, so
  these requests behave like `auto`. 0.32's `ChatParams` has no `parallel_tool_calls` field to pass on.
- Message text that spells a framing token (for example `<|end_message|>`) is encoded as plain text. Through the
  template it becomes the special token.
- Content parts of type `thinking` and `tool_error`, and the `developer` role: the template raises for them.
- An assistant message without any block gets no `<|content_model_end_sampling|>`. The template always adds one.
- Floats in tool declarations and tool-call arguments use the reference's JSON spelling. The template uses `tojson`.
- A numeric string is accepted as `reasoning_effort`.
- `continue_final_message` leaves a final plain-text assistant message open. Not tried with the template.
