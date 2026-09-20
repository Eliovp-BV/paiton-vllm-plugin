# Paiton DFlash integration

This module configures the installed vLLM API server and registers the Paiton
worker hooks. vLLM owns requests, model state, verification and sampling. Native
arithmetic is supplied by separately built HIP libraries with pinned artifact,
source and shape guards. The compiler and native libraries do not depend on
PyTorch or Triton; the external adapter retains existing vLLM tensor and graph
facilities.

Use the [Qwen3.8 image and launcher](../../models/Qwen3.8-MXFP4-DFlash2/README.md)
for the tested configuration. Installing this Python module alone does not supply
the native libraries. `paiton-dflash-serve` accepts the installed API server's
arguments, including `--model` and `--speculative-config`.

| Setting | Module default | Purpose |
|---|---|---|
| `PAITON_DFLASH_NATIVE` | `0` | Enable supported native component hooks |
| `PAITON_DFLASH_DOWN_PROJECTION` | `1` behind native master | Native down projection within shape guards |
| `PAITON_DFLASH_SILU_QUANT` | `1` behind native master | Native SiLU/FP8 producer |
| `PAITON_DFLASH_NATIVE_BUNDLE` | `/opt/paiton/runtime/dflash-down` | Native library and contract directory |
| `PAITON_DFLASH_CONTEXT_NORM_ROPE` | `0` | Supported native context normalization/RoPE |
| `PAITON_DFLASH_DRAFT_SAMPLE_METHOD` | `inherit` | Draft `greedy` or `probabilistic` proposal method |
| `PAITON_DFLASH_RERANK` | `0` | Draft-instance rerank policy; target policy stays unchanged |

The updated 65K image supplies its tested settings at startup, including native
MLP/context components and draft-only R128 probabilistic proposals. The target
retains R80/top-k20. Module defaults preserve the original behavior elsewhere.

`PAITON_DFLASH_GATE_PROJECTION` and `PAITON_DFLASH_CONTEXT_KV` are unavailable in
the supplied bundle. Verifier caps, alternative block retention, context graphs
and uniform graphs remain disabled in the released configuration. Other
combinations are not qualified by the published benchmark and some intentionally
fail validation. `configuration.py` defines the complete validation rules.

Settings are frozen before model load and graph capture. Restart to change them.
Unsupported shapes retain the original provider. Source or artifact mismatches
fail rather than bypassing the compatibility checks. Native MLP and native
context hooks compose in a fixed order, with the exact transformed source checked.
