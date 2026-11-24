# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 for vLLM 0.32 on ROCm, as plugin model code.

vLLM 0.32 has the first Inkling generation only. This package is vLLM's
Inkling 2 code brought to 0.32: the shared files of vLLM main (`configs.py`,
`processing.py`, `quant_config.py`), the AMD path of the draft pull request
67685 (`amd/`, gfx950 only), and a port of the looped model to that path
(`amd/loop.py`). `UPSTREAM.md` lists every upstream file and what changed.

Switched on by PAITON_INKLING2=1 (see `paiton_vllm_plugin.register_paiton_models`);
serve with `--quantization inkling2`.
"""

from .configs import Inkling2LoopMMConfig, Inkling2MMConfig


class _NoModelOptQuantMixin:
    def __init__(self, **kwargs) -> None:
        # config.json declares a ModelOpt `quantization_config`; the per-tensor
        # precisions are in hf_quant_config.json. vLLM 0.32 reads that file
        # only when the config carries none (main special-cases "inkling2").
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)


class PaitonInkling2MMConfig(_NoModelOptQuantMixin, Inkling2MMConfig):
    pass


class PaitonInkling2LoopMMConfig(_NoModelOptQuantMixin, Inkling2LoopMMConfig):
    pass


def register() -> None:
    """Add the Inkling 2 architectures, configs and quantization method to
    vLLM 0.32's registries (main has them built in)."""
    from vllm import ModelRegistry
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from .quant_config import Inkling2QuantConfig

    _CONFIG_REGISTRY["inkling2"] = PaitonInkling2MMConfig
    _CONFIG_REGISTRY["inkling2_loop"] = PaitonInkling2LoopMMConfig
    register_quantization_config("inkling2")(Inkling2QuantConfig)
    for arch, module in (
        ("Inkling2ForConditionalGeneration", "model"),
        ("Inkling2LoopForConditionalGeneration", "loop"),
    ):
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, f"{__name__}.amd.{module}:{arch}")
