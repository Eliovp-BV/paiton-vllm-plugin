# SPDX-License-Identifier: Apache-2.0
"""What vLLM 0.32 lacks to serve Inkling 2 with its own drafter (vLLM main has it built in).

`install()` adds three things: the draft config made from the target's `mtp_config`, the draft model class, and
the choice of `Inkling2DSparkSpeculator` for method `dspark` with that draft. Light on imports: it runs in every
process that loads the plugin.
"""

import sys

from .configs import Inkling2DSparkConfig

ARCH = "Inkling2DSparkModel"
_original_override = None


def _draft_hf_config(hf_config):
    """`SpeculativeConfig.hf_config_override` with the Inkling 2 branch. Module level: vLLM pickles it."""
    if hf_config.model_type in ("inkling2", "inkling2_loop"):
        draft = Inkling2DSparkConfig.from_target(hf_config)
        # vLLM 0.32 rewrites every dspark draft it does not know into another model's draft class. It leaves the
        # second name alone; the registry resolves the first.
        draft.architectures = [ARCH, "K3DSparkModel"]
        return draft
    return _original_override(hf_config)


def install() -> None:
    global _original_override
    if _original_override is not None:
        return
    from vllm import ModelRegistry
    from vllm.config.speculative import SpeculativeConfig
    from vllm.v1.worker.gpu import spec_decode

    _original_override = SpeculativeConfig.hf_config_override
    SpeculativeConfig.hf_config_override = staticmethod(_draft_hf_config)
    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, f"{__package__}.amd.draft:{ARCH}")

    original_init = spec_decode.init_speculator

    def init_speculator(vllm_config, device, req_states):
        spec = vllm_config.speculative_config
        if spec.method == "dspark" and ARCH in spec.draft_model_config.architectures:
            from .speculator import Inkling2DSparkSpeculator

            return Inkling2DSparkSpeculator(vllm_config, device)
        return original_init(vllm_config, device, req_states)

    spec_decode.init_speculator = init_speculator
    # The model runner binds the name at import; it may already be imported.
    runner = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    if runner is not None:
        runner.init_speculator = init_speculator
