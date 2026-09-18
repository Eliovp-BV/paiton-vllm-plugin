# SPDX-License-Identifier: Apache-2.0
"""
Paiton vLLM Plugin - Entry points for vLLM plugin system.

This plugin registers:
1. Paiton platform (based on ROCm) for optimized execution
2. Paiton model architectures for compiled models
"""

import os

from paiton_vllm_plugin.artifact_manifest import (
    ArtifactCompatibilityError,
    detect_runtime_gpu_arch,
)


def paiton_platform_plugin() -> str | None:
    """
    Platform plugin entry point.

    Returns the fully qualified name of the PaitonPlatform class if
    running on a supported AMD GPU, otherwise returns None.
    """
    # external runtime compatibility uses the upstream ROCm platform and model classes.
    if os.environ.get("PAITON_RUNTIME_COMPAT_FLOW", "0") == "1":
        return None
    # Allow explicit opt-out so users can run vanilla vLLM on MI300 systems.
    # (Paiton's platform changes KV-cache layout and is only compatible with
    # Paiton-compiled model runtimes.)
    if os.environ.get("VLLM_DISABLE_PAITON_PLATFORM", "0") == "1":
        return None
    # The pinned vLLM reference stack relies on AMD SMI for built-in ROCm
    # discovery, but that probe can fail after Torch initializes ROCm. Allow
    # correctness harnesses to select vLLM's unmodified ROCmPlatform through
    # this already-discovered entry point without enabling any Paiton layout
    # or attention overrides.
    if os.environ.get("VLLM_PAITON_VANILLA_ROCM_PLATFORM", "0") == "1":
        return "vllm.platforms.rocm.RocmPlatform"

    force_paiton = os.environ.get("VLLM_USE_PAITON_PLATFORM", "0") == "1"
    try:
        selected_arch = detect_runtime_gpu_arch()
    except ArtifactCompatibilityError:
        selected_arch = None
    supported_arches = {"gfx942", "gfx950", "gfx1200", "gfx1201"}
    if selected_arch in supported_arches or force_paiton:
        return "paiton_vllm_plugin.paiton_platform.PaitonPlatform"

    return None


def register_paiton_models() -> None:
    """
    General plugin entry point to register Paiton model architectures.

    This registers the PaitonLlamaForCausalLM and other Paiton-compiled
    model classes with the vLLM ModelRegistry.
    """
    if os.environ.get("PAITON_RUNTIME_COMPAT_FLOW", "0") == "1":
        from paiton_runtime_compat import register
        register()
        return
    if os.environ.get("PAITON_EXPERIMENTAL_MXFP4_W4A8") == "1":
        # Use upstream model classes and its ROCm platform in this isolated
        # integration lane; no unrelated model/loader compatibility patches.
        from .mxfp4_native import register_paiton_mxfp4
        register_paiton_mxfp4()
        return
    from .gguf_detokenizer import install_gguf_detokenizer_compat
    install_gguf_detokenizer_compat()
    from vllm import ModelRegistry
    from paiton_vllm_plugin import gguf_model_loader  # noqa: F401

    # Register Paiton model architectures
    # Users can specify these in their model config's architectures field
    model_registrations = {
        "PaitonMiniCPM5AWQForCausalLM": "paiton_vllm_plugin.models.paiton_minicpm5_awq:PaitonMiniCPM5AWQForCausalLM",
        "PaitonGptOssForCausalLM": "paiton_vllm_plugin.models.paiton_gptoss:PaitonGptOssForCausalLM",
        "PaitonLlamaForCausalLM": "paiton_vllm_plugin.models.paiton_llama:PaitonLlamaForCausalLM",
        "PaitonQwen2ForCausalLM": "paiton_vllm_plugin.models.paiton_qwen:PaitonQwen2ForCausalLM",
        "PaitonQwen3ForCausalLM": "paiton_vllm_plugin.models.paiton_qwen3:PaitonQwen3ForCausalLM",
        "PaitonQwen3MoeForCausalLM": "paiton_vllm_plugin.models.paiton_qwen3_moe:PaitonQwen3MoeForCausalLM",
        "PaitonQwen3CoderForCausalLM": "paiton_vllm_plugin.models.paiton_qwen3_coder:PaitonQwen3CoderForCausalLM",
        "PaitonQwen38ForCausalLM": "paiton_vllm_plugin.models.paiton_qwen38:PaitonQwen38ForCausalLM",
        "PaitonQwen38GGUFForCausalLM": "paiton_vllm_plugin.models.paiton_qwen38_gguf:PaitonQwen38GGUFForCausalLM",
        "PaitonQwen38GGUFForConditionalGeneration": "paiton_vllm_plugin.models.paiton_qwen38_gguf_multimodal:PaitonQwen38GGUFForConditionalGeneration",
        "PaitonQwen38ForConditionalGeneration": "paiton_vllm_plugin.models.paiton_qwen38_multimodal:PaitonQwen38ForConditionalGeneration",
        "PaitonOrnith15ForCausalLM": "paiton_vllm_plugin.models.paiton_ornith15:PaitonOrnith15ForCausalLM",
    }

    for arch, model_path in model_registrations.items():
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, model_path)
