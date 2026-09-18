"""Opt-in native recurrent prefill with stock vLLM GDN cache ownership.

The existing native HIP artifact receives vLLM's gathered initial state and
returns the final state for vLLM to scatter. Convolution, speculative recurrence,
cache specifications, prefix-cache copies and metadata remain upstream.
"""

import os


def validate_config(config):
    """Fail closed outside the APC configuration prepared for qualification."""
    if os.environ.get("PAITON_EXPERIMENTAL_GDN_REPLAY") == "1":
        raise ValueError("Stock-GDN native prefill cannot use compact GDN replay")
    if (not config.cache_config.enable_prefix_caching
            or config.cache_config.mamba_cache_mode != "align"):
        raise ValueError("Stock-GDN native prefill requires prefix caching and align mode")
    if (config.parallel_config.tensor_parallel_size != 1
            or config.parallel_config.pipeline_parallel_size != 1
            or not 1 <= config.scheduler_config.max_num_seqs <= 8):
        raise ValueError("Stock-GDN native prefill requires TP1/PP1 and at most8 requests")
    if config.scheduler_config.async_scheduling:
        raise ValueError("Stock-GDN native prefill requires synchronous scheduling for shared workspace")


def bind(model, scope):
    """Wrap only the pure prefill operation, without replacing the GDN layer."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    from .gdn_native_prefill import PaitonGDNPrefill

    layers = [layer for layer in model.modules()
              if isinstance(layer, QwenGatedDeltaNetAttention)]
    if len(layers) != 48:
        raise ValueError("Expected48 stock GDN layers before native prefill binding")

    # Check every layer before modifying any. In particular, do not attach to
    # the compact replay subclass even if an environment flag changed later.
    for layer in layers:
        if (hasattr(layer, "paiton_replay") or layer.tp_size != 1
                or layer.num_spec != 7 or layer.head_k_dim != 128
                or layer.head_v_dim != 128 or layer.num_v_heads != 48
                or layer.num_k_heads != 16 or layer.gqa_interleaved_layout
                or layer.enable_fused_gdn_decode):
            raise ValueError("Stock-GDN native prefill requires Qwen3.8 TP1 K7 geometry")
        if isinstance(layer.chunk_gated_delta_rule, PaitonGDNPrefill):
            raise ValueError("Stock-GDN native prefill was already bound")

    # Construct all wrappers before installing them. Their native workspace is
    # shared across these sequential layer calls, as in the qualified path.
    wrappers = [PaitonGDNPrefill(layer.chunk_gated_delta_rule, scope, layer.prefix)
                for layer in layers]
    for layer, wrapper in zip(layers, wrappers):
        layer.chunk_gated_delta_rule = wrapper
    return tuple(layer.prefix for layer in layers)
