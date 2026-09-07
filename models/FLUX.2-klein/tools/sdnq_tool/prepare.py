# SPDX-License-Identifier: GPL-3.0-only
"""Prepare immutable SDNQ inference weights once during pipeline loading."""
import copy
import torch
from sdnq.dequantizer import SDNQDequantizer


class PreparedDequantizer(SDNQDequantizer):
    def re_quantize_matmul(self, *args, **kwargs):
        return self.prepared_matmul


@torch.inference_mode()
def prepare_transformer_weights(transformer):
    """Keep source parameters intact and cache their deterministic runtime form.

    This pipeline is immutable after preparation. Loading adapters, training or
    changing quantization options requires loading a fresh pipeline.
    """
    count = 0
    total_bytes = 0
    for name, module in transformer.named_modules():
        dequantizer = getattr(module, "sdnq_dequantizer", None)
        if dequantizer is None or not dequantizer.use_quantized_matmul or not dequantizer.re_quantize_for_matmul:
            continue
        if isinstance(dequantizer, PreparedDequantizer):
            raise ValueError("Transformer weights have already been prepared")
        prepared = dequantizer.re_quantize_matmul(module.weight, module.scale, zero_point=module.zero_point)
        replacement = copy.copy(dequantizer)
        replacement.__class__ = PreparedDequantizer
        tensors = []
        for index, tensor in enumerate(prepared):
            module.register_buffer(f"_stock_prepared_{index}", tensor, persistent=False)
            tensors.append(tensor)
            total_bytes += tensor.numel() * tensor.element_size()
        replacement.prepared_matmul = tuple(tensors)
        module.sdnq_dequantizer = replacement
        count += 1
    return {"prepared_layers": count, "prepared_bytes": total_bytes}
