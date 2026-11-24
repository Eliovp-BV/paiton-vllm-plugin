# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short-convolution state as a sliding-window cache, without a Triton import.

vLLM 0.32's `vllm/models/inkling/amd/sconv_swa_attn.py` holds the same classes,
but it imports `.ops.sconv` (Triton kernels) at its top and its metadata
builder launches a Triton kernel. This module keeps the pure Python parts of
that file (cache spec, stream ranges, metadata record, backend) and builds the
per-token metadata with Paiton's native operator library (`native_ops`).

Layout of the cache tensor vLLM binds to `kv_cache`: bf16
`(blocks, 8 heads, 4 tokens, 1024)`; every head packs the K, V, attention-output
and MLP-output streams at `stream_ranges`.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, SlidingWindowSpec

from . import native_ops


@dataclass
class InklingSconvMetadata(AttentionMetadata):
    block_table: torch.Tensor  # [num_reqs, max_blocks] physical blocks per request
    slot_mapping: torch.Tensor  # [T] int64 flat slot of each token (-1: skip)
    seq_idx: torch.Tensor  # [T] int32 token -> request
    query_start: torch.Tensor  # [T] int32 first x-row of each token's request


class NativeSconvMetadataBuilder(AttentionMetadataBuilder[InklingSconvMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, SlidingWindowSpec)
        # Persistent per-token buffers, so a captured graph keeps its pointers.
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.seq_idx_buffer = torch.empty(max_num_tokens, dtype=torch.int32, device=device)
        self.query_start_buffer = torch.empty(max_num_tokens, dtype=torch.int32, device=device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> InklingSconvMetadata:
        num_actual_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
        num_padded_tokens = common_attn_metadata.slot_mapping.shape[0]
        assert num_padded_tokens >= num_actual_tokens
        native_ops.seq_metadata(
            common_attn_metadata.query_start_loc,
            common_attn_metadata.num_reqs,
            num_actual_tokens,
            self.seq_idx_buffer,
            self.query_start_buffer,
            num_padded_tokens,
        )
        return InklingSconvMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:num_padded_tokens],
            seq_idx=self.seq_idx_buffer[:num_padded_tokens],
            query_start=self.query_start_buffer[:num_padded_tokens],
        )


class NativeSconvBackend(AttentionBackend):
    """Backend of the conv-state cache: it only manages the cache; the
    convolution runs out-of-band."""

    @staticmethod
    def get_name() -> str:
        return "INKLING_SCONV_SWA"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("the conv-state backend has no attention implementation")

    @staticmethod
    def get_builder_cls() -> type[NativeSconvMetadataBuilder]:
        return NativeSconvMetadataBuilder


class InklingConvState(nn.Module, AttentionLayerBase):
    """Per-decoder-layer owner of one sliding-window conv-state cache (the
    layout of vLLM's class of the same name)."""

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        kernel_size: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        # Bound to the manager's paged cache by bind_kv_cache.
        self.kv_cache = torch.tensor([])
        tp_size = get_tensor_model_parallel_world_size()
        assert tp_size <= num_kv_heads
        self.num_kv_heads = num_kv_heads // tp_size
        hidden_per_head = hidden_size // num_kv_heads
        # Per-head width: K + V + attention-output chunk + MLP-output chunk,
        # padded to a power of two so the conv page is a multiple of the
        # attention page.
        raw_head_size = 2 * head_dim + 2 * hidden_per_head
        self.head_size = 1 << (raw_head_size - 1).bit_length()
        self.sliding_window = kernel_size
        self.block_size = kernel_size
        # (offset, width) of each stream within a head: K, V, attention, MLP.
        self.stream_ranges: tuple[tuple[int, int], ...] = (
            (0, head_dim),
            (head_dim, head_dim),
            (2 * head_dim, hidden_per_head),
            (2 * head_dim + hidden_per_head, hidden_per_head),
        )
        vllm_config = get_current_vllm_config()
        self._dtype = vllm_config.model_config.dtype
        assert self._dtype == torch.bfloat16, "the conv-state cache is bf16 only"
        # The runner finds attention-like layers in the static forward context.
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(self): ...

    @property
    def cache_block_size(self) -> int:
        if self.kv_cache.numel() > 0:
            return self.kv_cache.shape[2]
        return self.block_size

    def get_attn_backend(self) -> type[AttentionBackend]:
        return NativeSconvBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=0,  # all four streams are packed into head_size
            dtype=self._dtype,
            sliding_window=self.sliding_window,
        )
