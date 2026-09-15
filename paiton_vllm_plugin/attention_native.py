"""External tensor adapter for Paiton's native paged-attention C ABI.

Stock vLLM owns metadata, cache writes and unsupported attention.
The compiled HIP artifact has no Torch, Triton or Radiance dependency.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend, TritonAttentionImpl
from vllm.v1.kv_cache_interface import KVQuantMode

_RUNTIME = None


class NativeAttentionRuntime:
    def __init__(self):
        path = Path(os.environ['PAITON_ATTENTION_RUNTIME_MANIFEST']).resolve(strict=True)
        spec = json.loads(path.read_text())
        if ((spec.get('abi'), spec.get('arch')) != (1, 'gfx1201')
            or spec.get('arithmetic') not in (
                'bf16_query_fp8_kv_f16_wmma_fp32_softmax',
                'bf16_query_fp8_kv_f16_qk_bf16_pv_fp32_softmax',
                'bf16_query_fp8_kv_short_bf16_pv_long_f16_lazy_softmax',
                'bf16_query_fp8_kv_split_verify_f16_lazy_softmax')):
            raise ValueError('Unsupported native attention manifest')
        binary = (path.parent/spec['file']).resolve(strict=True)
        if binary.parent != path.parent or binary.suffix != '.so':
            raise ValueError('Expected sibling native attention artifact')
        if hashlib.sha256(binary.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError('Native attention artifact hash mismatch')
        self.lib = ctypes.CDLL(str(binary), mode=ctypes.RTLD_LOCAL)
        self.run = self.lib.paiton_paged_fp8_attention
        self.run.argtypes = [ctypes.c_void_p]*9 + [ctypes.c_int]*6 + [ctypes.c_int64]*3 + [ctypes.c_void_p]
        self.run.restype = ctypes.c_int
        self.verify = None
        self.verify_workspace_bytes = 0
        if spec['arithmetic'] == 'bf16_query_fp8_kv_split_verify_f16_lazy_softmax':
            self.verify = self.lib.paiton_paged_fp8_verify
            self.verify.argtypes = [ctypes.c_void_p]*10 + [ctypes.c_size_t] + [ctypes.c_int]*6 + [ctypes.c_int64]*3 + [ctypes.c_void_p]
            self.verify.restype = ctypes.c_int
            workspace_size = self.lib.paiton_verify_workspace_bytes
            workspace_size.argtypes = [ctypes.c_int]*2
            workspace_size.restype = ctypes.c_size_t
            self.verify_workspace_bytes = workspace_size(80,8)
            if self.verify_workspace_bytes != 15974400:
                raise ValueError('Unsupported verification workspace contract')
        self.sha256 = spec['sha256']


class PaitonAttentionBackend(TritonAttentionBackend):
    @staticmethod
    def get_name():
        return 'CUSTOM'

    @staticmethod
    def get_impl_cls():
        return PaitonAttentionImpl


class PaitonAttentionImpl(TritonAttentionImpl):
    is_paiton_paged_attention = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paiton_scope = None
        self.paiton_runtime = None
        self.paiton_audit = dict(native_query_lengths=set(), fallback_query_lengths=set(), native_verify_query_lengths=set())

    def bind_paiton_scope(self, scope):
        global _RUNTIME
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Bind native attention before graph capture')
        if _RUNTIME is None:
            _RUNTIME = NativeAttentionRuntime()
        self.paiton_runtime = _RUNTIME
        self.paiton_scope = scope
        if _RUNTIME.verify is not None and not hasattr(scope, 'attention_verify_scratch'):
            # One target forward owns a stream; all 16 layers reuse this allocation.
            scope.attention_verify_scratch = torch.empty(_RUNTIME.verify_workspace_bytes,
                                                        dtype=torch.uint8,device=scope.error.device)
        self.paiton_audit['artifact_sha256'] = _RUNTIME.sha256

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        meta = attn_metadata
        supported = (
            meta is not None and self.paiton_scope is not None
            and (1 if self.paiton_runtime.verify is not None else 2) <= meta.max_query_len <= 8192
            and 1 <= meta.num_actual_tokens <= 8192
            and self.num_heads == 24 and self.num_kv_heads == 4 and self.head_size == 256
            and self.scale == .0625 and self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
            and self.attn_type == 'decoder' and self.sliding_window == (-1, -1)
            and self.alibi_slopes is None and not self.use_alibi_sqrt
            and not self.logits_soft_cap and self.sinks is None and self.chunk_lookback == -1
            and meta.causal is True and not meta.use_cascade
            and meta.mm_prefix_range_tensor is None and meta.rswa_prefix_lens is None
            and output_scale is None and output_block_scale is None
            and query.dtype == torch.bfloat16 and output.dtype == torch.bfloat16
            and query.is_contiguous() and output.is_contiguous()
            and kv_cache.element_size() == 1 and tuple(kv_cache.shape[1:2]) == (4,)
            and kv_cache.shape[-1] == 512 and kv_cache.stride(-1) == 1
            and 1 <= meta.query_start_loc.numel()-1 <= 8
            and meta.block_table.is_contiguous() and meta.block_table.shape[1] <= 512
        )
        if not supported:
            if meta is not None:
                self.paiton_audit['fallback_query_lengths'].add(meta.max_query_len)
            return super().forward(layer, query, key, value, kv_cache, meta, output,
                                   output_scale, output_block_scale)
        scope = self.paiton_scope
        error = scope.profiling_error if scope.profiling else scope.error
        verify = (self.paiton_runtime.verify is not None
                  and meta.max_query_len <= 10 and kv_cache.shape[2] in (16,1648))
        native = self.paiton_runtime.verify if verify else self.paiton_runtime.run
        scratch_args = ((scope.attention_verify_scratch.data_ptr(),
                         scope.attention_verify_scratch.numel()) if verify else ())
        status = native(
            query.data_ptr(), kv_cache.data_ptr(), output.data_ptr(),
            meta.query_start_loc.data_ptr(), meta.seq_lens.data_ptr(), meta.block_table.data_ptr(),
            layer._k_scale.data_ptr(), layer._v_scale.data_ptr(), error.data_ptr(),
            *scratch_args,
            meta.query_start_loc.numel()-1, meta.num_actual_tokens, kv_cache.shape[0],
            kv_cache.shape[2], meta.block_table.shape[1], meta.max_query_len,
            *kv_cache.stride()[:3], torch.cuda.current_stream(query.device).cuda_stream)
        if status:
            raise RuntimeError(f'Native paged attention HIP status {status}')
        self.paiton_audit['native_query_lengths'].add(meta.max_query_len)
        if verify:
            self.paiton_audit['native_verify_query_lengths'].add(meta.max_query_len)
        return output
