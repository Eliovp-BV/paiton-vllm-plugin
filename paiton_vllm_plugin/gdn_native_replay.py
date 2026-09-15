"""Native compact GDN state through public vLLM layer/backend interfaces.

This external adapter preserves stock convolution/prefill and tensor ownership.
The separately compiled HIP artifact implements speculative recurrent updates,
accepted-prefix replay and state transitions. No Radiance import is required.
"""
import ctypes
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import torch
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend, GDNAttentionMetadataBuilder

from .mxfp4_native import _checked


class ReplayRuntime:
    def __init__(self, manifest_path):
        path = Path(manifest_path).resolve(strict=True)
        manifest = json.loads(path.read_text())
        if manifest.get('abi') != 1 or manifest.get('arch') != 'gfx1201':
            raise ValueError('Unsupported native GDN manifest')
        self.manifest = manifest
        self.libraries = {}
        for name in ('replay', 'bias_cast'):
            entry = manifest['artifacts'][name]
            binary = (path.parent / entry['file']).resolve(strict=True)
            if binary.parent != path.parent or binary.suffix != '.so' or hashlib.sha256(binary.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Native GDN artifact path/hash mismatch')
            self.libraries[name] = ctypes.CDLL(str(binary), mode=ctypes.RTLD_LOCAL)
        lib = self.libraries['replay']
        lib.paiton_paged_replay_abi_version.restype = ctypes.c_int
        if lib.paiton_paged_replay_abi_version() != 1:
            raise ValueError('Native GDN ABI mismatch')
        P, I, L = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
        self.forward = lib.paiton_paged_replay_forward
        self.forward.argtypes = [I]+[P]*5+[L,I,P,P,P,L,L,P,P,P,P,P,I,P,P,I,I,I,I,P]
        self.prepare = lib.paiton_paged_replay_prepare
        self.prepare.argtypes = [P,P,I,P,P,I,I,I,P]
        self.commit = lib.paiton_paged_replay_commit
        self.commit.argtypes = [P,L,L,P,P,P,I,P,P,P,I,I,I,P]
        self.cast = self.libraries['bias_cast'].paiton_bf16_expand
        self.cast.argtypes = [P,P,I,P]
        for function in (self.forward, self.prepare, self.commit, self.cast):
            function.restype = I
        self.stores = {}
        self.layers = {}
        self.calls = {'speculative': 0, 'prefill_prepare': 0, 'ordinary_commit': 0}
        self.speculative_layers = set()
        self.serving = False

    def prepare_serving(self, scope):
        if self.serving or len(self.layers) != 48:
            raise RuntimeError('Expected 48 native GDN layers before first serving boundary')
        stream = torch.cuda.current_stream(scope.error.device).cuda_stream
        for store in self.stores.values():
            valid = store['valid']
            _checked(scope.runtime.hip.hipMemsetAsync(valid.data_ptr(), 0, valid.numel()*4, stream))
        self.serving = True

    def audit(self):
        return dict(logical_layers=len(self.layers), physical_pools=len(self.stores),
                    speculative_layers=len(self.speculative_layers), host_dispatches=dict(self.calls),
                    serving=self.serving, cache_allocation='one committed block per request',
                    auxiliary_bytes=sum(s['updates'].numel()*4+s['valid'].numel()*4 for s in self.stores.values()),
                    artifacts=self.manifest['artifacts'])


@lru_cache(maxsize=1)
def runtime():
    return ReplayRuntime(os.environ['PAITON_GDN_RUNTIME_MANIFEST'])


class PaitonGDNMetadataBuilder(GDNAttentionMetadataBuilder):
    def build(self, *args, **kwargs):
        metadata = super().build(*args, **kwargs)
        indices = metadata.spec_state_indices_tensor
        if indices is not None:
            # The public cache spec owns one committed page. Convolution still
            # needs the original width8 history; expanded views keep that width.
            if indices.shape[1] == 1:
                metadata.spec_state_indices_tensor = indices.expand(-1, 8)
            elif indices.shape[1] != 8:
                raise ValueError('Native compact GDN requires K7 metadata')
        return metadata


class PaitonGDNBackend(GDNAttentionBackend):
    @staticmethod
    def get_name():
        return 'PAITON_GDN_ATTN'

    @staticmethod
    def get_builder_cls():
        return PaitonGDNMetadataBuilder


@PluggableLayer.register_oot(name='QwenGatedDeltaNetAttention')
class PaitonGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.tp_size != 1 or self.num_spec != 7 or self.head_k_dim != 128
                or self.head_v_dim != 128 or self.num_v_heads != 48 or self.num_k_heads != 16
                or self.gqa_interleaved_layout or self.enable_fused_gdn_decode):
            raise ValueError('Native compact GDN currently requires Qwen3.8 TP1 K7 geometry')
        self.paiton_replay = runtime()
        self.paiton_replay_scope = None

    def get_attn_backend(self):
        return PaitonGDNBackend

    def get_kv_cache_spec(self, vllm_config):
        if (vllm_config.cache_config.mamba_cache_mode != 'none'
                or vllm_config.cache_config.enable_prefix_caching
                or vllm_config.scheduler_config.max_num_seqs > 8):
            raise ValueError('Native compact GDN requires cache mode none, prefix caching off, max8 requests')
        return replace(super().get_kv_cache_spec(vllm_config), num_speculative_blocks=0)

    def bind_kv_cache(self, cache):
        super().bind_kv_cache(cache)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Bind native GDN storage before graph capture')
        state = self.kv_cache[1]
        if state.dtype != torch.float32 or tuple(state.shape[1:]) != (48,128,128) or state.stride()[-2:] != (128,1):
            raise ValueError('Unexpected native GDN FP32 cache layout')
        key = state.data_ptr()
        store = self.paiton_replay.stores.get(key)
        if store is None:
            slots = state.shape[0]
            store = dict(state=state, slots=slots,
                         updates=torch.empty((slots,48,2,8,193), device=state.device, dtype=torch.float32),
                         valid=torch.empty((slots,48,2), device=state.device, dtype=torch.int32))
            self.paiton_replay.stores[key] = store
            hip = ctypes.CDLL('libamdhip64.so.7')
            hip.hipMemsetAsync.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t,ctypes.c_void_p]
            _checked(hip.hipMemsetAsync(store['valid'].data_ptr(),0,store['valid'].numel()*4,torch.cuda.current_stream(state.device).cuda_stream))
        elif store['state'].shape != state.shape or store['state'].stride() != state.stride():
            raise ValueError('Aliased native GDN cache geometry differs')
        self.paiton_replay_store = store
        self.paiton_replay.layers[self.prefix] = key

    def bind_native_scope(self, scope):
        self.paiton_replay_scope = scope
        if os.environ.get('PAITON_EXPERIMENTAL_GDN_CONV_PREFILL') == '1':
            from .gdn_native_conv_prefill import runtime as conv_runtime
            self.paiton_conv_prefill = conv_runtime()
        if os.environ.get('PAITON_EXPERIMENTAL_GDN_PREFILL') == '1':
            from .gdn_native_prefill import PaitonGDNPrefill
            if not isinstance(self.chunk_gated_delta_rule, PaitonGDNPrefill):
                self.chunk_gated_delta_rule = PaitonGDNPrefill(
                    self.chunk_gated_delta_rule, scope, self.prefix)
        bias = self.dt_bias
        if bias.device != scope.error.device or not bias.is_contiguous() or bias.numel() != 48:
            raise ValueError('Unexpected GDN bias geometry')
        if bias.dtype == torch.bfloat16:
            expanded = torch.empty(48, device=bias.device, dtype=torch.float32)
            _checked(self.paiton_replay.cast(bias.data_ptr(), expanded.data_ptr(),48,torch.cuda.current_stream(bias.device).cuda_stream))
        elif bias.dtype == torch.float32:
            expanded = bias
        else:
            raise ValueError('Unsupported GDN bias type')
        self.register_buffer('paiton_replay_bias', expanded)

    def _prepare_native_replay(self, metadata):
        scope = self.paiton_replay_scope
        if scope is None:
            raise RuntimeError('Native GDN must bind its serving error before forward')
        store = self.paiton_replay_store
        stream = torch.cuda.current_stream(scope.error.device).cuda_stream
        if metadata.num_prefills:
            indices, initial = metadata.prefill_state_indices, metadata.prefill_has_initial_state
            if indices is None or initial is None or indices.dtype != torch.int32 or initial.dtype != torch.bool or not initial.is_contiguous():
                raise ValueError('Unsupported native GDN prefill transition')
            _checked(self.paiton_replay.prepare(store['valid'].data_ptr(),indices.data_ptr(),indices.stride(0),
                initial.data_ptr(),scope.error.data_ptr(),store['slots'],indices.numel(),48,stream))
            self.paiton_replay.calls['prefill_prepare'] += 1
        if metadata.num_decodes:
            indices = metadata.non_spec_state_indices_tensor[:metadata.num_decodes]
            if indices.dtype != torch.int32 or indices.ndim != 1:
                raise ValueError('Unsupported native GDN ordinary-decode transition')
            state = store['state']
            _checked(self.paiton_replay.commit(state.data_ptr(),state.stride(0),state.stride(1),store['updates'].data_ptr(),
                store['valid'].data_ptr(),indices.data_ptr(),indices.stride(0),0,0,scope.error.data_ptr(),
                store['slots'],indices.numel(),48,stream))
            self.paiton_replay.calls['ordinary_commit'] += 1

    def _native_speculative_update(self, *, A_log,a,b,dt_bias,q,k,v,initial_state,inplace_final_state,
                                   cu_seqlens,ssm_state_indices,num_accepted_tokens,use_qk_l2norm_in_kernel):
        scope, store = self.paiton_replay_scope, self.paiton_replay_store
        if (not inplace_final_state or not use_qk_l2norm_in_kernel
                or any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (q,k,v))
                or A_log.dtype != torch.float32 or not A_log.is_contiguous()
                or a.dtype not in (torch.bfloat16,torch.float32) or a.dtype != b.dtype
                or a.stride(0) != b.stride(0) or a.stride(-1) != 1 or b.stride(-1) != 1
                or ssm_state_indices.dtype != torch.int32 or ssm_state_indices.shape[1] != 8
                or ssm_state_indices.stride(1) not in (0,1) or cu_seqlens.dtype != torch.int32
                or not cu_seqlens.is_contiguous() or num_accepted_tokens.dtype != torch.int32
                or not num_accepted_tokens.is_contiguous()):
            raise ValueError('Unsupported native speculative GDN operands')
        output = torch.empty_like(v)
        state = store['state'];indices = ssm_state_indices
        _checked(self.paiton_replay.forward(1,q.data_ptr(),k.data_ptr(),v.data_ptr(),a.data_ptr(),b.data_ptr(),
            a.stride(0),int(a.dtype == torch.bfloat16),A_log.data_ptr(),self.paiton_replay_bias.data_ptr(),
            state.data_ptr(),state.stride(0),state.stride(1),store['updates'].data_ptr(),store['valid'].data_ptr(),
            output.data_ptr(),cu_seqlens.data_ptr(),indices.data_ptr(),indices.stride(0),
            num_accepted_tokens.data_ptr(),scope.error.data_ptr(),store['slots'],cu_seqlens.numel()-1,48,16,
            torch.cuda.current_stream(state.device).cuda_stream))
        self.paiton_replay.calls['speculative'] += 1
        self.paiton_replay.speculative_layers.add(self.prefix)
        return output, None

    # The stock control flow below is inherited from a separate pinned adapter
    # mixin; only speculative recurrence and state preparation are replaced.
    from .gdn_upstream_control import forward_core as _forward_core
