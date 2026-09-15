"""External vLLM adapter for attributed, standalone native HIP GDN prefill."""
import ctypes
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import torch

from .mxfp4_native import _checked


class PrefillRuntime:
    def __init__(self, path):
        path=Path(path).resolve(strict=True)
        self.manifest=json.loads(path.read_text())
        if self.manifest.get('abi')!=1 or self.manifest.get('arch')!='gfx1201':
            raise ValueError('Unsupported native GDN prefill manifest')
        binary=(path.parent/self.manifest['file']).resolve(strict=True)
        if binary.parent!=path.parent or binary.suffix!='.so' or hashlib.sha256(binary.read_bytes()).hexdigest()!=self.manifest['sha256']:
            raise ValueError('Native GDN prefill artifact path/hash mismatch')
        self.library=ctypes.CDLL(str(binary),mode=ctypes.RTLD_LOCAL)
        self.library.paiton_gdn_prefill_abi_version.restype=ctypes.c_int
        if self.library.paiton_gdn_prefill_abi_version()!=1:
            raise ValueError('Unsupported native GDN prefill ABI')
        self.call=self.library.paiton_gdn_prefill
        self.call.argtypes=[ctypes.c_void_p]*11+[ctypes.c_size_t]+[ctypes.c_int]*4+[ctypes.c_float,ctypes.c_void_p]
        self.call.restype=ctypes.c_int
        self.size=self.library.paiton_gdn_prefill_workspace_size
        self.size.argtypes=[ctypes.c_int]*2;self.size.restype=ctypes.c_size_t
        self.workspace=None
        self.layers=set();self.native_calls=0;self.fallback_calls=0;self.rows=set()

    def bind(self, scope):
        if self.workspace is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Native GDN prefill workspace must be bound before capture')
            self.workspace=torch.empty(self.size(8192,48),device=scope.error.device,dtype=torch.uint8)
        elif self.workspace.device!=scope.error.device:
            raise ValueError('Native GDN prefill requires one GPU')

    def audit(self):
        return dict(layers=sorted(self.layers),native_calls=self.native_calls,
                    fallback_calls=self.fallback_calls,rows=sorted(self.rows),
                    scratch_bytes=0 if self.workspace is None else self.workspace.numel(),
                    artifact=self.manifest['sha256'],
                    attribution=self.manifest['attribution'])


@lru_cache(maxsize=1)
def runtime():
    return PrefillRuntime(os.environ['PAITON_GDN_PREFILL_RUNTIME_MANIFEST'])


class PaitonGDNPrefill(torch.nn.Module):
    def __init__(self, original, scope, layer_name):
        super().__init__()
        self.original=original
        self.scope=scope
        self.native=runtime();self.native.bind(scope)
        self.layer_name=layer_name

    def forward(self, q,k,v,g,beta,scale=None,initial_state=None,
                output_final_state=False,cu_seqlens=None,chunk_indices=None,
                chunk_offsets=None,use_qk_l2norm_in_kernel=False,core_attn_out=None):
        tensors=(q,k,v,g,beta,initial_state,cu_seqlens)
        supported=(all(isinstance(x,torch.Tensor) and x.device==self.scope.error.device and x.is_contiguous() for x in tensors)
            and q.ndim==4 and k.shape==q.shape and q.shape[0]==1 and q.shape[2:]==(16,128)
            and 1<=q.shape[1]<=8192 and v.shape==(1,q.shape[1],48,128)
            and q.dtype==k.dtype==v.dtype==torch.bfloat16
            and g.shape==beta.shape==(1,q.shape[1],48) and g.dtype==beta.dtype==torch.float32
            and initial_state.ndim==4 and initial_state.shape[1:]==(48,128,128)
            and 1<=initial_state.shape[0]<=8 and initial_state.dtype==torch.float32
            and cu_seqlens.shape==(initial_state.shape[0]+1,) and cu_seqlens.dtype==torch.int32
            and output_final_state and not use_qk_l2norm_in_kernel and core_attn_out is None)
        if not supported:
            self.native.fallback_calls+=1
            return self.original(q=q,k=k,v=v,g=g,beta=beta,scale=scale,
                initial_state=initial_state,output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,chunk_indices=chunk_indices,chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,core_attn_out=core_attn_out)
        out=torch.empty_like(v);last=torch.empty_like(initial_state)
        buffers=(q,k,v,g,beta,initial_state,out,last,cu_seqlens,self.native.workspace,self.scope.error)
        _checked(self.native.call(*[x.data_ptr() for x in buffers],self.native.workspace.numel(),
                 initial_state.shape[0],q.shape[1],48,16,128**-.5 if scale is None else float(scale),
                 torch.cuda.current_stream(q.device).cuda_stream))
        self.native.native_calls+=1;self.native.rows.add(q.shape[1]);self.native.layers.add(self.layer_name)
        return out,last
