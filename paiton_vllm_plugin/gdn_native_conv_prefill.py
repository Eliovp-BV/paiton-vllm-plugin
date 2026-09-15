"""External tensor ownership for attributed native HIP fused GDN preparation."""
import ctypes
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import torch
from .mxfp4_native import _checked

class ConvArgs(ctypes.Structure):
    _fields_=[(n,ctypes.c_void_p) for n in ['x','weight','bias','state','indices','initial','a','b','alog','dtbias','q','k','v','g','beta','cu','error']]+[(n,ctypes.c_int64) for n in ['xpitch','cs_seq','cs_dim','cs_tok','ci_stride','ab_stride']]+[(n,ctypes.c_int) for n in ['sequences','tokens','slots','ab_bf16']]

class ConvRuntime:
    def __init__(self):
        p=Path(os.environ['PAITON_GDN_CONV_PREFILL_MANIFEST']).resolve(strict=True)
        self.manifest=json.loads(p.read_text())
        binary=(p.parent/self.manifest['file']).resolve(strict=True)
        if (self.manifest.get('abi'),self.manifest.get('arch'))!=(1,'gfx1201') or binary.parent!=p.parent or binary.suffix!='.so' or hashlib.sha256(binary.read_bytes()).hexdigest()!=self.manifest['sha256']:
            raise ValueError('Native convolution manifest/hash mismatch')
        self.lib=ctypes.CDLL(str(binary),mode=ctypes.RTLD_LOCAL)
        self.lib.paiton_gdn_conv_args_size.restype=ctypes.c_size_t
        if self.lib.paiton_gdn_conv_args_size()!=ctypes.sizeof(ConvArgs):
            raise ValueError('Native convolution ABI layout differs')
        self.call=self.lib.paiton_gdn_conv_prep
        self.call.argtypes=[ctypes.POINTER(ConvArgs),ctypes.c_void_p];self.call.restype=ctypes.c_int
        self.layers=set();self.rows=set();self.native_calls=0;self.fallback_calls=0

    def audit(self):
        return dict(layers=sorted(self.layers),rows=sorted(self.rows),native_calls=self.native_calls,
                    fallback_calls=self.fallback_calls,artifact=self.manifest['sha256'])

    def prepare(self,layer,x,a,b,state,weight,meta):
        indices=meta.non_spec_state_indices_tensor;initial=meta.has_initial_state;cu=meta.non_spec_query_start_loc
        bias=layer.conv1d.bias;alog=layer.A_log;dt=layer.paiton_replay_bias
        ts=(x,a,b,state,weight,indices,initial,cu,alog,dt)
        good=(meta.spec_sequence_masks is None and meta.num_prefills>0 and meta.num_decodes==0
            and layer.activation in ('silu','swish') and all(isinstance(t,torch.Tensor) and t.device==x.device for t in ts)
            and x.ndim==2 and x.shape[1]==10240 and x.stride(1)==1 and x.stride(0)%4==0 and 1<=x.shape[0]<=8192
            and a.shape==b.shape==(x.shape[0],48) and a.dtype==b.dtype and a.dtype in (torch.bfloat16,torch.float32)
            and a.stride(0)==b.stride(0) and a.stride(1)==b.stride(1)==1
            and state.ndim==3 and state.shape[1]==10240 and state.shape[2]>=3
            and weight.shape==(10240,4) and weight.is_contiguous()
            and x.dtype==state.dtype==weight.dtype==torch.bfloat16
            and (bias is None or (bias.dtype==torch.bfloat16 and bias.shape==(10240,) and bias.is_contiguous()))
            and indices.ndim==1 and 1<=indices.numel()<=8 and indices.dtype==torch.int32
            and initial.shape==indices.shape and initial.dtype==torch.bool and initial.is_contiguous()
            and cu.shape==(indices.numel()+1,) and cu.dtype==torch.int32 and cu.is_contiguous()
            and alog.dtype==dt.dtype==torch.float32 and alog.shape==dt.shape==(48,) and alog.is_contiguous() and dt.is_contiguous())
        if not good:
            self.fallback_calls+=1
            return None
        T=x.shape[0]
        q=torch.empty((T,16,128),device=x.device,dtype=torch.bfloat16);k=torch.empty_like(q)
        v=torch.empty((T,48,128),device=x.device,dtype=torch.bfloat16)
        g=torch.empty((T,48),device=x.device,dtype=torch.float32);beta=torch.empty_like(g)
        scope=layer.paiton_replay_scope
        error=scope.profiling_error if scope.profiling else scope.error
        args=ConvArgs(*[0 if t is None else t.data_ptr() for t in (x,weight,bias,state,indices,initial,a,b,alog,dt,q,k,v,g,beta,cu,error)],
                      x.stride(0),*state.stride(),indices.stride(0),a.stride(0),indices.numel(),T,state.shape[0],int(a.dtype==torch.bfloat16))
        _checked(self.call(ctypes.byref(args),torch.cuda.current_stream(x.device).cuda_stream))
        self.native_calls+=1;self.layers.add(layer.prefix);self.rows.add(T)
        return q,k,v,g,beta

@lru_cache(maxsize=1)
def runtime():
    return ConvRuntime()
