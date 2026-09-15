"""External adapter for Paiton's native residual/RMSNorm/FP8 producer."""
import ctypes,hashlib,json,os
from functools import lru_cache
from pathlib import Path
import torch
from .mxfp4_native import _checked

class Runtime:
    def __init__(self):
        p=Path(os.environ['PAITON_NORM_FP8_MANIFEST']).resolve(strict=True)
        self.spec=json.loads(p.read_text());binary=(p.parent/self.spec['file']).resolve(strict=True)
        if ((self.spec.get('abi'),self.spec.get('arch'))!=(1,'gfx1201') or binary.parent!=p.parent
                or binary.suffix!='.so' or hashlib.sha256(binary.read_bytes()).hexdigest()!=self.spec['sha256']):
            raise ValueError('Native norm/FP8 manifest mismatch')
        self.lib=ctypes.CDLL(str(binary),mode=ctypes.RTLD_LOCAL)
        self.lib.paiton_norm_fp8_abi_version.restype=ctypes.c_int
        if self.lib.paiton_norm_fp8_abi_version()!=1:raise ValueError('Native norm/FP8 ABI mismatch')
        self.call=self.lib.paiton_norm_fp8
        self.call.argtypes=[ctypes.c_void_p]*9+[ctypes.c_int]*2+[ctypes.c_float,ctypes.c_void_p]
        self.call.restype=ctypes.c_int
        self.layers=set();self.rows=set();self.calls=0;self.fallbacks=0
    def audit(self):
        return dict(layers=sorted(self.layers),rows=sorted(self.rows),native_calls=self.calls,
                    fallbacks=self.fallbacks,artifact=self.spec['sha256'])

@lru_cache(maxsize=1)
def runtime():return Runtime()

class NativeNormAndFp8(torch.nn.Module):
    def __init__(self,original,scope,name):
        super().__init__();self.original=original;self.scope=scope;self.layer_name=name;self.native=runtime()
        # Frozen checkpoint conversion in the external vLLM adapter, once before capture.
        self.register_buffer('gamma',original.weight.detach().float()+1.0)
        self.epsilon=original.variance_epsilon
    def forward(self,x,residual=None):
        good=(x.ndim==2 and x.shape[1]==5120 and 1<=x.shape[0]<=8192
              and x.dtype==torch.bfloat16 and x.is_contiguous() and x.device==self.scope.error.device)
        good=good and (residual is None or (residual.shape==x.shape and residual.dtype==x.dtype
                                           and residual.device==x.device and residual.is_contiguous()))
        padding=self.scope.padding
        good=good and (padding is None or (padding.device==x.device and padding.dtype in (torch.bool,torch.uint8)
                         and padding.ndim==1 and padding.is_contiguous() and padding.numel()>=x.shape[0]))
        if not good:self.native.fallbacks+=1;return self.original(x,residual)
        m=x.shape[0];out=torch.empty_like(x);codes=torch.empty((m,5120),device=x.device,dtype=torch.uint8)
        scale=torch.empty(m,device=x.device,dtype=torch.float32)
        residual_out=torch.empty_like(x) if residual is not None else None
        error=self.scope.profiling_error if self.scope.profiling else self.scope.error
        _checked(self.native.call(x.data_ptr(),self.gamma.data_ptr(),out.data_ptr(),codes.data_ptr(),scale.data_ptr(),
                 0 if residual is None else residual.data_ptr(),0 if residual_out is None else residual_out.data_ptr(),
                 0 if padding is None else padding.data_ptr(),error.data_ptr(),m,5120,self.epsilon,
                 torch.cuda.current_stream(x.device).cuda_stream))
        # GDN has two consumers; tensor ownership preserves codes until both finish.
        out._paiton_rms_prequantized=(self.scope,codes,scale)
        self.native.layers.add(self.layer_name);self.native.rows.add(m);self.native.calls+=1
        return out if residual is None else (out,residual_out)

def bind(model,scope):
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    count=0
    for name,layer in tuple(model.named_modules()):
        if name.endswith(('.input_layernorm','.post_attention_layernorm')):
            if not isinstance(layer,GemmaRMSNorm) or layer.weight.shape!=(5120,):
                raise ValueError('Expected target GemmaRMSNorm H5120')
            owner_name,attribute=name.rsplit('.',1);owner=model.get_submodule(owner_name)
            setattr(owner,attribute,NativeNormAndFp8(layer,scope,name));count+=1
    if count!=128:raise ValueError(f'Expected128 fused norm producers, found{count}')
    scope.rms_prequantized_consumptions=0
