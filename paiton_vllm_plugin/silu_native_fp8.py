"""External vLLM adapter for a native SiLU/product/FP8 producer."""
import ctypes,hashlib,json,os
from functools import lru_cache
from pathlib import Path
import torch
from .mxfp4_native import _checked
class Runtime:
    def __init__(self):
        p=Path(os.environ['PAITON_SILU_FP8_MANIFEST']).resolve(strict=True);self.spec=json.loads(p.read_text());binary=(p.parent/self.spec['file']).resolve(strict=True)
        if (self.spec.get('abi'),self.spec.get('arch'))!=(1,'gfx1201') or binary.parent!=p.parent or binary.suffix!='.so' or hashlib.sha256(binary.read_bytes()).hexdigest()!=self.spec['sha256']:
            raise ValueError('Native SiLU/FP8 manifest mismatch')
        self.lib=ctypes.CDLL(str(binary),mode=ctypes.RTLD_LOCAL);self.lib.paiton_silu_fp8_abi_version.restype=ctypes.c_int
        if self.lib.paiton_silu_fp8_abi_version()!=1:raise ValueError('Native SiLU/FP8 ABI mismatch')
        self.call=self.lib.paiton_silu_fp8;self.call.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*2+[ctypes.c_void_p];self.call.restype=ctypes.c_int
        self.layers=set();self.rows=set();self.calls=0;self.fallbacks=0
    def audit(self):return dict(layers=sorted(self.layers),rows=sorted(self.rows),native_calls=self.calls,fallbacks=self.fallbacks,artifact=self.spec['sha256'])
@lru_cache(maxsize=1)
def runtime():return Runtime()
class NativeSiluAndFp8(torch.nn.Module):
    def __init__(self,original,scope,name):
        super().__init__();self.original=original;self.scope=scope;self.layer_name=name;self.native=runtime()
    def forward(self,x):
        good=(x.ndim==2 and x.shape[1]==34816 and 1<=x.shape[0]<=8192 and x.dtype==torch.bfloat16 and x.is_contiguous() and x.device==self.scope.error.device)
        padding=self.scope.padding
        good=good and (padding is None or (padding.device==x.device and padding.dtype in (torch.bool,torch.uint8) and padding.ndim==1 and padding.is_contiguous() and padding.numel()>=x.shape[0]))
        if not good:self.native.fallbacks+=1;return self.original(x)
        m=x.shape[0];out=torch.empty((m,17408),device=x.device,dtype=torch.bfloat16);codes=torch.empty((m,17408),device=x.device,dtype=torch.uint8);scale=torch.empty(m,device=x.device,dtype=torch.float32)
        error=self.scope.profiling_error if self.scope.profiling else self.scope.error
        _checked(self.native.call(x.data_ptr(),out.data_ptr(),codes.data_ptr(),scale.data_ptr(),0 if padding is None else padding.data_ptr(),error.data_ptr(),m,17408,torch.cuda.current_stream(x.device).cuda_stream))
        # The verified MLP passes this exact tensor directly to its own native
        # down projection. Tensor-local ownership avoids a pointer-address cache.
        out._paiton_prequantized=(self.scope,codes,scale)
        self.native.layers.add(self.layer_name);self.native.rows.add(m);self.native.calls+=1
        return out

def bind(model,scope):
    count=0
    for name,layer in tuple(model.named_modules()):
        if name.endswith('.mlp') and getattr(getattr(layer,'gate_up_proj',None),'paiton_mxfp4_shape',None)==(34816,5120) and getattr(getattr(layer,'down_proj',None),'paiton_mxfp4_shape',None)==(5120,17408):
            if not isinstance(layer.act_fn,NativeSiluAndFp8):layer.act_fn=NativeSiluAndFp8(layer.act_fn,scope,name)
            count+=1
    if count!=64:raise ValueError(f'Expected64 fused native MLP producers, found{count}')
    scope.silu_prequantized_consumptions=0
