"""Scoped compiled Wan VAE normalization; stock convolution and caching remain."""
from contextlib import contextmanager
import copy
import ctypes
import torch
from .fusions import Fusions

class VaeNorm:
    def __init__(self,artifact):
        self.artifact=Fusions(artifact)
        lib=self.artifact.lib
        if lib.PaitonWan22VaeNormGetAbiVersion()!=1:raise ValueError('Unsupported VAE normalization ABI')
        self.run=lib.PaitonWan22VaeNormRun
        self.run.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
        self.run.restype=ctypes.c_int
        self.calls=0
    def __call__(self,x,gamma):
        if x.dtype!=torch.bfloat16 or not x.is_cuda or not x.is_contiguous() or x.ndim not in (4,5) or x.shape[0]!=1 or x.shape[1] not in (256,512,1024):
            raise ValueError('Expected contiguous BF16 Wan VAE activation with batch one')
        c=x.shape[1];positions=x.numel()//c
        if gamma.numel()!=c or positions>2097152:raise ValueError('Unsupported normalization shape')
        g=gamma.to(x).contiguous();out=torch.empty_like(x)
        status=self.run(x.data_ptr(),g.data_ptr(),out.data_ptr(),c,positions,torch.cuda.current_stream(x.device).cuda_stream)
        if status:raise RuntimeError(f'VAE normalization failed: HIP {status}')
        self.calls+=1
        return out

@contextmanager
def scope(vae,fusion):
    saved=[]
    try:
        for seq in vae.first_stage_model.modules():
            if not isinstance(seq,torch.nn.Sequential):continue
            children=list(seq.children())
            for norm,activation in zip(children,children[1:]):
                if type(norm).__name__!='RMS_norm' or not isinstance(activation,torch.nn.SiLU):continue
                if not norm.channel_first or norm.bias is not None or norm.gamma.numel() not in (256,512,1024):continue
                original_norm,original_activation=norm.forward,activation.forward
                active=[False]
                def norm_forward(x,_norm=norm,_original=original_norm,_active=active):
                    _active[0]=(x.dtype==torch.bfloat16 and x.is_cuda and x.is_contiguous() and x.shape[0]==1 and x.numel()//x.shape[1]<=2097152)
                    return fusion(x,_norm.gamma) if _active[0] else _original(x)
                def activation_forward(x,_original=original_activation,_active=active):
                    return x if _active[0] else _original(x)
                saved.extend([(norm,original_norm),(activation,original_activation)])
                norm.forward=norm_forward;activation.forward=activation_forward
        yield
    finally:
        for module,forward in reversed(saved):module.forward=forward


def install(vae,artifact):
    """Return a wrapper; restore all module methods after each serialized decode."""
    result=copy.copy(vae);fusion=VaeNorm(artifact);result.paiton_wan_vae=fusion
    for name in ('decode','decode_tiled'):
        original=getattr(vae,name)
        def decode(*args,_original=original,**kwargs):
            with scope(vae,fusion):return _original(*args,**kwargs)
        setattr(result,name,decode)
    return result
