"""Stream-safe bindings for Wan TI2V's compiled normalization profile."""
import ctypes
import os
from pathlib import Path
import torch


class Fusions:
    def __init__(self, artifact):
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('Paiton Wan artifacts require an AMD ROCm GPU')
        if torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(':')[0] != 'gfx1201':
            raise RuntimeError('This Wan artifact is compiled for gfx1201')
        self.lib = ctypes.CDLL(str(Path(artifact).resolve()), mode=os.RTLD_LOCAL | os.RTLD_DEEPBIND)
        if self.lib.PaitonWan22FusionGetAbiVersion() != 1:
            raise ValueError('Unsupported Wan fusion artifact ABI')
        self.calls = 0
        self.run = self.lib.PaitonWan22FusionRun
        self.run.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*5 + [ctypes.c_void_p]
        self.run.restype = ctypes.c_int
        status = self.lib.PaitonWan22FusionInitialize()
        if status:
            raise RuntimeError(f'Wan artifact initialization failed: HIP {status}')

    def __call__(self, x, e, part, y=None):
        if x.ndim != 3 or x.shape[-1] != 3072 or x.shape[0] not in (1, 2):
            raise ValueError('Expected [B,S,3072] with B in {1,2}')
        if e.ndim != 4 or e.shape[0] != x.shape[0] or tuple(e.shape[2:]) != (6,3072):
            raise ValueError('Expected modulation [B,T,6,3072]')
        if x.dtype not in (torch.bfloat16,torch.float16) or part not in (0,2,3,5):
            raise ValueError('Unsupported Wan fusion dtype or operation')
        for tensor in (x,e) + (() if y is None else (y,)):
            if tensor.device != x.device or not tensor.is_cuda or tensor.dtype != x.dtype or not tensor.is_contiguous():
                raise ValueError('Fusion inputs must be contiguous on one GPU with matching dtype')
        if part in (2,5) and (y is None or y.shape != x.shape):
            raise ValueError('Gated residual requires a matching residual branch')
        self.calls += 1
        out=torch.empty_like(x)
        status=self.run(x.data_ptr(),y.data_ptr() if y is not None else None,e.data_ptr(),out.data_ptr(),
                        x.shape[0],x.shape[1],e.shape[1],int(x.dtype==torch.float16),part,
                        torch.cuda.current_stream(x.device).cuda_stream)
        if status:
            raise RuntimeError(f'Wan fusion failed: HIP {status}')
        return out


def install(model, artifact):
    """Use Comfy's per-block replacement API; leave the base model intact."""
    import comfy.model_management as mm
    fusion=Fusions(artifact)
    patched=model.clone()
    patched.paiton_wan_fusions = fusion
    dm=patched.get_model_object('diffusion_model')
    if dm.dim != 3072 or len(dm.blocks) != 30 or dm.in_dim != 48:
        raise ValueError('This artifact supports Wan2.2 TI2V 5B only')
    for index,block in enumerate(dm.blocks):
        def wrapper(args,extra,block=block):
            options=args['transformer_options']
            if options.get('patches'):
                return extra['original_block'](args)
            x,e=args['img'],args['vec']
            if e.ndim != 4 or e.dtype != x.dtype:
                return extra['original_block'](args)
            e=(mm.cast_to(block.modulation,dtype=x.dtype,device=x.device).unsqueeze(0)+e).contiguous()
            x=x.contiguous()
            y=block.self_attn(fusion(x,e,0),args['pe'],transformer_options=options)
            x=fusion(x,e,2,y.contiguous())
            x=x+block.cross_attn(block.norm3(x),args['txt'],context_img_len=None,transformer_options=options)
            y=block.ffn(fusion(x.contiguous(),e,3))
            return {'img':fusion(x.contiguous(),e,5,y.contiguous())}
        patched.set_model_patch_replace(wrapper,'dit','double_block',index)
    return patched
