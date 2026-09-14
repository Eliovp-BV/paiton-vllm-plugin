"""Bindings for the qualified BF16 image decoder artifact."""
import ctypes
import json
from pathlib import Path
import torch
from .runtime.utils.lib_wrapper import MemLoader

_library = None
_conv = _norm = _attention = _workspace = None
_prepared_conv = _prepare_conv_weight = None
CONV_PROFILES = {
    (128,128,32,32,1),(128,128,32,512,3),(128,128,512,512,3),
    (256,256,512,512,3),(512,512,512,512,3),(512,512,512,256,3),
    (512,512,256,256,3),(512,512,512,256,1),
    (1024,1024,256,256,3),(1024,1024,256,128,3),
    (1024,1024,128,128,3),(1024,1024,256,128,1),(1024,1024,128,3,3)}
NORM_PROFILES = {(16384,512),(65536,512),(262144,512),(262144,256),(1048576,256),(1048576,128)}


def _validate(*tensors):
    if _library is None:
        raise RuntimeError('The decoder artifact has not been loaded')
    for t in tensors:
        if t.dtype != torch.bfloat16 or not t.is_cuda or not t.is_contiguous() or t.device != tensors[0].device:
            raise ValueError('Decoder operations require contiguous BF16 tensors on the same GPU')


def _check(status):
    if status:
        raise RuntimeError(f'Decoder launch failed with HIP status {status}')


@torch.library.custom_op('paiton_image::decoder_conv',mutates_args=())
def convolution(x:torch.Tensor,w:torch.Tensor,b:torch.Tensor,prepared:bool=False)->torch.Tensor:
    _validate(x,w,b)
    if x.ndim != 4 or w.ndim != 4 or x.shape[0] != 1:
        raise ValueError('Unsupported decoder convolution rank or batch size')
    _,h,width,ci=x.shape
    co,r,r2,wci=w.shape
    if (h,width,ci,co,r) not in CONV_PROFILES or r != r2 or ci != wci or b.shape != (co,):
        raise ValueError('Unsupported decoder convolution profile')
    output=torch.empty((1,h,width,co),device=x.device,dtype=x.dtype)
    launch=_prepared_conv if prepared else _conv
    _check(launch(*(t.data_ptr() for t in (x,w,b,output)),h,width,ci,co,r,torch.cuda.current_stream(x.device).cuda_stream))
    return output.permute(0,3,1,2)


@convolution.register_fake
def _fake_conv(x,w,b,prepared=False):
    return torch.empty((x.shape[0],x.shape[1],x.shape[2],w.shape[0]),device=x.device,dtype=x.dtype).permute(0,3,1,2)


def prepare_conv_weight(weight):
    _validate(weight)
    if weight.ndim!=4:
        raise ValueError('Unsupported decoder weight rank')
    co,r,r2,ci=weight.shape
    if r!=r2 or not any((ci,co,r)==p[2:] for p in CONV_PROFILES) or r!=3 or ci<256:
        raise ValueError('Unsupported prepared decoder weight profile')
    output=torch.empty_like(weight)
    _check(_prepare_conv_weight(weight.data_ptr(),output.data_ptr(),co,r,ci,torch.cuda.current_stream(weight.device).cuda_stream))
    return output


@torch.library.custom_op('paiton_image::decoder_norm',mutates_args=())
def group_norm(x:torch.Tensor,w:torch.Tensor,b:torch.Tensor,swish:bool)->torch.Tensor:
    _validate(x,w,b)
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError('Unsupported decoder normalization rank or batch size')
    _,h,width,c=x.shape
    if (h*width,c) not in NORM_PROFILES or w.shape != (c,) or b.shape != (c,):
        raise ValueError('Unsupported decoder normalization profile')
    size=_workspace(h*width,c)
    if not size:
        raise RuntimeError('The decoder artifact lacks this normalization profile')
    workspace=torch.empty(size,device=x.device,dtype=torch.uint8)
    output=torch.empty_like(x)
    _check(_norm(*(t.data_ptr() for t in (x,w,b,output,workspace)),h*width,c,int(swish),torch.cuda.current_stream(x.device).cuda_stream))
    return output.permute(0,3,1,2)


@group_norm.register_fake
def _fake_norm(x,w,b,swish):
    return torch.empty_like(x).permute(0,3,1,2)


@torch.library.custom_op('paiton_image::decoder_attention',mutates_args=())
def decoder_attention(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor)->torch.Tensor:
    _validate(q,k,v)
    if q.shape != k.shape or q.shape != v.shape or q.shape != (1,1,16384,512):
        raise ValueError('Unsupported decoder attention profile')
    output=torch.empty_like(q)
    _check(_attention(*(t.data_ptr() for t in (q,k,v,output)),torch.cuda.current_stream(q.device).cuda_stream))
    return output


@decoder_attention.register_fake
def _fake_attention(q,k,v):
    return torch.empty_like(q)


class DecoderAttentionProcessor:
    def __call__(self,attn,hidden_states,encoder_hidden_states=None,attention_mask=None,temb=None,*args,**kwargs):
        if encoder_hidden_states is not None or attention_mask is not None or temb is not None or args or kwargs:
            raise ValueError('The decoder supports unmasked self attention only')
        residual=hidden_states
        b,c,h,w=hidden_states.shape
        if (b,c,h,w) != (1,512,128,128):
            raise ValueError('The decoder supports the 1024-pixel generation profile only')
        hidden=hidden_states.view(b,c,h*w).transpose(1,2)
        if attn.group_norm is not None:
            hidden=attn.group_norm(hidden.transpose(1,2)).transpose(1,2)
        q=attn.to_q(hidden).view(1,1,16384,512)
        k=attn.to_k(hidden).view(1,1,16384,512)
        v=attn.to_v(hidden).view(1,1,16384,512)
        attended=decoder_attention(q.contiguous(),k.contiguous(),v.contiguous()).reshape(1,16384,512)
        hidden=attn.to_out[1](attn.to_out[0](attended))
        hidden=hidden.transpose(1,2).reshape(b,c,h,w)
        if attn.residual_connection:
            hidden=hidden+residual
        return hidden/attn.rescale_output_factor


class Conv(torch.nn.Module):
    def __init__(self,old):
        super().__init__()
        if old.stride != (1,1) or old.groups != 1 or old.dilation != (1,1) or old.bias is None or old.padding != (old.kernel_size[0]//2,)*2:
            raise ValueError('Unsupported decoder convolution configuration')
        weight=old.weight.detach().permute(0,2,3,1).contiguous()
        self.prepared=weight.shape[1]==3 and weight.shape[3]>=256
        self.register_buffer('weight',prepare_conv_weight(weight) if self.prepared else weight)
        self.register_buffer('bias',old.bias.detach())

    def forward(self,x):
        return convolution(x.permute(0,2,3,1).contiguous(),self.weight,self.bias,self.prepared)


class GroupNorm(torch.nn.Module):
    def __init__(self,old,swish):
        super().__init__()
        if old.num_groups != 32 or old.eps != 1e-6 or old.weight is None or old.bias is None:
            raise ValueError('Unsupported decoder normalization configuration')
        self.swish=swish
        self.register_buffer('weight',old.weight.detach())
        self.register_buffer('bias',old.bias.detach())

    def forward(self,x):
        flat=x.ndim==3
        if flat:
            x=x.unsqueeze(-1)
        output=group_norm(x.permute(0,2,3,1).contiguous(),self.weight,self.bias,self.swish)
        return output.squeeze(-1) if flat else output


def load_artifact(artifact):
    global _library,_conv,_norm,_attention,_workspace,_prepared_conv,_prepare_conv_weight
    if _library is not None:
        raise RuntimeError('The decoder artifact is already loaded')
    artifact=Path(artifact).resolve()
    manifest=json.loads(artifact.with_suffix('.manifest.json').read_text())
    if manifest.get('decoder_abi_version') != 2:
        raise RuntimeError('Paiton decoder ABI v2 is required')
    library=MemLoader(str(artifact))
    version=library.lib.PaitonDecoderGetAbiVersion
    version.argtypes=[];version.restype=ctypes.c_uint
    if version()!=2:
        raise RuntimeError('Decoder binary ABI mismatch')
    _conv=library.lib.PaitonDecoderConvRun
    _conv.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*5+[ctypes.c_void_p];_conv.restype=ctypes.c_int
    _prepared_conv=library.lib.PaitonDecoderConvPreparedRun
    _prepared_conv.argtypes=_conv.argtypes;_prepared_conv.restype=ctypes.c_int
    _prepare_conv_weight=library.lib.PaitonDecoderPrepareConvWeight
    _prepare_conv_weight.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]*3+[ctypes.c_void_p];_prepare_conv_weight.restype=ctypes.c_int
    _norm=library.lib.PaitonDecoderNormRun
    _norm.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*3+[ctypes.c_void_p];_norm.restype=ctypes.c_int
    _attention=library.lib.PaitonDecoderAttentionRun
    _attention.argtypes=[ctypes.c_void_p]*5;_attention.restype=ctypes.c_int
    _workspace=library.lib.PaitonDecoderNormWorkspaceBytes
    _workspace.argtypes=[ctypes.c_int]*2;_workspace.restype=ctypes.c_ulonglong
    initialize=library.lib.PaitonDecoderInitialize
    initialize.argtypes=[];initialize.restype=ctypes.c_int
    _check(initialize())
    _library=library


def install_decoder(vae,artifact):
    load_artifact(artifact)
    for attn in vae.decoder.mid_block.attentions:
        if attn.heads != 1 or attn.spatial_norm is not None or attn.norm_q is not None or attn.norm_k is not None or attn.to_q.in_features != 512 or attn.to_q.out_features != 512:
            raise ValueError('Unsupported decoder attention configuration')
        attn.set_processor(DecoderAttentionProcessor())
    for name,mod in vae.named_modules():
        if name.startswith('decoder.') and hasattr(mod,'norm1') and hasattr(mod,'norm2') and hasattr(mod,'nonlinearity'):
            if not isinstance(mod.nonlinearity,torch.nn.SiLU):
                raise ValueError('Unsupported decoder activation')
            mod.nonlinearity=torch.nn.Identity()
    if not isinstance(vae.decoder.conv_act,torch.nn.SiLU):
        raise ValueError('Unsupported decoder output activation')
    vae.decoder.conv_act=torch.nn.Identity()
    for name,mod in list(vae.named_modules()):
        if isinstance(mod,(torch.nn.Conv2d,torch.nn.GroupNorm)) and (name.startswith('decoder.') or name=='post_quant_conv'):
            if '.' in name:
                parent,child=name.rsplit('.',1);parent=vae.get_submodule(parent)
            else:
                parent,child=vae,name
            swish=name.endswith(('.norm1','.norm2','.conv_norm_out'))
            setattr(parent,child,Conv(mod) if isinstance(mod,torch.nn.Conv2d) else GroupNorm(mod,swish))
