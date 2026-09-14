"""H3 projection artifact binding; preserve stock ConvRot quantization."""
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import threading

import torch

PROFILES = {(21504,5376), (28672,5376), (5376,7168), (5376,14336)}
max_tokens = 32768
_library = None
_launch = None
_packed_launch = None
_local = threading.local()
_original = None
calls = 0


@torch.library.custom_op("paiton_video::h3_projection", mutates_args=())
def compiled_projection(a: torch.Tensor, w: torch.Tensor, sa: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    if _launch is None:
        raise RuntimeError("Load the H3 projection artifact before inference")
    if a.ndim != 2 or w.ndim != 2:
        raise ValueError("Expected A[M,K] and W[N,K]")
    m,k=a.shape
    n,wk=w.shape
    if not 1 <= m <= max_tokens or wk != k or (n,k) not in PROFILES:
        raise ValueError("Unsupported H3 projection shape")
    for t,dtype in ((a,torch.int8),(w,torch.int8),(sa,torch.float32),(sw,torch.float32)):
        if t.dtype != dtype or not t.is_cuda or not t.is_contiguous() or t.device != a.device or t.data_ptr() % 16:
            raise ValueError("H3 projections require aligned contiguous tensors on one GPU")
    if sa.shape != (m,) or sw.shape != (n,):
        raise ValueError("Expected one FP32 scale per input/output row")
    out=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
    stream=torch.cuda.current_stream(a.device).cuda_stream
    if _packed_launch is not None and m >= 128:
        workspace_bytes=n*k+(((m+127)//128)*128*k if n > 5376 else 0)
        workspace=torch.empty(workspace_bytes,device=a.device,dtype=torch.uint8)
        status=_packed_launch(*(t.data_ptr() for t in (a,w,sa,sw,out)),
                              workspace.data_ptr(),workspace_bytes,m,n,k,stream)
    else:
        status=_launch(*(t.data_ptr() for t in (a,w,sa,sw,out)),m,n,k,stream)
    if status:
        raise RuntimeError(f"H3 projection failed with HIP status {status}")
    return out


@compiled_projection.register_fake
def _fake(a,w,sa,sw):
    return torch.empty((a.shape[0],w.shape[0]),device=a.device,dtype=torch.bfloat16)


def install_projections(artifact, packed=True):
    global _library,_launch,_packed_launch,_original,max_tokens
    if _launch is not None:
        raise RuntimeError("H3 projections are already installed")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1201":
        raise RuntimeError("This H3 artifact requires gfx1201")
    artifact=Path(artifact).resolve()
    manifest=json.loads(artifact.with_suffix(".manifest.json").read_text())
    if manifest.get("h3_projection_abi_version") != 1:
        raise RuntimeError("H3 projection ABI v1 is required")
    if manifest.get("target",{}).get("arch") != "gfx1201":
        raise RuntimeError("Artifact target must be gfx1201")
    with artifact.open("rb") as binary:
        digest=hashlib.file_digest(binary,"sha256").hexdigest()
    if digest != manifest.get("artifact",{}).get("sha256"):
        raise RuntimeError("H3 projection artifact SHA-256 mismatch")
    mode=os.RTLD_NOW | os.RTLD_LOCAL | getattr(os,"RTLD_DEEPBIND",0)
    _library=ctypes.CDLL(str(artifact),mode=mode)
    version=_library.PaitonH3ProjectionGetAbiVersion
    version.restype=ctypes.c_uint
    if version() != 1:
        raise RuntimeError("H3 projection binary ABI mismatch")
    _launch=_library.PaitonH3ProjectionRun
    _launch.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*3+[ctypes.c_void_p]
    _launch.restype=ctypes.c_int
    if packed and hasattr(_library,"PaitonH3PackedProjectionRun"):
        _packed_launch=_library.PaitonH3PackedProjectionRun
        _packed_launch.argtypes=[ctypes.c_void_p]*6+[ctypes.c_size_t]+[ctypes.c_int]*3+[ctypes.c_void_p]
        _packed_launch.restype=ctypes.c_int
    if _packed_launch is not None and hasattr(_library, "PaitonH3GetMaxTokens"):
        limit = _library.PaitonH3GetMaxTokens
        limit.restype = ctypes.c_uint
        max_tokens = limit()
        if max_tokens != manifest.get("h3_max_tokens") or not 32768 <= max_tokens <= 49152:
            raise RuntimeError("H3 artifact token limit mismatch")
    from comfy_kitchen.backends import hip
    _original=hip.int8_linear
    def linear(x,weight,weight_scale,bias=None,out_dtype=torch.bfloat16,convrot=False,convrot_groupsize=256,input_act=None):
        global calls
        m=x.numel()//x.shape[-1]
        eligible=(getattr(_local,"enabled",False) and convrot and convrot_groupsize==256
                  and out_dtype==torch.bfloat16 and x.dtype==torch.bfloat16 and bias is None
                  and weight.ndim==2 and tuple(weight.shape) in PROFILES and 1<=m<=max_tokens
                  and input_act in (None,"swiglu") and weight_scale.numel()==weight.shape[0])
        if not eligible:
            return _original(x,weight,weight_scale,bias,out_dtype,convrot,convrot_groupsize,input_act)
        # The pinned HIP quantizer includes the same optional SwiGLU, rotation,
        # reduction and rounding as stock. Only its following GEMM is replaced.
        weight=hip._aligned(weight.to(device=x.device).contiguous())
        sw=weight_scale.to(device=x.device,dtype=torch.float32).reshape(-1)
        q,sa=hip._rotate_quant_int8(x.reshape(m,x.shape[-1]).contiguous(),convrot_groupsize,input_act)
        result=compiled_projection(q,weight,sa.reshape(-1).contiguous(),sw)
        calls+=1
        return result.reshape(*x.shape[:-1],weight.shape[0])
    hip.int8_linear=linear
    original_w4a8=hip.w4a8_int8_linear
    def w4a8_linear(x,qdata,s_rel,s_channel,codebook=None,correction=None,bias=None,group_size=16,convrot_groupsize=256,out_dtype=torch.bfloat16):
        global calls
        m=x.numel()//x.shape[-1]
        shape=(qdata.shape[0],qdata.shape[1]*2) if qdata.ndim==2 else None
        eligible=(getattr(_local,"enabled",False) and shape in PROFILES and 1<=m<=max_tokens
                  and x.dtype==torch.bfloat16 and out_dtype==torch.bfloat16
                  and correction is None and bias is None and group_size==16 and convrot_groupsize==256)
        if not eligible:
            return original_w4a8(x,qdata,s_rel,s_channel,codebook,correction,bias,group_size,convrot_groupsize,out_dtype)
        hip.validate_w4a8_operands(qdata,s_rel,s_channel,codebook,correction,group_size,convrot_groupsize)
        if x.shape[-1] != shape[1]:
            raise ValueError("W4A8 input and weight dimensions differ")
        # Same stock codebook/FP8-scale decode to the same INT8 grid. The
        # workspace lasts one projection; no permanent INT8 weight expansion.
        q,sa=hip._rotate_quant_int8(x.reshape(m,shape[1]).contiguous(),convrot_groupsize)
        weight=hip._dequant_int4_grouped_to_int8(qdata,s_rel,codebook,group_size)
        sw=s_channel.to(device=x.device,dtype=torch.float32).reshape(-1).contiguous()
        result=compiled_projection(q,weight,sa.reshape(-1).contiguous(),sw)
        calls+=1
        return result.reshape(*x.shape[:-1],shape[0])
    hip.w4a8_int8_linear=w4a8_linear
    if hasattr(_library,"PaitonH3FusionGetAbiVersion"):
        from . import fusions
        import sys
        fusions.initialize(sys.modules[__name__])


@contextmanager
def projection_scope():
    """Enable only around this thread's H3 denoiser; all other calls use stock."""
    previous=getattr(_local,"enabled",False)
    _local.enabled=True
    try:
        yield
    finally:
        _local.enabled=previous
