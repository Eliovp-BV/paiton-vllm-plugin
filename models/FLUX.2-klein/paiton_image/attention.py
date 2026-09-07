"""Connect the qualified compiled artifact to the Diffusers pipeline."""
from pathlib import Path
import ctypes
import json
import torch

from .runtime.utils.lib_wrapper import MemLoader


_model = None
_launch = None
_installed = False
SHAPE = (1, 4608, 24, 128)
PACKED_SHAPE = (1, 24, 4608, 128)


@torch.library.custom_op("paiton_image::attention", mutates_args=())
def compiled_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       qs: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
    if _model is None:
        raise RuntimeError("Paiton artifact has not been loaded")
    for tensor, dtype in ((q, torch.int8), (k, torch.int8), (v, torch.bfloat16)):
        if tensor.numel() != 24*4608*128 or tensor.ndim!=6 or tensor.dtype != dtype or tensor.device.type != "cuda" or not tensor.is_contiguous() or tensor.device!=q.device:
            raise ValueError("Paiton requires the batch-one 1024-square INT8 QK / BF16 PV profile")
    for tensor in (qs, ks):
        if tuple(tensor.shape) != PACKED_SHAPE[:3] or tensor.dtype != torch.float32 or tensor.device != q.device:
            raise ValueError("Paiton attention requires FP32 per-row scales")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    output = torch.empty(PACKED_SHAPE,device=v.device,dtype=v.dtype)
    qs, ks = qs.contiguous(), ks.contiguous()
    status = _launch(*(tensor.data_ptr() for tensor in (q, k, v, qs, ks, output)),
                     torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(f"Paiton attention launch failed with HIP status {status}")
    return output


@compiled_attention.register_fake
def _fake_attention(q, k, v, qs, ks):
    return torch.empty(PACKED_SHAPE,device=v.device,dtype=v.dtype)


def install_paiton_attention(artifact):
    global _model, _launch, _installed
    if _installed:
        raise RuntimeError("Paiton attention is already installed in this process")
    properties = torch.cuda.get_device_properties(0)
    if properties.name != "AMD Radeon AI PRO R9700" or properties.gcnArchName.split(":")[0] != "gfx1201":
        raise RuntimeError("This Paiton release requires a Radeon AI PRO R9700 (gfx1201)")
    artifact = Path(artifact).resolve()
    manifest = json.loads(artifact.with_suffix(".manifest.json").read_text())
    if manifest.get("attention_abi_version") != 2:
        raise RuntimeError("Paiton attention ABI v2 is required")
    _model = MemLoader(str(artifact))
    version = _model.lib.PaitonDiffusionAttentionGetAbiVersion
    version.restype = ctypes.c_uint
    if version() != 2:
        raise RuntimeError("Paiton attention binary ABI mismatch")
    _launch = _model.lib.PaitonDiffusionAttentionRun
    _launch.argtypes = [ctypes.c_void_p] * 7
    _launch.restype = ctypes.c_int
    # Load the GPU code before any caller starts graph capture.
    initialize=_model.lib.PaitonDiffusionAttentionInitialize
    initialize.argtypes=[];initialize.restype=ctypes.c_int
    status=initialize()
    if status:
        raise RuntimeError(f"Paiton attention initialization failed with HIP status {status}")
    import diffusers.models.transformers.transformer_flux2 as flux2

    def dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, **kwargs):
        if attn_mask is not None or dropout_p or is_causal:
            raise ValueError("The qualified Paiton profile uses unmasked non-causal attention")
        if scale is not None and abs(scale - 128**-0.5) > 1e-12:
            raise ValueError("Unsupported Paiton attention scale")
        from .projection import quantize_rows
        q_float = query.transpose(1, 2).contiguous().float()
        k_float = key.transpose(1, 2).float()
        k_float = (k_float - k_float.mean(dim=2, keepdim=True)).contiguous()
        q, qs = quantize_rows(q_float)
        k, ks = quantize_rows(k_float)
        prepare=torch.ops.paiton_diffusion.prepare_attention_operand
        q,k=prepare(q[0]),prepare(k[0])
        v=prepare(value[0].permute(1,2,0))
        return compiled_attention(q,k,v,qs.contiguous(),ks.contiguous()).transpose(1,2)

    flux2.dispatch_attention_fn = dispatch
    _installed = True
