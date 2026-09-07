"""Runtime bindings for the qualified compiled integer projections."""
import ctypes
import json
from pathlib import Path
import torch
from .runtime.utils.lib_wrapper import MemLoader

_library = None
_launch = None
_prepare = None
PROFILES = {(m,n,k) for m in (512,4096) for n,k in ((3072,3072),(18432,3072),(3072,9216))} | {
    (4608,27648,3072), (4608,3072,12288)}


def quantize_rows(values):
    """Symmetric INT8 row quantization, retaining the input scale dtype."""
    scale = values.abs().amax(dim=-1, keepdim=True) / 127
    denominator = torch.where(scale == 0, torch.ones_like(scale), scale)
    integers = (values / denominator).round().clamp(-128, 127).to(torch.int8)
    return integers, scale.squeeze(-1)


@torch.library.custom_op("paiton_image::projection", mutates_args=())
def compiled_projection(a: torch.Tensor, weight: torch.Tensor, sa: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    if _launch is None:
        raise RuntimeError("Paiton projections have not been loaded")
    m,k = a.shape
    n,wk = weight.shape
    if (m,n,k) not in PROFILES or wk != k:
        raise ValueError("Projection shape is outside the qualified model profile")
    for tensor, dtype in ((a,torch.int8),(weight,torch.int8),(sa,torch.bfloat16),(sw,torch.bfloat16)):
        if tensor.dtype != dtype or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("Invalid projection tensor type or layout")
    if tuple(sa.shape) != (m,) or tuple(sw.shape) != (n,):
        raise ValueError("Invalid row scale shapes")
    result = torch.empty((m,n),dtype=torch.bfloat16,device=a.device)
    if any(t.device != a.device for t in (weight,sa,sw)):
        raise ValueError('Projection tensors must share a device')
    workspace = torch.empty_like(a)
    status = _launch(*(t.data_ptr() for t in (a,weight,sa,sw,result,workspace)),m,n,k,
                     torch.cuda.current_stream(a.device).cuda_stream)
    if status:
        raise RuntimeError(f"Paiton projection launch failed with HIP status {status}")
    return result


@compiled_projection.register_fake
def _fake_projection(a,weight,sa,sw):
    return torch.empty((a.shape[0],weight.shape[0]),dtype=torch.bfloat16,device=a.device)


class Int8Linear(torch.nn.Module):
    def __init__(self, out_features, in_features, device="meta"):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight",torch.empty((out_features,in_features),dtype=torch.int8,device=device))
        self.register_buffer("scale",torch.empty((out_features,),dtype=torch.bfloat16,device=device))
        self.prepared = False

    def prepare(self):
        if self.prepared:
            raise RuntimeError('Projection weights are already prepared')
        self.weight = prepare_weight(self.weight)
        self.prepared = True

    def forward(self, inputs):
        if not self.prepared:
            raise RuntimeError('Projection weights must be prepared before inference')
        rows = inputs.reshape(-1,self.in_features)
        quantized, scales = quantize_rows(rows)
        output = compiled_projection(quantized.contiguous(),self.weight,scales.contiguous(),self.scale)
        return output.reshape(*inputs.shape[:-1],self.out_features)


def install_projections(artifact):
    global _library,_launch,_prepare
    if _launch is not None:
        raise RuntimeError("Projections are already installed")
    artifact = Path(artifact).resolve()
    manifest = json.loads(artifact.with_suffix(".manifest.json").read_text())
    if manifest.get("projection_abi_version") != 2:
        raise RuntimeError("Paiton projection ABI v2 is required")
    _library = MemLoader(str(artifact))
    version = _library.lib.PaitonDiffusionProjectionGetAbiVersion
    version.restype = ctypes.c_uint
    if version() != 2:
        raise RuntimeError("Paiton projection binary ABI mismatch")
    _launch = _library.lib.PaitonDiffusionProjectionRun
    _launch.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int]*3 + [ctypes.c_void_p]
    _launch.restype = ctypes.c_int
    _prepare = _library.lib.PaitonDiffusionProjectionPrepareWeight
    _prepare.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*2 + [ctypes.c_void_p]
    _prepare.restype = ctypes.c_int
    initialize = _library.lib.PaitonDiffusionProjectionInitialize
    initialize.argtypes = []
    initialize.restype = ctypes.c_int
    if initialize():
        raise RuntimeError('Paiton projection initialization failed')


def prepare_weight(weight):
    """Prepare immutable weights once using the artifact's opaque format."""
    if _prepare is None:
        raise RuntimeError('Load the projection artifact first')
    if weight.ndim != 2 or not weight.is_cuda or not weight.is_contiguous() or weight.dtype != torch.int8:
        raise ValueError('Expected a contiguous GPU INT8 weight matrix')
    n,k = weight.shape
    if (n,k) not in {(n,k) for _,n,k in PROFILES}:
        raise ValueError('Unsupported projection weight shape')
    result = torch.empty_like(weight)
    status = _prepare(weight.data_ptr(),result.data_ptr(),n,k,torch.cuda.current_stream(weight.device).cuda_stream)
    if status:
        raise RuntimeError(f'Weight preparation failed with HIP status {status}')
    return result
