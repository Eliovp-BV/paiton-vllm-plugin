"""Experimental native MXFP4 integration through upstream vLLM's kernel API.

The external adapter owns vLLM tensors and passes their pointers/current stream
to allowlisted HIP artifacts. Quantization, layout conversion and GEMM execute
in those artifacts. The proprietary compiler is never imported or packaged.

Initial qualification uses eager execution and synchronously checks the native
quantization error flag. This is an integration candidate, not a speed claim.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.kernels.linear import register_linear_kernel
from vllm.model_executor.kernels.linear.mxfp4.base import (
    MxFp4LinearKernel,
    MxFp4LinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.platforms import PlatformEnum, current_platform


def _checked(status: int) -> None:
    if status:
        raise RuntimeError(f"Paiton native MXFP4 HIP status {status}")


class _Runtime:
    def __init__(self, path: str):
        manifest_path = Path(path).resolve(strict=True)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("abi") != 1 or manifest.get("arch") != "gfx1201":
            raise ValueError("Unsupported native MXFP4 manifest ABI/architecture")
        if manifest.get("arithmetic") != "e4m3_folded_magnitude_ties_to_zero":
            raise ValueError("Native MXFP4 arithmetic contract mismatch")
        required = {"layout", "quantization", "projection"}
        optional = {"projection_rows", "projection_pipeline"}
        names = set(manifest.get("artifacts", {}))
        if names - {'prefill'} not in (required, required | optional):
            raise ValueError("Native MXFP4 manifest requires base artifacts, optionally both tuned projection artifacts and native prefill")
        self.libraries = {}
        for name, entry in manifest["artifacts"].items():
            binary = (manifest_path.parent / entry["file"]).resolve(strict=True)
            if binary.parent != manifest_path.parent or binary.suffix != ".so":
                raise ValueError("Native MXFP4 artifact must be a local sibling .so")
            if hashlib.sha256(binary.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Native MXFP4 {name} artifact hash mismatch")
            self.libraries[name] = ctypes.CDLL(str(binary), mode=ctypes.RTLD_LOCAL)
        prefixes = {"layout": "paiton_mxfp4_layout", "quantization": "paiton_row_fp8_quant",
                    "projection": "paiton_fp8", "projection_rows": "paiton_fp8",
                    "projection_pipeline": "paiton_fp8", "prefill": "paiton_fp8_prefill"}
        for name in self.libraries:
            prefix = prefixes[name]
            library = self.libraries[name]
            abi = getattr(library, prefix + "_abi_version")
            abi.restype = ctypes.c_int
            arch = getattr(library, prefix + "_arch")
            arch.restype = ctypes.c_char_p
            if abi() != 1 or arch() != b"gfx1201":
                raise ValueError(f"Native MXFP4 {name} binary ABI mismatch")
        self.layout = self.libraries["layout"].paiton_mxfp4_layout
        self.layout.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
        self.quantize = self.libraries["quantization"].paiton_row_fp8_quant
        self.quantize.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        self.quantize_masked = getattr(self.libraries['quantization'], 'paiton_row_fp8_quant_masked', None)
        if self.quantize_masked is not None:
            self.quantize_masked.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
            self.quantize_masked.restype = ctypes.c_int
        self.project = self.libraries["projection"].paiton_fp8_direct
        self.project.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        self.tuned_projects = {}
        for name in optional & names:
            function = self.libraries[name].paiton_fp8_direct
            function.argtypes = self.project.argtypes
            function.restype = ctypes.c_int
            self.tuned_projects[name] = function
        self.decode_routes = manifest.get('decode_routes', [])
        covered = set()
        for route in self.decode_routes:
            n,k,lo,hi = (route[key] for key in ('n','k','min_m','max_m'))
            nw,sk,nt = route['config']
            if (route['artifact'] not in self.tuned_projects or (lo,hi) not in ((1,16),(17,32),(33,64))
                    or nw not in (1,2,4) or sk not in (1,2,4,8) or nt not in (0,1)
                    or n<16 or n%(16*nw) or k<32 or k%(32*sk) or (n,k,lo,hi) in covered):
                raise ValueError('Unsupported or overlapping native projection route')
            covered.add((n,k,lo,hi))
        for function in (self.layout, self.quantize, self.project):
            function.restype = ctypes.c_int
        self.prefill_min_rows = {}
        for route in manifest.get('prefill_routes', []):
            shape=(route['n'],route['k']);minimum=route['min_m']
            if shape not in ((16384,5120),(14336,5120),(34816,5120),(5120,17408)) or minimum!=96 or shape in self.prefill_min_rows:
                raise ValueError('Unsupported native prefill route')
            self.prefill_min_rows[shape]=minimum
        self.prefill = None
        if 'prefill' in names:
            self.prefill = self.libraries['prefill'].paiton_fp8_prefill
            self.prefill.argtypes = [ctypes.c_void_p]*6+[ctypes.c_int]*3+[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
            self.prefill.restype = ctypes.c_int
            self.prefill_workspace_size = self.libraries['prefill'].paiton_fp8_prefill_workspace_size
            self.prefill_workspace_size.argtypes = [ctypes.c_int]*2
            self.prefill_workspace_size.restype = ctypes.c_size_t
        self.hip = ctypes.CDLL("libamdhip64.so.7", mode=ctypes.RTLD_LOCAL)
        self.hip.hipMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
        self.hip.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                          ctypes.c_int, ctypes.c_void_p]
        self.hip.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
        for name in ("hipMemsetAsync", "hipMemcpyAsync", "hipStreamSynchronize"):
            getattr(self.hip, name).restype = ctypes.c_int

    def clear_error(self, error: torch.Tensor, stream: int) -> None:
        _checked(self.hip.hipMemsetAsync(error.data_ptr(), 0, 4, stream))

    def projection_for(self, m: int, n: int, k: int):
        for route in self.decode_routes:
            if (n,k)==(route['n'],route['k']) and route['min_m']<=m<=route['max_m']:
                name=route['artifact'];config=tuple(route['config'])
                return self.tuned_projects[name],config,name+':'+','.join(map(str,config))
        # Reuse the measured native dispatch from the qualified hybrid. Both
        # artifacts consume the same fragment layout produced by native layout.
        if self.tuned_projects and (n, k) == (34816, 5120) and m <= 64:
            name = "projection_rows" if m <= 32 else "projection_pipeline"
            return self.tuned_projects[name], (1 if m <= 32 else 2, 1, 1), name
        return self.project, (1, 1, 1), "projection"

    def check_error(self, error: torch.Tensor, stream: int) -> None:
        value = ctypes.c_int()
        _checked(self.hip.hipMemcpyAsync(ctypes.byref(value), error.data_ptr(), 4, 2, stream))
        _checked(self.hip.hipStreamSynchronize(stream))
        if value.value:
            raise ValueError(f"Paiton native MXFP4 rejected nonfinite activation or E8M0 scale (flag={value.value})")


class PaitonNativeMxFp4LinearKernel(MxFp4LinearKernel):
    """Explicit W4A8 research profile; weights retain original E2M1 codes."""

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_rocm():
            return False, "Native MXFP4 requires ROCm"
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        if properties.gcnArchName.split(":", 1)[0] != "gfx1201":
            return False, "Native MXFP4 is qualified only for gfx1201"
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig):
        if os.getenv("PAITON_EXPERIMENTAL_MXFP4_W4A8") != "1":
            return False, "Explicit W4A8 experimental profile is required"
        if config.activation_quant_key != kMxfp4Dynamic:
            return False, "Expected the checkpoint's dynamic MXFP4 activation declaration"
        if not os.getenv("PAITON_MXFP4_RUNTIME_MANIFEST"):
            return False, "Native MXFP4 runtime manifest is required"
        return True, None

    def process_weights_after_loading(self, layer):
        config = get_current_vllm_config()
        if not config.model_config.enforce_eager and os.environ.get("PAITON_EXPERIMENTAL_MXFP4_FORWARD_ERROR_SCOPE") != "1":
            raise ValueError("Native MXFP4 integration qualification currently requires --enforce-eager")
        if config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("Native MXFP4 integration qualification requires tensor parallel size1")
        weight, scales = layer.weight, layer.weight_scale
        if (weight.dtype != torch.uint8 or scales.dtype != torch.uint8 or weight.ndim != 2
                or not weight.is_contiguous() or not scales.is_contiguous()
                or weight.device != scales.device or weight.device.type != "cuda"):
            raise ValueError("Native MXFP4 expects contiguous device U8 weights/scales")
        n, half_k = weight.shape
        k = half_k * 2
        if n % 16 or not 16 <= n <= 262144 or k % 32 or not 32 <= k <= 32768:
            raise ValueError("Native MXFP4 weight geometry is unsupported")
        if tuple(scales.shape) != (n, k // 32):
            raise ValueError("Native MXFP4 group32 scale shape mismatch")
        runtime = _Runtime(os.environ["PAITON_MXFP4_RUNTIME_MANIFEST"])
        packed = torch.empty_like(weight)
        transposed = torch.empty((k // 32, n), dtype=torch.uint8, device=weight.device)
        refs = torch.empty(n, dtype=torch.uint8, device=weight.device)
        error = torch.empty(1, dtype=torch.int32, device=weight.device)
        stream = torch.cuda.current_stream(weight.device).cuda_stream
        runtime.clear_error(error, stream)
        _checked(runtime.layout(weight.data_ptr(), scales.data_ptr(), packed.data_ptr(),
                                transposed.data_ptr(), refs.data_ptr(), error.data_ptr(), n, k, stream))
        runtime.check_error(error, stream)
        layer.weight = torch.nn.Parameter(packed, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(transposed, requires_grad=False)
        layer.register_buffer("paiton_row_refs", refs)
        layer.register_buffer("paiton_quant_error", error)
        # Keep library handles alive as long as their device kernels may execute.
        layer.paiton_mxfp4_runtime = runtime
        layer.paiton_mxfp4_shape = (n, k)
        layer.paiton_projection_profiles = set()

    def apply_weights(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("Native MXFP4 qualification covers bias-free projections")
        n, k = layer.paiton_mxfp4_shape
        if (x.dtype != torch.bfloat16 or x.device != layer.weight.device
                or not x.is_contiguous() or x.shape[-1] != k):
            raise ValueError("Native MXFP4 input must be contiguous device BF16")
        scope = getattr(layer, "paiton_forward_error_scope", None)
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and scope is None:
            raise RuntimeError("Synchronous error-checking candidate cannot execute under graph capture")
        flat = x.view(-1, k)
        m = flat.shape[0]
        if not 1 <= m <= 8192:
            raise ValueError("Native MXFP4 input row bound exceeded")
        output = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        prepared = getattr(x, '_paiton_prequantized', None)
        producer='silu'
        if prepared is None:
            prepared=getattr(x,'_paiton_rms_prequantized',None);producer='rms'
        if prepared is not None:
            shape_ok=((n,k)==(5120,17408) if producer=='silu' else k==5120 and n in (96,14336,16384,34816))
            if (scope is None or not isinstance(prepared, tuple) or len(prepared)!=3 or prepared[0] is not scope
                    or not shape_ok or not all(isinstance(t,torch.Tensor) for t in prepared[1:])):
                raise ValueError('Native prequantized producer contract mismatch')
            _, quantized, activation_scale = prepared
            if (quantized.dtype!=torch.uint8 or quantized.shape!=(m,k) or not quantized.is_contiguous()
                    or activation_scale.dtype!=torch.float32 or activation_scale.shape!=(m,) or not activation_scale.is_contiguous()
                    or quantized.device!=x.device or activation_scale.device!=x.device):
                raise ValueError('Native prequantized buffer contract mismatch')
            if producer=='silu':scope.silu_prequantized_consumptions+=1
            else:scope.rms_prequantized_consumptions+=1
            layer.paiton_projection_profiles.add(producer+'_prequantized')
        else:
            quantized = torch.empty((m, k), dtype=torch.uint8, device=x.device)
            activation_scale = torch.empty(m, dtype=torch.float32, device=x.device)
        runtime = layer.paiton_mxfp4_runtime
        stream = torch.cuda.current_stream(x.device).cuda_stream
        if scope is None:
            runtime.clear_error(layer.paiton_quant_error, stream)
        elif capturing:
            scope.capture_rows.add(m)
        error = scope.profiling_error if scope is not None and scope.profiling else layer.paiton_quant_error
        padding = getattr(scope, 'padding', None)
        pointers = (flat.data_ptr(), quantized.data_ptr(), activation_scale.data_ptr(), error.data_ptr())
        if prepared is not None:
            pass  # The native producer already applied the current padding/error contract.
        elif padding is not None:
            if (padding.device != x.device or padding.dtype not in (torch.bool, torch.uint8)
                    or padding.ndim != 1 or not padding.is_contiguous() or padding.numel() < m):
                raise ValueError('Native MXFP4 padding requires a contiguous device byte mask covering every row')
            if runtime.quantize_masked is None:
                raise ValueError('This vLLM graph profile requires the native masked quantization artifact')
            _checked(runtime.quantize_masked(*pointers, padding.data_ptr(), m, k, 0, 0, stream))
        else:
            _checked(runtime.quantize(*pointers, m, k, 0, 0, stream))
        if os.getenv("PAITON_NATIVE_DIAG_ERROR") == "1" and scope is not None and not scope.profiling:
            if capturing:
                raise RuntimeError("Per-projection diagnostic must finish before capture")
            try:
                scope.check()
            except ValueError as cause:
                from vllm.forward_context import get_forward_context
                metadata = get_forward_context().attn_metadata
                actual = sorted({getattr(value, "num_actual_tokens", -1) for value in metadata.values()}) if isinstance(metadata, dict) else None
                bits = flat.view(torch.uint16).cpu().numpy()
                bad = (bits & 0x7f80) == 0x7f80
                bad_rows = bad.any(axis=1).nonzero()[0].tolist()
                raise ValueError(f"Native quantization diagnostic layer={getattr(layer, 'paiton_layer_name', '?')} M={m} N={n} K={k} actual_tokens={actual} nonfinite_rows={bad_rows}") from cause
        if scope is None:
            runtime.check_error(layer.paiton_quant_error, stream)
        if runtime.prefill is not None and m >= runtime.prefill_min_rows.get((n,k),128) and n % 64 == 0 and k % 128 == 0:
            scratch = getattr(scope, 'prefill_scratch', None)
            if scratch is None:
                raise RuntimeError('Native tiled prefill requires model-owned shared scratch')
            _checked(runtime.prefill(quantized.data_ptr(),layer.weight.data_ptr(),layer.weight_scale.data_ptr(),
                layer.paiton_row_refs.data_ptr(),activation_scale.data_ptr(),output.data_ptr(),m,n,k,
                scratch.data_ptr(),scratch.numel(),stream))
            layer.paiton_projection_profiles.add('tiled_prefill')
            return output.view(*x.shape[:-1],n)
        project, config, profile = runtime.projection_for(m, n, k)
        layer.paiton_projection_profiles.add(profile)
        _checked(project(quantized.data_ptr(), layer.weight.data_ptr(), layer.weight_scale.data_ptr(),
                                 layer.paiton_row_refs.data_ptr(), activation_scale.data_ptr(), output.data_ptr(),
                                 m, n, k, *config, stream))
        return output.view(*x.shape[:-1], n)


_REGISTERED = False


def register_paiton_mxfp4() -> None:
    global _REGISTERED
    if not _REGISTERED:
        if os.environ.get('PAITON_EXPERIMENTAL_ATTENTION') == '1':
            from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
            register_backend(AttentionBackendEnum.CUSTOM,
                'paiton_vllm_plugin.attention_native.PaitonAttentionBackend')
        if os.environ.get('PAITON_EXPERIMENTAL_GDN_REPLAY') == '1':
            from . import gdn_native_replay  # noqa: F401; public pluggable layer registration
        from . import mxfp4_native_loader  # noqa: F401; public vLLM loader registration
        if os.environ.get("PAITON_EXPERIMENTAL_MXFP4_FORWARD_ERROR_SCOPE") == "1":
            from vllm import ModelRegistry
            ModelRegistry.register_model("Qwen3_5ForConditionalGeneration",
                "paiton_vllm_plugin.mxfp4_native_model:PaitonQwen3_5ForConditionalGeneration")
        if os.environ.get("PAITON_EXPERIMENTAL_DFLASH_CONTEXT") == "1":
            from vllm import ModelRegistry
            ModelRegistry.register_model("DFlash2DraftModel",
                "paiton_vllm_plugin.dflash2_native_context:PaitonDFlash2ForCausalLM")
        register_linear_kernel(PaitonNativeMxFp4LinearKernel, PlatformEnum.ROCM, kernel_type="mxfp4")
        _REGISTERED = True
