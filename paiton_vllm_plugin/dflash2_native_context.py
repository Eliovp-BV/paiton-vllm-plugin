"""Native context-weight preparation for the pinned FP8 DFlash2 checkpoint.

This external vLLM model extension owns tensors; the HIP artifact performs
E4M3/block-scale expansion. The upstream query path and context GEMM remain
inherited. Neither the compiler nor the artifact imports a framework.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.models.qwen3_dflash2 import (
    DFlash2Qwen3ForCausalLM,
    DFlash2Qwen3Model,
)


def _check(status):
    if status:
        raise RuntimeError(f"Paiton native context HIP status {status}")


class _BlockExpansion:
    def __init__(self):
        path = Path(os.environ["PAITON_DFLASH_CONTEXT_RUNTIME_MANIFEST"]).resolve(strict=True)
        manifest = json.loads(path.read_text())
        if (manifest.get("abi"), manifest.get("arch"), manifest.get("arithmetic")) != (
            1, "gfx1201", "e4m3fn_fp32_block128_scale_bf16_rne"
        ):
            raise ValueError("Unsupported native DFlash2 context contract")
        binary = (path.parent / manifest["file"]).resolve(strict=True)
        if binary.parent != path.parent or binary.suffix != ".so":
            raise ValueError("Expected sibling native context .so")
        if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("Native DFlash2 context artifact hash mismatch")
        self.lib = ctypes.CDLL(str(binary), mode=ctypes.RTLD_LOCAL)
        arch = self.lib.paiton_fp8_block_expand_arch
        arch.restype = ctypes.c_char_p
        if self.lib.paiton_fp8_block_expand_abi_version() != 1 or arch() != b"gfx1201":
            raise ValueError("Native DFlash2 context binary ABI mismatch")
        self.expand = self.lib.paiton_fp8_block_expand
        self.expand.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_void_p]
        self.expand.restype = ctypes.c_int
        self.hip = ctypes.CDLL("libamdhip64.so.7", mode=ctypes.RTLD_LOCAL)
        self.hip.hipMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
        self.hip.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self.hip.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.manifest = manifest


class PaitonDFlash2Model(DFlash2Qwen3Model):
    def _build_context_kv_buffers(self, layers_attn, has_bias):
        config = get_current_vllm_config()
        graph_candidate = os.environ.get("PAITON_EXPERIMENTAL_MXFP4_FORWARD_ERROR_SCOPE") == "1"
        if (not config.model_config.enforce_eager and not graph_candidate) or config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("Native DFlash2 context requires eager TP1 or the explicit forward-error qualification candidate")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Prepare DFlash2 context weights before graph capture")
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if props.gcnArchName.split(":", 1)[0] != "gfx1201":
            raise ValueError("Native DFlash2 context is qualified for gfx1201 only")
        if len(layers_attn) != 5 or has_bias:
            raise ValueError("Expected five bias-free DFlash2 attention layers")
        for attention in layers_attn:
            layer = attention.qkv_proj
            scale = getattr(layer, "weight_scale_inv", None)
            if (attention.q_size, attention.kv_size) != (4096, 1024):
                raise ValueError("Unexpected DFlash2 Q/KV partition sizes")
            if (tuple(layer.weight.shape) != (6144, 5120)
                or layer.weight.dtype != torch.float8_e4m3fn
                or not layer.weight.is_contiguous() or not layer.weight.is_cuda
                or scale is None or tuple(scale.shape) != (48, 40)
                or scale.dtype != torch.float32 or not scale.is_contiguous()
                or scale.device != layer.weight.device
                or tuple(layer.weight_block_size) != (128, 128)):
                raise ValueError("Unsupported DFlash2 FP8 weight/scale layout")
        # Keep upstream normalization/bias buffers and their lifecycle intact.
        super()._build_context_kv_buffers(layers_attn, has_bias)
        if self._hidden_norm_weight.dtype != torch.bfloat16:
            raise ValueError("Native DFlash2 context requires BF16 activations")
        runtime = _BlockExpansion()
        device = layers_attn[0].qkv_proj.weight.device
        output = torch.empty((10240, 5120), dtype=torch.bfloat16, device=device)
        error = torch.empty((), dtype=torch.int32, device=device)
        stream = torch.cuda.current_stream(device).cuda_stream
        _check(runtime.hip.hipMemsetAsync(error.data_ptr(), 0, 4, stream))
        for index, attention in enumerate(layers_attn):
            layer = attention.qkv_proj
            _check(runtime.expand(layer.weight.data_ptr(), layer.weight_scale_inv.data_ptr(),
                output.data_ptr() + index*2048*5120*2, error.data_ptr(),
                6144, 5120, 4096, 2048, stream))
        status = ctypes.c_uint()
        _check(runtime.hip.hipMemcpyAsync(ctypes.byref(status), error.data_ptr(), 4, 2, stream))
        _check(runtime.hip.hipStreamSynchronize(stream))
        if status.value:
            raise ValueError(f"Native DFlash2 context rejected nonfinite weights/scales: {status.value}")
        self._fused_kv_weight = output
        self._paiton_context_runtime = runtime
        self.paiton_native_context_audit = dict(
            layers=5, rows=10240, columns=5120, dtype="bfloat16",
            arithmetic=runtime.manifest["arithmetic"], artifact_sha256=runtime.manifest["sha256"],
        )


class PaitonDFlash2ForCausalLM(DFlash2Qwen3ForCausalLM):
    model_cls = PaitonDFlash2Model

    def prepare_native_head(self, scope):
        if getattr(self, "paiton_native_head", None) is not None:
            raise RuntimeError("Native draft head was already prepared")
        from .dflash2_native_head import NativeDraftHead
        self.paiton_native_head = NativeDraftHead(self, scope)

    def compute_candidates(self, hidden_states):
        head = getattr(self, "paiton_native_head", None)
        if head is not None:
            result = head(hidden_states, self.lm_head.weight)
            if result is not None:
                return result
        return super().compute_candidates(hidden_states)
