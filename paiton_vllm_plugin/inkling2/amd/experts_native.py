# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini experts on Paiton's native expert library (gfx1201).

`PAITON_INK2_EXPERTS=native` (read in `amd/model.py`) puts every layer's
experts on `inkling2_moe_gfx1201.so` (path in `PAITON_INK2_MOE_SO`): NVFP4 and
MXFP8 weights stay in the checkpoint layout on the card, and the library
applies the trained input quantization itself. The library has no framework
dependency; this module only hands it device pointers.

shortcut: the router stays in PyTorch fp32 (`Inkling2BringupMoE.forward`),
replace it with a native router kernel once a profile shows its cost.
"""

import ctypes as C
import os

import torch
from torch import nn

from .experts_bringup import Inkling2BringupMoE


class _Weights(C.Structure):  # PaitonInkling2MoeWeights, ABI version 1
    _fields_ = [
        (n, C.c_void_p)
        for n in ("w13", "w13_scale", "w13_amax", "w2", "w2_scale", "w2_amax")
    ] + [("w13_denominator", C.c_float), ("w2_denominator", C.c_float), ("format", C.c_int)]


_lib = None
_workspace: dict[torch.device, torch.Tensor] = {}


def _library():
    global _lib
    if _lib is None:
        path = os.environ.get("PAITON_INK2_MOE_SO")
        if not path:
            raise ValueError("PAITON_INK2_EXPERTS=native needs PAITON_INK2_MOE_SO")
        _lib = C.CDLL(path)
        _lib.PaitonInkling2MoeWorkspaceBytes.restype = C.c_size_t
        _lib.PaitonInkling2MoeWorkspaceBytes.argtypes = [C.c_int, C.c_int]
        _lib.PaitonInkling2MoeLayer.restype = C.c_int
        _lib.PaitonInkling2MoeLayer.argtypes = (
            [C.POINTER(_Weights)] + [C.c_void_p] * 5 + [C.c_size_t, C.c_int, C.c_int, C.c_void_p]
        )
    return _lib


class Inkling2NativeExperts(nn.Module):
    def __init__(self, config, expert_format: str) -> None:
        super().__init__()
        assert expert_format in ("nvfp4", "mxfp8")
        self.nvfp4 = expert_format == "nvfp4"
        # The library's dimensions are fixed to Inkling 2 Mini.
        num, hidden, inter = config.n_routed_experts, config.hidden_size, config.intermediate_size
        assert (num, hidden, inter, config.num_experts_per_tok) == (128, 2304, 1152, 8)
        assert (config.mlp_gate_softcap, config.mlp_up_softcap) == (4.0, 25.0)
        self.flags = int(os.environ.get("PAITON_INK2_MOE_FLAGS", "0"))
        self._w: _Weights | None = None
        for name, (n, k) in (("w13", (2 * inter, hidden)), ("w2", (hidden, inter))):
            if self.nvfp4:
                self._raw(f"{name}_weight", (num, n, k // 2), torch.uint8)
                self._raw(f"{name}_weight_scale", (num, n, k // 16), torch.float8_e4m3fn)
                self._raw(f"{name}_weight_amax", (num,), torch.float32)
                self._raw(f"{name}_weight_outer_scale_denominator", (1,), torch.float32)
            else:
                self._raw(f"{name}_weight", (num, n, k), torch.float8_e4m3fn)
                self._raw(f"{name}_weight_scale", (num, n, k // 32), torch.uint8)

    def _raw(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        self.register_parameter(
            name, nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        )

    def _weights(self) -> _Weights:
        # Built at the first call: the tensors are on the card by then.
        w = _Weights(format=0 if self.nvfp4 else 1)
        for name in ("w13", "w2"):
            for field, suffix in ((name, "_weight"), (f"{name}_scale", "_weight_scale")):
                tensor = getattr(self, name + suffix)
                assert tensor.is_cuda and tensor.is_contiguous()
                setattr(w, field, tensor.data_ptr())
            if self.nvfp4:
                setattr(w, f"{name}_amax", getattr(self, f"{name}_weight_amax").data_ptr())
                denominator = getattr(self, f"{name}_weight_outer_scale_denominator")
                setattr(w, f"{name}_denominator", denominator.item())
        return w

    def forward(self, x: torch.Tensor, idx: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """bf16 `(T, H)` rows, their `(T, 8)` experts and fp32 routing
        weights -> bf16 `(T, H)`."""
        tokens = x.shape[0]
        out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        if tokens == 0:
            return out.to(x.dtype)
        lib = _library()
        if self._w is None:
            self._w = self._weights()
        assert x.dtype == torch.bfloat16
        x, idx, weights = x.contiguous(), idx.int().contiguous(), weights.float().contiguous()
        need = lib.PaitonInkling2MoeWorkspaceBytes(tokens, self.flags)
        # One workspace per card for all layers: the calls of one stream run in order.
        workspace = _workspace.get(x.device)
        if workspace is None or workspace.numel() < need:
            workspace = _workspace[x.device] = torch.empty(need, dtype=torch.uint8, device=x.device)
        rc = lib.PaitonInkling2MoeLayer(
            C.byref(self._w),
            x.data_ptr(),
            idx.data_ptr(),
            weights.data_ptr(),
            out.data_ptr(),
            workspace.data_ptr(),
            workspace.numel(),
            tokens,
            self.flags,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        if rc != 0:
            raise RuntimeError(f"PaitonInkling2MoeLayer failed with HIP error {rc}")
        return out.to(x.dtype)


class Inkling2NativeMoE(nn.Module):
    """Drop-in for `Inkling2MoE` (same checkpoint names under `.mlp`)."""

    def __init__(self, config, expert_format: str) -> None:
        super().__init__()
        from .moe import Inkling2Gate  # imports vLLM and Triton

        assert config.n_shared_experts == 0
        self.gate = Inkling2Gate(config)
        self.experts = Inkling2NativeExperts(config, expert_format)

    forward = Inkling2BringupMoE.forward
