# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini attention on Paiton's native attention library (gfx1201).

Binding of `inkling2_attn_gfx1201.so` (path in `PAITON_INK2_ATTN_SO`, ABI
version 2): relative-bias attention with sink and output gate for window and
full layers, prefill and decode, and the write of clipped K and normed V into
the vLLM cache tensor of a visit. The library has no framework dependency; this
module hands it device pointers and strides of the tensors vLLM owns. No Triton
and no upstream operator module is imported here.
"""

import ctypes as C
import os

import torch


class _Layer(C.Structure):  # PaitonInkling2AttnLayer
    _fields_ = [("q_scale", C.c_void_p), ("r_scale", C.c_void_p), ("rel_table", C.c_void_p),
                ("window", C.c_int), ("log_alpha", C.c_float), ("log_floor", C.c_float)]


class _Cache(C.Structure):  # PaitonInkling2AttnCache
    _fields_ = [("k", C.c_void_p), ("v", C.c_void_p), ("stride_block", C.c_int64),
                ("stride_head", C.c_int64), ("stride_slot", C.c_int64), ("block_size", C.c_int)]


class _Batch(C.Structure):  # PaitonInkling2AttnBatch
    _fields_ = [("block_table", C.c_void_p), ("seq_lens", C.c_void_p), ("query_start", C.c_void_p),
                ("block_table_cols", C.c_int), ("requests", C.c_int), ("tokens", C.c_int),
                ("max_query_len", C.c_int)]


class _Inputs(C.Structure):  # PaitonInkling2AttnInputs
    _fields_ = [("q", C.c_void_p), ("q_stride", C.c_int64), ("r", C.c_void_p), ("r_stride", C.c_int64),
                ("gate", C.c_void_p), ("gate_stride", C.c_int64), ("sink", C.c_void_p),
                ("sink_stride", C.c_int64)]


_lib = None
_workspace: dict[torch.device, torch.Tensor] = {}
DECODE_ROWS = 16  # the library splits a full-layer read only for calls of at most 16 rows per request
SPLITS = 8


def _library():
    global _lib
    if _lib is None:
        path = os.environ.get("PAITON_INK2_ATTN_SO")
        if not path:
            raise ValueError("the native attention needs PAITON_INK2_ATTN_SO")
        lib = C.CDLL(path)
        if lib.PaitonInkling2AttnAbiVersion() != 2:
            raise RuntimeError(f"{path}: attention ABI {lib.PaitonInkling2AttnAbiVersion()}, need 2")
        lib.PaitonInkling2AttnRelTableBytes.restype = C.c_size_t
        lib.PaitonInkling2AttnWorkspaceBytes.restype = C.c_size_t
        lib.PaitonInkling2AttnWorkspaceBytes.argtypes = [C.c_int, C.c_int, C.c_void_p]
        lib.PaitonInkling2AttnPrepare.argtypes = [C.c_void_p] * 3
        lib.PaitonInkling2AttnWrite.argtypes = (
            [C.POINTER(_Cache), C.c_void_p, C.c_int64, C.c_void_p, C.c_int64, C.c_void_p, C.c_int, C.c_void_p]
        )
        lib.PaitonInkling2AttnForward.argtypes = (
            [C.POINTER(_Layer), C.POINTER(_Batch), C.POINTER(_Cache), C.POINTER(_Inputs)]
            + [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_int, C.c_void_p]
        )
        _lib = lib
    return _lib


def _ok(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with HIP error {rc}")


def _stream(device: torch.device):
    return torch.cuda.current_stream(device).cuda_stream


def _rows(tensor: torch.Tensor, width: int) -> tuple[int, int]:
    """Pointer and byte stride between token rows of a bf16 `(T, width)` view."""
    assert tensor.dtype == torch.bfloat16 and tensor.shape[1] == width
    assert tensor.dim() == 2 and (width == 1 or tensor.stride(1) == 1)
    return tensor.data_ptr(), tensor.stride(0) * 2 if tensor.shape[0] > 1 else width * 2


def _int32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dtype == torch.int32 and tensor.is_contiguous() else tensor.int().contiguous()


class NativeAttention:
    """One attention layer's constants. `prepare` runs once, after the
    weights are on the card."""

    def __init__(self, is_local: bool, window: int, log_alpha: float, log_floor: float) -> None:
        self.window = window if is_local else 0
        self.log_alpha = 0.0 if is_local else log_alpha
        self.log_floor = log_floor
        self.flags = int(os.environ.get("PAITON_INK2_ATTN_FLAGS", "0"))
        self.layer: _Layer | None = None

    def prepare(self, rel_proj: torch.Tensor, q_scale: torch.Tensor, r_scale: torch.Tensor) -> None:
        """`rel_proj` bf16 `(16, 1024)` in checkpoint layout, `q_scale` bf16
        `(24, 128)`, `r_scale` bf16 `(24, 16)`. The tensors must stay alive."""
        lib = _library()
        for tensor, shape in ((rel_proj, (16, 1024)), (q_scale, (24, 128)), (r_scale, (24, 16))):
            assert tensor.is_cuda and tensor.dtype == torch.bfloat16 and tensor.is_contiguous()
            assert tuple(tensor.shape) == shape, tuple(tensor.shape)
        self.keep = (q_scale, r_scale, torch.empty(
            lib.PaitonInkling2AttnRelTableBytes(), dtype=torch.uint8, device=rel_proj.device))
        _ok(lib.PaitonInkling2AttnPrepare(rel_proj.data_ptr(), self.keep[2].data_ptr(), _stream(rel_proj.device)),
            "PaitonInkling2AttnPrepare")
        self.layer = _Layer(q_scale.data_ptr(), r_scale.data_ptr(), self.keep[2].data_ptr(),
                            self.window, self.log_alpha, self.log_floor)

    @staticmethod
    def _cache(kv_cache: torch.Tensor) -> _Cache:
        """vLLM's cache tensor of one visit: bf16 `(blocks, 8, 16, 256)`, K
        in the first 128 values of a slot and V in the last 128."""
        assert kv_cache.dtype == torch.bfloat16 and tuple(kv_cache.shape[1:]) == (8, 16, 256)
        assert kv_cache.stride(3) == 1
        k = kv_cache.data_ptr()
        return _Cache(k, k + 256, *(kv_cache.stride(i) * 2 for i in range(3)), 16)

    def write(self, kv_cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              slot_mapping: torch.Tensor, tokens: int) -> None:
        """Clip K, norm V (bf16 `(T, 1024)` outputs of the K and V short
        convolutions) and store them at `slot_mapping` (int64; negative skips)."""
        if tokens == 0:
            return
        assert slot_mapping.dtype == torch.int64 and slot_mapping.is_contiguous()
        cache = self._cache(kv_cache)
        _ok(_library().PaitonInkling2AttnWrite(
            C.byref(cache), *_rows(k, 1024), *_rows(v, 1024), slot_mapping.data_ptr(), tokens,
            _stream(k.device)), "PaitonInkling2AttnWrite")

    def forward(self, kv_cache: torch.Tensor, block_table: torch.Tensor, seq_lens: torch.Tensor,
                query_start: torch.Tensor, max_query_len: int, tokens: int, q: torch.Tensor,
                r: torch.Tensor, gate: torch.Tensor, sinks: torch.Tensor, out: torch.Tensor) -> None:
        """Gated attention output, bf16 `(T, 3072)`, into `out`. `q`, `r`,
        `gate`, `sinks` are the bf16 columns of the projection output (before
        scale and clip); `block_table`, `seq_lens` and `query_start` are the
        visit's attention metadata. Rows past `tokens` are zeroed."""
        assert self.layer is not None, "prepare() was not called"
        assert out.dtype == torch.bfloat16 and out.is_contiguous() and out.shape[1] == 3072
        out[tokens:].zero_()
        if tokens == 0:
            return
        lib = _library()
        block_table, seq_lens, query_start = _int32(block_table), _int32(seq_lens), _int32(query_start)
        requests = seq_lens.shape[0]
        assert block_table.shape[0] >= requests and query_start.shape[0] >= requests + 1
        batch = _Batch(block_table.data_ptr(), seq_lens.data_ptr(), query_start.data_ptr(),
                       block_table.shape[1], requests, tokens, max_query_len)
        inputs = _Inputs(*_rows(q, 3072), *_rows(r, 384), *_rows(gate, 3072), *_rows(sinks, 24))
        cache = self._cache(kv_cache)
        need = lib.PaitonInkling2AttnWorkspaceBytes(tokens, SPLITS if max_query_len <= DECODE_ROWS else 1, None)
        # One workspace per card for all layers: the calls of one stream run in order.
        workspace = _workspace.get(out.device)
        if workspace is None or workspace.numel() < need:
            workspace = _workspace[out.device] = torch.empty(need, dtype=torch.uint8, device=out.device)
        _ok(lib.PaitonInkling2AttnForward(
            C.byref(self.layer), C.byref(batch), C.byref(cache), C.byref(inputs), out.data_ptr(),
            workspace.data_ptr(), workspace.numel(), 0, self.flags, _stream(out.device)),
            "PaitonInkling2AttnForward")
