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
import math
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

# Cache format of the 8 full-attention visits (`PAITON_INK2_KV`, launcher option --kv-cache); the window visits and
# the drafter stay bf16. A packed format is one vLLM head slot of `bytes` per token (uint8). The library addresses
# pages of 16 slots x 8 key heads (`16 * bytes`), V `v` bytes after K in a head's part of the page, as
# `page index * stride_block`. A vLLM block of N slots holds N / 16 library pages back to back, and vLLM may pad
# the block (format 4: 3 pages of 16,896 bytes in a 65,536-byte block), so the plugin hands the library a
# stride_block of U = gcd(vLLM block bytes, page bytes) and page indices in units of U: page j of vLLM block b is
# b * (block bytes / U) + j * (page bytes / U). When U is not the page, slots are handed over the same way.
#   name: (bytes per token and visit, forward flag, write entry, V offset in a head's page, K bytes per slot,
#          the library's format id)
KV_FORMATS = {"fp8": (2048, 512, "PaitonInkling2AttnWriteFp8", 2048, 128, 2),
              "kv4": (1056, 2048, "PaitonInkling2AttnWriteKv4", 1024, 64, 4)}  # the library's layout file, format 4
KV4_FLAG = 2048  # a format-4 layer sums every row in 8 parts, prefill rows too: its workspace is the 8-part one
KV = os.environ.get("PAITON_INK2_KV", "bf16")
if KV != "bf16" and KV not in KV_FORMATS:
    raise ValueError(f"PAITON_INK2_KV must be bf16 or one of {sorted(KV_FORMATS)}")
PAGE = 16  # slots per library page
# PAITON_INK2_KV_DENSE=1: a packed block holds one page more when whole pages do not fill the bf16 window page
# (format 4: 4 pages, 67,584 bytes, instead of 3 in 65,536). That block is then the largest page and vLLM pads the
# window and conv-state pages up to it, so every reader of those caches must take the block stride from the tensor.
KV_DENSE = os.environ.get("PAITON_INK2_KV_DENSE", "0") == "1"
_tables: list = [None, {}]  # [forward context, {id(tensor): (tensor, same in library units)}]


def _per_forward(key: torch.Tensor, make):
    """`make()` once per forward and tensor (for the 8 visits): vLLM refills the same metadata tensors every step,
    so the forward context is part of the key."""
    from vllm.forward_context import get_forward_context

    now = get_forward_context()
    if _tables[0] is not now:
        _tables[0], _tables[1] = now, {}
    hit = _tables[1].get(id(key))
    if hit is None or hit[0] is not key:
        hit = _tables[1][id(key)] = (key, make())
    return hit[1]


def _units(kv_cache: torch.Tensor, size: int) -> tuple[int, int, int]:
    """(U, vLLM block in units of U, library page in units of U) of a packed cache tensor."""
    block, page = kv_cache.stride(0) if kv_cache.shape[0] > 1 else kv_cache.shape[2] * size, PAGE * size
    unit = math.gcd(block, page)
    return unit, block // unit, page // unit


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
        writes = ["PaitonInkling2AttnWrite"] + ([KV_FORMATS[KV][2]] if KV in KV_FORMATS else [])
        if not hasattr(lib, writes[-1]):
            raise RuntimeError(f"{path}: no {writes[-1]}, needed for PAITON_INK2_KV={KV}")
        for entry in writes:  # the format's write entry takes the same arguments
            getattr(lib, entry).argtypes = (
                [C.POINTER(_Cache), C.c_void_p, C.c_int64, C.c_void_p, C.c_int64, C.c_void_p, C.c_int, C.c_void_p]
            )
        if KV in KV_FORMATS:
            size, fmt = KV_FORMATS[KV][0], KV_FORMATS[KV][5]
            # a library without the Formats entry reads bf16 and fp8 (formats 1 and 2)
            formats = lib.PaitonInkling2AttnFormats() if hasattr(lib, "PaitonInkling2AttnFormats") else 3
            if not formats & fmt:
                raise RuntimeError(f"{path}: reads cache formats {formats}, PAITON_INK2_KV={KV} is format {fmt}")
            if hasattr(lib, "PaitonInkling2AttnPageBytes"):
                lib.PaitonInkling2AttnPageBytes.restype, lib.PaitonInkling2AttnPageBytes.argtypes = C.c_size_t, [C.c_int]
                if lib.PaitonInkling2AttnPageBytes(fmt) != PAGE * size:
                    raise RuntimeError(f"{path}: a page of format {fmt} is {lib.PaitonInkling2AttnPageBytes(fmt)} bytes, "
                                       f"the plugin's layout has {PAGE * size}")
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
        self.format = KV_FORMATS.get(KV) if not is_local else None
        if self.format is not None:
            self.flags |= self.format[1]
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
    def _cache(kv_cache: torch.Tensor, fmt: tuple | None = None) -> _Cache:
        """vLLM's cache tensor of one visit: bf16 `(blocks, 8, 16, 256)`, K
        in the first 128 values of a slot and V in the last 128; or, for a
        packed format, uint8 `(blocks, 1, N, bytes)` holding N / 16 library pages."""
        if fmt is None:
            assert kv_cache.dtype == torch.bfloat16 and tuple(kv_cache.shape[1:]) == (8, 16, 256)
            assert kv_cache.stride(3) == 1
            k = kv_cache.data_ptr()
            return _Cache(k, k + 256, *(kv_cache.stride(i) * 2 for i in range(3)), 16)
        size, _, _, v, slot, _ = fmt
        blocks, slots, n, width = kv_cache.shape
        assert kv_cache.dtype == torch.uint8 and slots == 1 and width == size and n % PAGE == 0, tuple(kv_cache.shape)
        assert kv_cache.stride(2) == size and kv_cache.stride(3) == 1, kv_cache.stride()
        unit = _units(kv_cache, size)[0]
        assert unit % 64 == 0 or unit == PAGE * size, unit  # the library's stride_block rule
        k = kv_cache.data_ptr()
        return _Cache(k, k + v, unit, PAGE * size // 8, slot, PAGE)

    @staticmethod
    def _page_table(block_table: torch.Tensor, kv_cache: torch.Tensor, size: int) -> torch.Tensor:
        """The block table in library pages, as page indices in units of U (see KV_FORMATS)."""
        _, block, page = _units(kv_cache, size)
        r = kv_cache.shape[2] // PAGE
        if block == 1 and r == 1:
            return block_table

        def make():
            offsets = torch.arange(r, dtype=block_table.dtype, device=block_table.device) * page
            return (block_table.unsqueeze(-1) * block + offsets).flatten(1).contiguous()
        return _per_forward(block_table, make)

    @staticmethod
    def _slots(slot_mapping: torch.Tensor, kv_cache: torch.Tensor, size: int) -> torch.Tensor:
        """vLLM slots as the library's slots (`page index * 16 + slot in the page`); the same when U is the page."""
        _, block, page = _units(kv_cache, size)
        if page == 1:
            return slot_mapping
        n = kv_cache.shape[2]

        def make():
            b, t = slot_mapping.div(n, rounding_mode="floor"), slot_mapping.remainder(n)
            mapped = (b * block + t.div(PAGE, rounding_mode="floor") * page) * PAGE + t.remainder(PAGE)
            return torch.where(slot_mapping < 0, slot_mapping, mapped)
        return _per_forward(slot_mapping, make)

    def write(self, kv_cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              slot_mapping: torch.Tensor, tokens: int) -> None:
        """Clip K, norm V (bf16 `(T, 1024)` outputs of the K and V short
        convolutions) and store them at `slot_mapping` (int64; negative skips)."""
        if tokens == 0:
            return
        assert slot_mapping.dtype == torch.int64 and slot_mapping.is_contiguous()
        cache = self._cache(kv_cache, self.format)
        entry = self.format[2] if self.format else "PaitonInkling2AttnWrite"
        if self.format is not None:
            slot_mapping = self._slots(slot_mapping, kv_cache, self.format[0])
        _ok(getattr(_library(), entry)(
            C.byref(cache), *_rows(k, 1024), *_rows(v, 1024), slot_mapping.data_ptr(), tokens,
            _stream(k.device)), entry)

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
        if self.format is not None:
            block_table = self._page_table(block_table, kv_cache, self.format[0])
        requests = seq_lens.shape[0]
        assert block_table.shape[0] >= requests and query_start.shape[0] >= requests + 1
        batch = _Batch(block_table.data_ptr(), seq_lens.data_ptr(), query_start.data_ptr(),
                       block_table.shape[1], requests, tokens, max_query_len)
        inputs = _Inputs(*_rows(q, 3072), *_rows(r, 384), *_rows(gate, 3072), *_rows(sinks, 24))
        cache = self._cache(kv_cache, self.format)
        parts = SPLITS if max_query_len <= DECODE_ROWS or self.flags & KV4_FLAG else 1
        need = lib.PaitonInkling2AttnWorkspaceBytes(tokens, parts, None)
        # One workspace per card for all layers: the calls of one stream run in order.
        workspace = _workspace.get(out.device)
        if workspace is None or workspace.numel() < need:
            workspace = _workspace[out.device] = torch.empty(need, dtype=torch.uint8, device=out.device)
        _ok(lib.PaitonInkling2AttnForward(
            C.byref(self.layer), C.byref(batch), C.byref(cache), C.byref(inputs), out.data_ptr(),
            workspace.data_ptr(), workspace.numel(), 0, self.flags, _stream(out.device)),
            "PaitonInkling2AttnForward")
