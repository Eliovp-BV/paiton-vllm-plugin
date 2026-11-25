# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini small operators on Paiton's native operator library (gfx1201).

Binding of `inkling2_ops_gfx1201.so` (path in `PAITON_INK2_OPS_SO`, ABI
version 2 or 3; 3 adds the LM head kernel, its 8-bit rows and the int64 lookup): RMS norms, embedding with its norm, loop controller norms, AttnRes
mix, short convolution over the paged conv state (static and dynamic taps),
the conv metadata, and one bf16 linear layer for the attention projections,
the controller products and the LM head. `PAITON_INK2_OPS=native` (read in
`amd/model.py`, `amd/loop.py` and `amd/sconv.py`) sends every call here.

The functions keep the Python signatures of the operators in `amd/ops/` they
replace. They take torch tensors that are already on the card, hand the
library device pointers on torch's current stream and return torch tensors.
The library takes contiguous buffers only, has no workspace and does not
synchronize. No Triton, no upstream operator module and no PyTorch operator
is used here (tensors are allocated with `torch.empty`; an index tensor of
another integer type than the library reads is cast once, which the model's
tensors need only with an ABI 2 library, for `input_ids`).
"""

import ctypes as C
import os

import torch

BF16, F32 = torch.bfloat16, torch.float32
HIDDEN = 2304  # the library's row width


class _ConvState(C.Structure):  # PaitonInkling2OpsConvState
    _fields_ = [(n, C.c_void_p) for n in ("cache", "positions", "seq_idx", "slot_mapping", "query_start",
                                           "block_table")] + [("max_blocks", C.c_int)]


_lib = None
_abi = 0


def _library():
    global _lib, _abi
    if _lib is None:
        path = os.environ.get("PAITON_INK2_OPS_SO")
        if not path:
            raise ValueError("PAITON_INK2_OPS=native needs PAITON_INK2_OPS_SO")
        lib = C.CDLL(path)
        abi = lib.PaitonInkling2OpsAbiVersion()
        if abi not in (2, 3):
            raise RuntimeError(f"{path}: operator ABI {abi}, need 2 or 3")
        p, i, f = C.c_void_p, C.c_int, C.c_float
        newer = (
            ("EmbedNormI64", [p, i, p, p, p, i, f, p]),
            ("Fp8Rows", [p, p, p, i, i, p]),
            ("LmHeadFp8", [p, p, p, p, i, i, i, f, p]),
            ("LmHead", [p, p, p, i, i, i, f, p]),
        )
        for name, args in (*(newer if abi >= 3 else ()),
            ("RmsNorm", [p, i, p, p, i, i, f, p]),
            ("EmbedNorm", [p, i, p, p, p, i, f, p]),
            ("PairRmsNorm", [p, i, p, p, i, i, p]),
            ("BlendRmsNorm", [p, i, p, p, p, i, p, i, i, f, p]),
            ("AttnRes", [p, i, p, p, p, i, p, p, i, f, f, i, p]),
            ("ShortConv", [C.POINTER(_ConvState), p, i, p, p, p, p, i, i, i, p]),
            ("ShortConvDynamic", [C.POINTER(_ConvState), p, i, p, p, p, i, i, i, p]),
            ("SeqMetadata", [p, i, i, p, p, i, p]),
            ("Linear", [p, p, p, i, i, i, i, i, f, p]),
        ):
            fn = getattr(lib, "PaitonInkling2Ops" + name)
            fn.restype, fn.argtypes = C.c_int, args
        for name, args in (("Int8Rows", [p, p, p, i, i, p]), ("LmHeadInt8", [p, p, p, p, i, i, i, f, p]),
                           ("LinearFp8", [p, p, p, p, i, i, i, i, p]), ("LinearW8", [p, p, p, p, i, i, i, i, p]),
                           ("LinearBf16", [p, p, p, i, i, i, i, p]), ("MixRmsNorm", [p, C.c_size_t, p, p, p, i, i, f, p]),
                           ("AddRmsNorm", [p, p, p, p, i, f, p])):
            if hasattr(lib, "PaitonInkling2Ops" + name):   # in later builds of ABI 3 only
                fn = getattr(lib, "PaitonInkling2Ops" + name)
                fn.restype, fn.argtypes = C.c_int, args
        _lib, _abi = lib, abi
    return _lib


def head_formats() -> tuple[str, ...]:
    """The 8-bit LM head row formats the loaded library has."""
    lib = _library()
    return (*(("fp8",) if _abi >= 3 else ()), *(("int8",) if hasattr(lib, "PaitonInkling2OpsInt8Rows") else ()))


def proj_formats() -> tuple[str, ...]:
    """The 8-bit row formats the loaded library has a linear layer for (int8: `LinearW8`, the entry with a prefill
    form)."""
    lib = _library()
    return (*(("fp8",) if hasattr(lib, "PaitonInkling2OpsLinearFp8") else ()),
            *(("int8",) if hasattr(lib, "PaitonInkling2OpsLinearW8") else ()))


def _call(name: str, device: torch.device, *args) -> None:
    rc = getattr(_library(), "PaitonInkling2Ops" + name)(*args, torch.cuda.current_stream(device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"PaitonInkling2Ops{name} failed with HIP error {rc}")


def _ptr(tensor: torch.Tensor | None, dtype: torch.dtype | tuple[torch.dtype, ...], what: str) -> int | None:
    """Device pointer of a contiguous tensor of `dtype` (None stays None)."""
    if tensor is None:
        return None
    dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
    if not (tensor.is_cuda and tensor.dtype in dtypes and tensor.is_contiguous()):
        raise ValueError(f"{what}: need a contiguous {dtypes} tensor on the card, got {tensor.dtype} "
                         f"{tuple(tensor.shape)} strides {tensor.stride()}")
    return tensor.data_ptr()


def _streams(tensor: torch.Tensor, streams: int) -> tuple[torch.Tensor, int]:
    """A `(S, ...)` tensor that may broadcast over the streams (one row, or
    expanded with stride 0): its one stored copy and the stream count."""
    if tensor.shape[0] == 1 or tensor.stride(0) == 0:
        return tensor[0], 1
    assert tensor.shape[0] == streams
    return tensor, streams


def _as(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return tensor if tensor.dtype == dtype and tensor.is_contiguous() else tensor.to(dtype).contiguous()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float,
             out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """bf16 RMS norm of the rows of `x` (bf16 or fp32 `(T, 2304)`). `weight`
    `(2304,)` gives `(T, 2304)`; `weight` `(W, 2304)`, 1 to 8 rows, gives
    `(W, T, 2304)` from one pass over `x`."""
    assert x.dim() == 2 and x.shape[1] == HIDDEN == weight.shape[-1]
    if (out_dtype or x.dtype) != BF16:
        raise NotImplementedError("the native RMS norm writes bf16 only")
    rows = weight.shape[0] if weight.dim() == 2 else 1
    out = torch.empty((rows, *x.shape) if weight.dim() == 2 else x.shape, dtype=BF16, device=x.device)
    if x.shape[0]:
        _call("RmsNorm", x.device, _ptr(x, (BF16, F32), "rms_norm x"), x.dtype == F32,
              _ptr(weight, BF16, "rms_norm weight"), out.data_ptr(), x.shape[0], rows, eps)
    return out


def has(name: str) -> bool:
    """Whether the loaded library has the entry point `name` (later builds
    of ABI 3 added some, e.g. "MixRmsNorm", "AddRmsNorm", "LinearBf16")."""
    return hasattr(_library(), "PaitonInkling2Ops" + name)


def mix_rms_norm(features: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Draft model: the bf16 RMS norm of the `alpha`-weighted sum of the taps
    of every row, bf16 `(T, 2304)`. `features` bf16 `(T, taps * 2304)` (the
    rows may be strided), `alpha` fp32 `(taps,)`, `weight` bf16 `(2304,)`.
    The mix is fp32 in tap order (product, then sum), rounded once."""
    taps = alpha.shape[0]
    if not (features.is_cuda and features.dtype == BF16 and features.dim() == 2 and features.stride(1) == 1
            and features.shape[1] == taps * HIDDEN and 1 <= taps <= 8):
        raise ValueError(f"mix_rms_norm: need bf16 (T, {taps} * {HIDDEN}) rows on the card, got {features.dtype} "
                         f"{tuple(features.shape)} strides {features.stride()}")
    out = torch.empty((features.shape[0], HIDDEN), dtype=BF16, device=features.device)
    if features.shape[0]:
        _call("MixRmsNorm", features.device, features.data_ptr(), features.stride(0) * 2,
              _ptr(alpha, F32, "mix_rms_norm alpha"), _ptr(weight, BF16, "mix_rms_norm weight"), out.data_ptr(),
              features.shape[0], taps, eps)
    return out


def add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Draft model: `residual += x` in place (fp32 sum, rounded to bf16) and
    the bf16 RMS norm of the unrounded sum, `(T, 2304)`. `x`, `residual`
    bf16 `(T, 2304)`, `weight` bf16 `(2304,)`."""
    assert x.shape == residual.shape and x.dim() == 2 and x.shape[1] == HIDDEN == weight.shape[-1]
    out = torch.empty_like(x)
    if x.shape[0]:
        _call("AddRmsNorm", x.device, _ptr(x, BF16, "add_rms_norm x"), _ptr(residual, BF16, "add_rms_norm residual"),
              _ptr(weight, BF16, "add_rms_norm weight"), out.data_ptr(), x.shape[0], eps)
    return out


def linear_wide(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = BF16) -> torch.Tensor:
    """`x @ weight.T` for bf16 `x` `(T, K)` and a wide bf16 `weight` `(N, K)`
    in the LM head's kernel form (K a multiple of 128): the value of `linear`
    up to the fp32 summation order, at about twice its weight bytes per second."""
    assert out_dtype in (BF16, F32) and x.dim() == weight.dim() == 2 and x.shape[1] == weight.shape[1]
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=out_dtype, device=x.device)
    if x.shape[0]:
        _call("LinearBf16", x.device, _ptr(x, BF16, "linear_wide x"), _ptr(weight, BF16, "linear_wide weight"),
              out.data_ptr(), x.shape[0], weight.shape[0], x.shape[1], out_dtype == F32)
    return out


def embed_norm(input_ids: torch.Tensor, table: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The normed embedding, bf16 `(T, 2304)`: rows `input_ids` of `table`
    (bf16 `(vocab, 2304)`) through the embed norm (`weight` bf16, 2304 values)."""
    _library()
    ids = input_ids.reshape(-1)
    wide = _abi >= 3 and ids.dtype == torch.int64 and ids.is_contiguous()   # vLLM's ids as they are
    if not wide:
        ids = _as(ids, torch.int32)
    out = torch.empty((ids.shape[0], HIDDEN), dtype=BF16, device=table.device)
    if ids.shape[0]:
        _call("EmbedNormI64" if wide else "EmbedNorm", table.device, _ptr(table, BF16, "embedding table"),
              table.shape[0], ids.data_ptr(),
              _ptr(weight, BF16, "embed norm weight"), out.data_ptr(), ids.shape[0], eps)
    return out


def pair_rms_norm(dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    """bf16 `(S, T, 4608)` `[rms(dest), rms(message)]` of fp32 `(S, T, 2304)`
    inputs, weightless; `dest` may broadcast over the streams."""
    streams, tokens, _ = message.shape
    dest, dest_streams = _streams(dest, streams)
    out = torch.empty((streams, tokens, 2 * HIDDEN), dtype=BF16, device=message.device)
    if tokens:
        _call("PairRmsNorm", message.device, _ptr(dest, F32, "pair_rms_norm dest"), dest_streams,
              _ptr(message, F32, "pair_rms_norm message"), out.data_ptr(), streams, tokens)
    return out


def blend_rms_norm(dest: torch.Tensor, message: torch.Tensor, gain_forget: torch.Tensor,
                   norm_weight: torch.Tensor, norm_eps: float) -> torch.Tensor:
    """bf16 `(S, T, 2304)` RMS norm (row `s` of `norm_weight`) of
    `(1 - tanh(forget)) * dest + tanh(gain) * message`: fp32 `(S, T, 2304)`
    `dest` (may broadcast) and `message`, fp32 `(S, T, 4608)` `gain_forget`,
    bf16 `(S, 2304)` `norm_weight` (may broadcast)."""
    streams, tokens, _ = message.shape
    dest, dest_streams = _streams(dest, streams)
    norm_weight, weight_streams = _streams(norm_weight, streams)
    out = torch.empty((streams, tokens, HIDDEN), dtype=BF16, device=message.device)
    if tokens:
        _call("BlendRmsNorm", message.device, _ptr(dest, F32, "blend dest"), dest_streams,
              _ptr(message, F32, "blend message"), _ptr(gain_forget, F32, "blend gain_forget"),
              _ptr(norm_weight, BF16, "blend norm weight"), weight_streams, out.data_ptr(), streams, tokens,
              norm_eps)
    return out


def attn_res(stack: torch.Tensor, last: torch.Tensor, addend: torch.Tensor | None, queries: torch.Tensor,
             norm_weight: torch.Tensor | None, norm_eps: float, normalize_values: bool,
             eps: float = 1e-6) -> torch.Tensor:
    """Mix `[*stack, last (+ addend)]` with each pseudo-query. `stack` bf16
    `(N, T, 2304)`, N 0 to 7; `last`, `addend` bf16 `(T, 2304)`; `queries`
    fp32 `(Q, 2304)`, already scaled by 2304**-0.5. With `norm_weight` (bf16
    `(Q, 2304)`): the mixes through that RMS norm, bf16 `(Q, T, 2304)`;
    without: the fp32 mixes."""
    num_queries, tokens = queries.shape[0], last.shape[0]
    assert norm_weight is None or norm_weight.shape[0] == num_queries
    out = torch.empty((num_queries, tokens, HIDDEN), dtype=F32 if norm_weight is None else BF16,
                      device=last.device)
    if tokens:
        _call("AttnRes", last.device, _ptr(stack, BF16, "attn_res stack") if stack.shape[0] else None,
              stack.shape[0], _ptr(last, BF16, "attn_res last"), _ptr(addend, BF16, "attn_res addend"),
              _ptr(queries, F32, "attn_res queries"), num_queries, _ptr(norm_weight, BF16, "attn_res norm"),
              out.data_ptr(), tokens, eps, norm_eps, bool(normalize_values))
    return out


def _conv_state(cache: torch.Tensor, positions: torch.Tensor, block_table: torch.Tensor, seq_idx: torch.Tensor,
                slot_mapping: torch.Tensor, query_start: torch.Tensor) -> tuple[_ConvState, tuple]:
    """The library's view of vLLM's conv-state cache tensor (bf16
    `(blocks, 8, 4, 1024)`) and of the conv metadata of the forward. The
    second value keeps cast index tensors alive until the launch is queued."""
    assert tuple(cache.shape[1:]) == (8, 4, 1024), tuple(cache.shape)
    keep = (_as(positions, torch.int64), _as(seq_idx, torch.int32), _as(slot_mapping, torch.int64),
            _as(query_start, torch.int32), _as(block_table, torch.int32))
    return _ConvState(_ptr(cache, BF16, "conv-state cache"), *(t.data_ptr() for t in keep),
                      block_table.shape[1]), keep


def short_conv(x: torch.Tensor, weight: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor,
               block_table: torch.Tensor, seq_idx: torch.Tensor, slot_mapping: torch.Tensor,
               query_start: torch.Tensor, stream_offset: int, stream_width: int,
               addend: torch.Tensor | None = None, residual: torch.Tensor | None = None,
               out: torch.Tensor | None = None) -> torch.Tensor:
    """Insert `x` (bf16 or fp32 `(T, 8 * stream_width)`, rounded to bf16) into
    the conv-state cache and return the static 4-tap convolution with `(C, 4)`
    taps `weight`, bf16. With `addend`: `(residual + addend) + conv`;
    `residual` may be `out`. `x` must not be `out`."""
    channels = 8 * stream_width
    assert x.dim() == 2 and x.shape[1] == channels and tuple(weight.shape) == (channels, 4)
    if out is None:
        out = torch.empty(x.shape, dtype=BF16, device=x.device)
    if x.shape[0]:
        state, keep = _conv_state(cache, positions, block_table, seq_idx, slot_mapping, query_start)
        _call("ShortConv", x.device, C.byref(state), _ptr(x, (BF16, F32), "short_conv x"), x.dtype == F32,
              _ptr(weight, BF16, "short_conv weight"), _ptr(addend, BF16, "short_conv addend"),
              _ptr(residual, BF16, "short_conv residual"), _ptr(out, BF16, "short_conv out"), x.shape[0],
              stream_offset, stream_width)
    return out


def short_conv_dynamic(x: torch.Tensor, static_weight: torch.Tensor, dynamic: torch.Tensor, cache: torch.Tensor,
                       positions: torch.Tensor, block_table: torch.Tensor, seq_idx: torch.Tensor,
                       slot_mapping: torch.Tensor, query_start: torch.Tensor, stream_offset: int,
                       stream_width: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """The K or V convolution of a full layer: `static_weight` bf16 `(4, C)`
    plus the per-token tap deltas `dynamic` (bf16 `(T, 32)`, `linear` of the
    stream's input with the layer's `weight_proj`). Returns bf16 `(T, C)`."""
    channels = 8 * stream_width
    assert x.dim() == 2 and x.shape[1] == channels and tuple(static_weight.shape) == (4, channels)
    assert tuple(dynamic.shape) == (x.shape[0], 32)
    if out is None:
        out = torch.empty(x.shape, dtype=BF16, device=x.device)
    if x.shape[0]:
        state, keep = _conv_state(cache, positions, block_table, seq_idx, slot_mapping, query_start)
        _call("ShortConvDynamic", x.device, C.byref(state), _ptr(x, (BF16, F32), "short_conv x"), x.dtype == F32,
              _ptr(static_weight, BF16, "short_conv static weight"), _ptr(dynamic, BF16, "short_conv dynamic"),
              _ptr(out, BF16, "short_conv out"), x.shape[0], stream_offset, stream_width)
    return out


def seq_metadata(query_start_loc: torch.Tensor, num_reqs: int, num_actual_tokens: int, seq_idx_out: torch.Tensor,
                 query_start_out: torch.Tensor, num_padded_tokens: int | None = None) -> None:
    """Fill the per-token request index and first row of the request (the
    conv metadata; vLLM's `sconv_seq_metadata`). Padded rows get 0."""
    if num_padded_tokens is None:
        num_padded_tokens = num_actual_tokens
    if not num_actual_tokens <= num_padded_tokens <= min(seq_idx_out.shape[0], query_start_out.shape[0]):
        raise ValueError("the metadata buffers are too small for the padded token count")
    starts = _as(query_start_loc, torch.int32)
    _call("SeqMetadata", seq_idx_out.device, starts.data_ptr(), num_reqs, num_actual_tokens,
          _ptr(seq_idx_out, torch.int32, "seq_idx"), _ptr(query_start_out, torch.int32, "query_start"),
          num_padded_tokens)


def linear(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = BF16, soft_cap: float = 0.0,
           rows: slice | None = None) -> torch.Tensor:
    """`x @ weight.T` without bias for bf16 `x` `(T, K)` and `weight`
    `(N, K)`, or batched `(B, T, K)` and `(B, N, K)`; bf16 or fp32 out.
    `rows` takes a row range of a 2-D `weight` without a copy. K must be a
    multiple of 64. `soft_cap` > 0 applies `soft_cap * tanh(y / soft_cap)`."""
    assert out_dtype in (BF16, F32) and x.dim() == weight.dim() and x.shape[-1] == weight.shape[-1]
    batch = x.shape[0] if x.dim() == 3 else 1
    tokens, k = x.shape[-2:]
    pw = _ptr(weight, BF16, "linear weight")
    n = weight.shape[-2]
    if rows is not None:
        assert weight.dim() == 2 and rows.step in (None, 1)
        start, stop, _ = rows.indices(n)
        pw, n = pw + start * k * 2, stop - start
    out = torch.empty((*x.shape[:-1], n), dtype=out_dtype, device=x.device)
    if tokens:
        _call("Linear", x.device, _ptr(x, BF16, "linear x"), pw, out.data_ptr(), batch, tokens, n, k,
              out_dtype == F32, soft_cap)
    return out


def linear_rows(x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor, fmt: str, out_dtype: torch.dtype = BF16,
                rows: slice | None = None) -> torch.Tensor:
    """`x @ W.T` for bf16 `x` `(T, K)` on the 8-bit rows `(codes, scale)` that
    `head_rows(W, fmt)` made of the bf16 matrix `W`; `rows` takes a row range.
    K must be a multiple of 128."""
    n, k = codes.shape
    start = 0
    if rows is not None:
        start, stop, _ = rows.indices(n)
        n = stop - start
    out = torch.empty((x.shape[0], n), dtype=out_dtype, device=x.device)
    if x.shape[0]:
        _call("LinearFp8" if fmt == "fp8" else "LinearW8", x.device, _ptr(x, BF16, "linear x"),
              _ptr(codes, torch.uint8, "linear codes") + start * k, _ptr(scale, F32, "linear scale") + start * 4,
              out.data_ptr(), x.shape[0], n, k, out_dtype == F32)
    return out


# PAITON_INK2_PROJ=fp8 or int8 (default bf16): the attention input and output
# projections read 8-bit rows with one scale per row, converted on the card the
# first time a weight is used. int8 also covers the tap projections of the full
# layers, and has a prefill kernel. `release_projections` makes the rows of the
# input and output projections at load and gives back their bf16 matrices.
PROJ = os.environ.get("PAITON_INK2_PROJ", "bf16")
if PROJ not in ("bf16", "fp8", "int8"):
    raise ValueError("PAITON_INK2_PROJ must be bf16, fp8 or int8")
_proj_rows: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}  # id(weight): (weight, codes, scale)


def _rows_of(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The 8-bit rows of a projection weight (the module's parameter itself: the key outlives its bf16 data)."""
    hit = _proj_rows.get(id(weight))
    if hit is None:
        if PROJ not in proj_formats():
            raise ValueError(f"the operator library has no linear layer on {PROJ} rows (has {proj_formats()})")
        hit = _proj_rows[id(weight)] = (weight, *head_rows(weight, PROJ))
    return hit[1:]


def release_projections(weights) -> None:
    """Make the 8-bit rows of `weights` (parameters only ever read through `_rows_of`: the fused input projection
    and the output projection of every layer) and free their bf16 data, which nothing reads again: same rows, same
    tokens, 1.2 GiB more for the cache when called before vLLM measures the card."""
    assert PROJ != "bf16"
    for weight in weights:
        _rows_of(weight)
        weight.data = weight.data.new_empty((0, weight.shape[1]))
    torch.cuda.empty_cache()


def o_project(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The attention output projection (`wo_ud`): `linear(x, weight)`, or the
    same on 8-bit rows with `PAITON_INK2_PROJ`."""
    return linear(x, weight) if PROJ == "bf16" else linear_rows(x, *_rows_of(weight), PROJ)


def tap_project(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The tap projection of a full layer's K or V convolution (`weight_proj`,
    `(32, 2304)`): `linear(x, weight)`, on int8 rows with `PAITON_INK2_PROJ=int8`."""
    return linear_rows(x, *_rows_of(weight), PROJ) if PROJ == "int8" else linear(x, weight)


def linear_k_groups(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = F32) -> torch.Tensor:
    """The port's K-grouped projection: the same product on the native kernel."""
    return linear(x, weight, out_dtype)


def qkvr_project(x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, weight: torch.Tensor,
                 split_sizes: list[int]) -> list[torch.Tensor]:
    """The fused attention input projection: `[q, r, gate, sinks, k, v]`.
    `weight` bf16 `(8600, 2304)` with rows in the order of `split_sizes`
    (q 3072, r 384, gate 3072, sinks 24, k 1024, v 1024). q, r, gate and sinks
    are column views of one `(T, 6552)` buffer (row stride 6552 values); k and
    v are contiguous `(T, 1024)`, as the short convolution takes them."""
    q_side, kv = sum(split_sizes[:4]), split_sizes[4]
    if PROJ == "bf16":
        project = lambda x, rows: linear(x, weight, rows=rows)  # noqa: E731
    else:
        codes, scale = _rows_of(weight)
        project = lambda x, rows: linear_rows(x, codes, scale, PROJ, rows=rows)  # noqa: E731
    query = project(x_q, slice(0, q_side))
    k = project(x_k, slice(q_side, q_side + kv))
    v = project(x_v, slice(q_side + kv, q_side + 2 * kv))
    views, at = [], 0
    for size in split_sizes[:4]:
        views.append(query[:, at:at + size])
        at += size
    return [*views, k, v]


def controller_products(pair: torch.Tensor, down: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """fp32 `(S, T, 4608)` `[gain, forget]` of a loop controller: bf16 `pair`
    `(S, T, 4608)` through `down` `(S, 64, 4608)` (rounded to bf16), then
    `up` `(S, 4608, 64)`."""
    return linear(linear(pair, down), up, F32)


def lm_head(hidden: torch.Tensor, weight: torch.Tensor, vocab_size: int, soft_cap: float | None) -> torch.Tensor:
    """fp32 logits `(T, vocab_size)` from the first `vocab_size` rows of the
    unembed matrix, with the final soft cap. An ABI 3 library runs the LM
    head kernel (128-k chunks, about twice the weight bytes per second of
    `linear`); the result differs from `linear` by fp32 summation order only."""
    _library()
    if _abi < 3:
        return linear(hidden, weight, F32, soft_cap or 0.0, rows=slice(0, vocab_size))
    out = torch.empty((hidden.shape[0], vocab_size), dtype=F32, device=hidden.device)
    if hidden.shape[0]:
        _call("LmHead", hidden.device, _ptr(hidden, BF16, "lm head hidden"), _ptr(weight, BF16, "lm head weight"),
              out.data_ptr(), hidden.shape[0], vocab_size, hidden.shape[1], soft_cap or 0.0)
    return out


_HEAD = {"fp8": ("Fp8Rows", "LmHeadFp8"), "int8": ("Int8Rows", "LmHeadInt8")}


def head_rows(weight: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """LM head rows in 8 bits, once at load: codes `(N, K)` (bytes: e4m3 for
    `fmt` "fp8", int8 for "int8") and one fp32 scale per row `(N,)` of the
    bf16 rows `weight` (ABI 3)."""
    if fmt not in head_formats():
        raise ValueError(f"the operator library has no {fmt} LM head rows (has {head_formats()})")
    n, k = weight.shape
    codes = torch.empty((n, k), dtype=torch.uint8, device=weight.device)
    scale = torch.empty(n, dtype=F32, device=weight.device)
    _call(_HEAD[fmt][0], weight.device, _ptr(weight, BF16, "LM head weight"), codes.data_ptr(), scale.data_ptr(), n, k)
    return codes, scale


def lm_head_rows(hidden: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor, soft_cap: float | None,
                 fmt: str) -> torch.Tensor:
    """fp32 logits `(T, N)` from the 8-bit rows of `head_rows`, with the final
    soft cap. `hidden` stays bf16 (only the weights are quantized)."""
    out = torch.empty((hidden.shape[0], codes.shape[0]), dtype=F32, device=hidden.device)
    if hidden.shape[0]:
        _call(_HEAD[fmt][1], hidden.device, _ptr(hidden, BF16, "lm head hidden"),
              _ptr(codes, torch.uint8, "lm head codes"), _ptr(scale, F32, "lm head scale"), out.data_ptr(),
              hidden.shape[0], codes.shape[0], codes.shape[1], soft_cap or 0.0)
    return out
