"""Compiled H3 Turbo residuals, paired SwiGLU and dense INT8 attention.

The adapter GEMMs and stock quantizers are retained. Every fused epilogue
materializes the same BF16 rounding boundaries as the pinned runtime.
"""
import ctypes
import sys
import threading

import torch

max_tokens = 32768
_launch = None
_unit_launch = None
_attention = None
_pack = None
_lut = None
_state = threading.local()
calls = {"residual": 0, "swiglu": 0, "attention": 0}


@torch.library.custom_op("paiton_video::h3_projection_residual", mutates_args=())
def compiled_residual(a: torch.Tensor, w: torch.Tensor, sa: torch.Tensor,
                      sw: torch.Tensor, residual: torch.Tensor,
                      lut: torch.Tensor, swiglu: bool, residual_scale: float = 0.0625) -> torch.Tensor:
    from .projection import PROFILES
    if _launch is None:
        raise RuntimeError("Load an H3 fusion artifact before inference")
    if a.ndim != 2 or w.ndim != 2:
        raise ValueError("Expected A[M,K] and W[N,K]")
    m, k = a.shape
    n = w.shape[0]
    if not 1 <= m <= max_tokens or w.shape[1] != k or (n, k) not in PROFILES:
        raise ValueError("Unsupported H3 projection shape")
    if swiglu and (n, k) != (28672, 5376):
        raise ValueError("Paired SwiGLU requires the main H3 FC1 projection")
    if sa.shape != (m,) or sw.shape != (n,) or residual.shape != (m, n) or lut.shape != (65536,):
        raise ValueError("Invalid scale, residual or SiLU table shape")
    for t, dtype in ((a, torch.int8), (w, torch.int8), (sa, torch.float32),
                     (sw, torch.float32), (residual, torch.bfloat16), (lut, torch.bfloat16)):
        if t.dtype != dtype or not t.is_cuda or not t.is_contiguous() or t.device != a.device or t.data_ptr() % 16:
            raise ValueError("Fusion operands must be aligned, contiguous and on one GPU")
    if residual_scale not in (0.0625, 1.0) or (residual_scale == 1.0 and _unit_launch is None):
        raise ValueError("Unsupported compiled H3 residual scale")
    launch = _unit_launch if residual_scale == 1.0 else _launch
    size = n*k + (((m+127)//128)*128*k if n > 5376 else 0)
    workspace = torch.empty(size, device=a.device, dtype=torch.uint8)
    out = torch.empty((m, n//2 if swiglu else n), device=a.device, dtype=torch.bfloat16)
    status = launch(*(t.data_ptr() for t in (a, w, sa, sw, residual, lut, out, workspace)),
                     size, m, n, k, int(swiglu), torch.cuda.current_stream(a.device).cuda_stream)
    if status:
        raise RuntimeError(f"H3 fused projection failed with HIP status {status}")
    return out


@compiled_residual.register_fake
def _fake(a, w, sa, sw, residual, lut, swiglu, residual_scale=0.0625):
    return torch.empty((a.shape[0], w.shape[0]//2 if swiglu else w.shape[0]),
                       device=a.device, dtype=torch.bfloat16)


def attend(q, k, v, qs, ks, vs):
    """Consume the unchanged stock quantizer's D128/B1/H56 buffers."""
    if _attention is None:
        raise RuntimeError("H3 attention artifact is not initialized")
    if q.ndim != 4 or q.shape[:2] != (1, 56) or q.shape[-1] != 128 or q.shape != k.shape:
        raise ValueError("H3 attention requires B1/H56/D128 self-attention")
    t, heads = q.shape[2], q.shape[1]
    qp, kp = qs.shape[-1], v.shape[-1]
    if not 1 <= t <= max_tokens or v.shape != (heads*128, kp) or qp < t or qp % 128 or kp < t or kp % 64:
        raise ValueError("Invalid H3 attention padding")
    if qs.shape != (1, heads, qp) or ks.shape != (1, heads, kp//16) or vs.shape != (heads*128,):
        raise ValueError("Invalid H3 attention scale shapes")
    for value, dtype in ((q, torch.int8), (k, torch.int8), (v, torch.int8),
                         (qs, torch.float32), (ks, torch.float32), (vs, torch.float32)):
        if value.dtype != dtype or not value.is_cuda or value.device != q.device or not value.is_contiguous():
            raise ValueError("Attention operands must be contiguous and on one GPU")
    pk = torch.empty((heads, kp, 128), device=q.device, dtype=torch.int8)
    pv = torch.empty_like(pk)
    out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(q.device).cuda_stream
    for status in (_pack(k.data_ptr(), pk.data_ptr(), t, 128, kp, heads, stream),
                   _pack(v.data_ptr(), pv.data_ptr(), 128, kp, 128, heads, stream)):
        if status:
            raise RuntimeError(f"H3 attention packing failed with HIP status {status}")
    status = _attention(*(x.data_ptr() for x in (q, pk, pv, out, qs, ks, vs)), t, heads, qp, kp, stream)
    if status:
        raise RuntimeError(f"H3 attention failed with HIP status {status}")
    calls["attention"] += 1
    return out


def initialize(p):
    global _launch, _unit_launch, _attention, _pack, _lut, max_tokens
    max_tokens = p.max_tokens
    version = p._library.PaitonH3FusionGetAbiVersion
    version.restype = ctypes.c_uint
    if version() != 1:
        raise RuntimeError("Unsupported H3 fusion ABI")
    _launch = p._library.PaitonH3ProjectionResidualRun
    _launch.argtypes = [ctypes.c_void_p]*8 + [ctypes.c_size_t] + [ctypes.c_int]*4 + [ctypes.c_void_p]
    _launch.restype = ctypes.c_int
    if hasattr(p._library, "PaitonH3ProjectionResidualUnitRun"):
        _unit_launch = p._library.PaitonH3ProjectionResidualUnitRun
        _unit_launch.argtypes = _launch.argtypes
        _unit_launch.restype = ctypes.c_int
    _attention = p._library.PaitonH3AttentionRun
    _attention.argtypes = [ctypes.c_void_p]*7 + [ctypes.c_int]*4 + [ctypes.c_void_p]
    _attention.restype = ctypes.c_int
    _pack = p._library.PaitonH3AttentionPack
    _pack.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*4 + [ctypes.c_void_p]
    _pack.restype = ctypes.c_int
    # BF16 has only 65,536 bit patterns. This table preserves stock SiLU's
    # BF16 result instead of approximating its exponential in another kernel.
    with torch.inference_mode():
        bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
        _lut = torch.nn.functional.silu(bits.view(torch.bfloat16))
    if "comfy.ops" in sys.modules:
        _install_comfy(p)


def _install_comfy(p):
    from comfy_kitchen.backends import hip
    from comfy.ldm.minimax.model import MLP
    from comfy.weight_adapter.lora import LoRAAdapter
    import comfy.ops

    original_projection = p.compiled_projection
    original_w4 = hip.w4a8_int8_linear
    original_mlp = MLP.forward
    original_attention = hip.sage_int8_attend

    def project(a, w, sa, sw):
        residual = getattr(_state, "residual", None)
        if residual is None:
            return original_projection(a, w, sa, sw)
        use_act = getattr(_state, "swiglu_request", False) and w.shape == (28672, 5376)
        out = compiled_residual(a, w, sa, sw, residual.reshape(a.shape[0], w.shape[0]), _lut, use_act, _state.residual_scale)
        _state.consumed = True
        if use_act:
            _state.swiglu_done = True
        calls["swiglu" if use_act else "residual"] += 1
        return out

    def bypass(self, original_forward, x, *args, **kwargs):
        values = self.weights
        eligible = (getattr(p._local, "enabled", False) and type(self) is LoRAAdapter
                    and not getattr(self, "is_conv", False) and len(values) == 6
                    and all(t is None for t in values[3:]) and x.dtype == torch.bfloat16
                    and x.is_cuda and x.ndim >= 2 and x.shape[-1] > 0 and x.is_contiguous()
                    and 1 <= x.numel()//x.shape[-1] <= max_tokens and values[0].ndim == values[1].ndim == 2
                    and (values[0].shape[0], values[1].shape[1]) in p.PROFILES)
        scale = ((values[2]/values[1].shape[0] if values[2] is not None else 1.0)
                 * getattr(self, "multiplier", 1.0)) if eligible else None
        if not isinstance(scale, (float, int)) or scale not in (0.0625, 1.0) or (scale == 1.0 and _unit_launch is None):
            base = original_forward(x, *args, **kwargs)
            return self.g(base + self.h(x, base))
        up, down = (v.to(dtype=x.dtype) for v in values[:2])
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x, down), up)
        previous, consumed = getattr(_state, "residual", None), getattr(_state, "consumed", False)
        previous_scale = getattr(_state, "residual_scale", 0.0625)
        _state.residual, _state.consumed, _state.residual_scale = delta, False, scale
        try:
            out = original_forward(x, *args, **kwargs)
            if not _state.consumed:
                out = out + delta*scale
        finally:
            _state.residual, _state.consumed, _state.residual_scale = previous, consumed, previous_scale
        return self.g(out)

    def w4(x, qdata, s_rel, s_channel, codebook=None, correction=None, bias=None,
           group_size=16, convrot_groupsize=256, out_dtype=torch.bfloat16):
        active = (getattr(_state, "swiglu_request", False) and getattr(_state, "residual", None) is not None
                  and qdata.shape == (28672, 2688) and correction is None and bias is None
                  and group_size == 16 and convrot_groupsize == 256 and x.dtype == torch.bfloat16
                  and out_dtype == torch.bfloat16)
        if not active:
            return original_w4(x, qdata, s_rel, s_channel, codebook, correction, bias,
                               group_size, convrot_groupsize, out_dtype)
        hip.validate_w4a8_operands(qdata, s_rel, s_channel, codebook, correction, group_size, convrot_groupsize)
        m = x.numel()//x.shape[-1]
        a, sa = hip._rotate_quant_int8(x.reshape(m, x.shape[-1]).contiguous(), 256)
        weight = hip._dequant_int4_grouped_to_int8(qdata, s_rel, codebook, group_size)
        sw = s_channel.to(device=x.device, dtype=torch.float32).reshape(-1).contiguous()
        out = project(a, weight, sa.reshape(-1).contiguous(), sw)
        p.calls += 1
        return out.reshape(*x.shape[:-1], 14336)

    def mlp(self, x):
        if not getattr(p._local, "enabled", False) or self.fc1.out_features != 28672:
            return original_mlp(self, x)
        previous, done = getattr(_state, "swiglu_request", False), getattr(_state, "swiglu_done", False)
        _state.swiglu_request, _state.swiglu_done = True, False
        try:
            h = self.fc1(x)
            consumed = _state.swiglu_done
        finally:
            _state.swiglu_request, _state.swiglu_done = previous, done
        return self.fc2(h) if consumed else comfy.ops.linear_input_act(self.fc2, h, "swiglu")

    def attention(q, k, v, qs, ks, vs, *, attention_scale, attn_mask, output_dtype, cta_k=64):
        active = (getattr(p._local, "enabled", False) and q.ndim == 4 and q.shape[:2] == (1, 56)
                  and q.shape[-1] == 128 and q.shape == k.shape and 1024 <= q.shape[2] <= max_tokens
                  and attention_scale == 128**-.5 and attn_mask is None
                  and output_dtype == torch.bfloat16 and cta_k == 64
                  and all(t.is_cuda and t.is_contiguous() and t.device == q.device for t in (q,k,v,qs,ks,vs)))
        if active:
            return attend(q, k, v, qs, ks, vs)
        return original_attention(q, k, v, qs, ks, vs, attention_scale=attention_scale,
                                  attn_mask=attn_mask, output_dtype=output_dtype, cta_k=cta_k)

    p.compiled_projection = project
    LoRAAdapter.bypass_forward = bypass
    hip.w4a8_int8_linear = w4
    MLP.forward = mlp
    hip.sage_int8_attend = attention
