# SPDX-License-Identifier: Apache-2.0
"""Bring-up experts for Inkling 2 on gfx950: PyTorch on dequantized weights.

This is NOT the R9700 path and not a serving path. It exists so the looped
model can be checked on a card with room for dequantized experts (2 GB per
layer in bf16 values held as fp32: 4 GB) before the native 4-bit kernel exists.

The checkpoint's experts are NVFP4 (layers 0-19) or MXFP8 (layers 20-23) and
were trained with their inputs quantized to the same format
(`hf_quant_config.json`, `required_activation_quantization`). `act_quant`
applies that input quantization (gate-up input, and SwiGLU output times the
routing weight before the down projection), written from upstream's NVFP4 and
MXFP8 quantizers.

PAITON_INK2_EXPERTS picks what a layer uses (read in `amd/model.py`):
  native  every layer: Paiton's expert library (`experts_native.py`), the
          only mode that PAITON_INK2_OPS=native accepts.
  ref-b   every layer: this module with `act_quant` (trained arithmetic).
  ref-a   every layer: this module without it (bf16 expert inputs).
`ref-a` / `ref-b` also round the dequantized weights to bf16, and the NVFP4
per-expert outer scale's amax before that, as the PyTorch reference does when
it loads the checkpoint (its loader casts every source tensor to bf16).

shortcut: one Python loop over the routed experts per call on fp32 weights,
block-error sums are plain fp32 sums (upstream sums pairwise, so a rare
4-vs-6 tie can differ); replace with the native expert kernel.
"""

import torch
import torch.nn.functional as F
from torch import nn

FMAX = 3.4028234663852886e38
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest even onto float8 e4m3fn, saturating at 448."""
    a = x.abs().clamp(max=448.0)
    _, ex = torch.frexp(a.clamp(min=2.0**-9))
    step = torch.exp2((ex - 1).clamp(min=-6).float() - 3)
    return torch.sign(x) * (torch.round(a / step) * step).clamp(max=448.0)


def e2m1(x: torch.Tensor) -> torch.Tensor:
    """Thresholds of upstream `_e2m1` (nearest even, saturating)."""
    a = x.abs()
    v = torch.full_like(a, 6.0)
    for hi, val, incl in (
        (5.0, 4.0, True),
        (3.5, 3.0, False),
        (2.5, 2.0, True),
        (1.75, 1.5, False),
        (1.25, 1.0, True),
        (0.75, 0.5, False),
        (0.25, 0.0, True),
    ):
        v = torch.where(a <= hi if incl else a < hi, torch.full_like(a, val), v)
    return torch.sign(x) * v


def fq_nvfp4(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 per token: row scale amax / 1536, e4m3 block scale per 16 values
    chosen between a 6- and a 4-based one by absolute error."""
    m, k = x.shape
    amax = x.abs().amax(1, keepdim=True)
    enc = (1536.0 / amax).clamp(max=FMAX)
    dec, outer = 1.0 / enc, (amax * (1.0 / 1536.0))[..., None]
    xb = x.view(m, k // 16, 16)
    bmax = xb.abs().amax(-1)
    best = err = None
    for div in (1.0 / 6.0, 0.25):
        s = e4m3((bmax * (enc * div)).clamp(max=448.0))
        inv = (1.0 / (s * dec)).clamp(max=FMAX)
        r = torch.nan_to_num(e2m1(xb * inv[..., None]) * s[..., None] * outer)
        e = (r - xb).abs().sum(-1, keepdim=True)
        best, err = (r, e) if best is None else (torch.where(e < err, r, best), None)
    return best.view(m, k)


def fq_mxfp8(x: torch.Tensor) -> torch.Tensor:
    """MXFP8: e4m3 values with a power-of-two scale per 32 values."""
    m, k = x.shape
    xb = x.view(m, k // 32, 32)
    raw = xb.abs().amax(-1, keepdim=True) * (1.0 / 448.0)
    scale = torch.exp2(torch.ceil(torch.log2(raw.clamp(min=2.0**-127))))
    return (e4m3((xb / scale).clamp(-448.0, 448.0)) * scale).view(m, k)


def _e4m3_values(codes: torch.Tensor) -> torch.Tensor:
    """fp32 values of float8_e4m3fn codes, through a 256-entry table."""
    table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    return table.to(codes.device)[codes.view(torch.uint8).long()]


def dequant_nvfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 `(N, K)` of NVFP4 weights before the outer scale: uint8
    `(N, K/2)` e2m1 pairs (even element in the low nibble, sign in bit 3)
    times their e4m3 `(N, K/16)` block scales."""
    grid = torch.tensor(_E2M1, device=packed.device)
    table = torch.cat((grid, -grid))
    codes = torch.stack((packed & 0xF, packed >> 4), -1).flatten(-2).long()
    return table[codes] * _e4m3_values(scale).repeat_interleave(16, -1)


def dequant_mxfp8(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 `(N, K)` of MXFP8 weights: e4m3 values times their uint8 e8m0
    `(N, K/32)` block scales."""
    block = torch.exp2(scale.float() - 127.0).repeat_interleave(32, -1)
    return _e4m3_values(values) * block


class Inkling2BringupExperts(nn.Module):
    def __init__(
        self, config, expert_format: str, act_quant: bool, round_bf16: bool
    ) -> None:
        super().__init__()
        assert expert_format in ("nvfp4", "mxfp8")
        self.nvfp4 = expert_format == "nvfp4"
        self.act_quant = act_quant
        self.round_bf16 = round_bf16
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        num, hidden, inter = (
            config.n_routed_experts,
            config.hidden_size,
            config.intermediate_size,
        )
        # Checkpoint layout; replaced by `w13` / `w2` (fp32) after loading.
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
        param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        self.register_parameter(name, param)

    def process_weights_after_loading(self) -> None:
        for name in ("w13", "w2"):
            values = getattr(self, f"{name}_weight")
            scale = getattr(self, f"{name}_weight_scale")
            experts = []
            # Per expert: a whole layer's fp32 temporaries are 3x its size.
            for e in range(values.shape[0]):
                if self.nvfp4:
                    amax = getattr(self, f"{name}_weight_amax")[e]
                    denom = getattr(self, f"{name}_weight_outer_scale_denominator")
                    if self.round_bf16:
                        # The reference's loader casts both to bf16 first.
                        amax = amax.to(torch.bfloat16).float()
                        denom = denom.to(torch.bfloat16).float()
                    weight = dequant_nvfp4(values[e], scale[e]) * (amax / denom)
                else:
                    weight = dequant_mxfp8(values[e], scale[e])
                if self.round_bf16:
                    weight = weight.to(torch.bfloat16).float()
                experts.append(weight)
            self.register_buffer(name, torch.stack(experts), persistent=False)
            del experts
            for param in (values, scale):
                param.data = param.data.new_empty((0,))

    def _softcap(self, x: torch.Tensor, cap: float) -> torch.Tensor:
        return torch.tanh(x / cap) * cap if cap > 0 else x

    def forward(
        self, x: torch.Tensor, idx: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """bf16 `(T, H)` rows, their `(T, topk)` experts and fp32 routing
        weights -> bf16 `(T, H)`."""
        fq = fq_nvfp4 if self.nvfp4 else fq_mxfp8
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        # With `act_quant` the GEMMs run in fp32 on the quantized inputs;
        # without it in bf16, as the reference does.
        xq = fq(x.float()) if self.act_quant else x
        for e in idx.unique().tolist():
            tok, pos = torch.where(idx == e)
            # w13 rows are interleaved (gate, up) pairs.
            gate_up = F.linear(xq[tok], self.w13[e].to(xq.dtype)).to(x.dtype).float()
            gate, up = gate_up[:, 0::2], gate_up[:, 1::2]
            h = self._softcap(gate, self.gate_cap) * torch.sigmoid(gate)
            h = (h * self._softcap(up, self.up_cap)).to(x.dtype)
            routed = weights[tok, pos, None]
            if self.act_quant:
                out.index_add_(0, tok, F.linear(fq(h.float() * routed), self.w2[e]))
            else:
                down = F.linear(h, self.w2[e].to(x.dtype)).float()
                out.index_add_(0, tok, down * routed)
        return out.to(x.dtype)
