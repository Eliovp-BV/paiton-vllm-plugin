"""Kimi-K3.1-Pro forward pass in plain PyTorch.

Everything here is straightforward PyTorch: weights are unpacked from MXFP4 on every use, attention is computed in
full fp32, and the KDA recurrence is a token-by-token loop. It is slow on purpose: it is meant to be easy to read
and to run anywhere PyTorch runs, not to be fast.

Layers are split evenly over the visible GPUs. Each GPU keeps its layers' attention and dense weights. Expert weights
go on the GPU while they fit; the rest are read from the checkpoint in host memory when a token routes to them.
"""
from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass

import torch
import torch.nn.functional as F

FP4_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """MXFP4: uint8 [out, in/2] (low nibble = even column) with e8m0 uint8 scales [out, in/32]."""
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    q = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    vals = FP4_E2M1.to(packed.device)[q]
    s = torch.exp2(scale.float() - 127.0)
    vals = vals.view(packed.shape[0], -1, 32) * s.unsqueeze(-1)
    return vals.reshape(packed.shape[0], -1).to(dtype)


class Checkpoint:
    """Lazy access to a sharded safetensors checkpoint."""

    def __init__(self, path: str):
        from safetensors import safe_open
        self.path = path
        idx = os.path.join(path, "model.safetensors.index.json")
        files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
        self.map, self.nbytes = {}, {}
        for fn in files:
            with open(os.path.join(path, fn), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            h.pop("__metadata__", None)
            for k, v in h.items():
                self.map[k] = fn
                self.nbytes[k] = v["data_offsets"][1] - v["data_offsets"][0]
        self._open = {}
        self._safe_open = safe_open

    def has(self, k):
        return k in self.map

    def get(self, k) -> torch.Tensor:
        fn = self.map[k]
        if fn not in self._open:
            self._open[fn] = self._safe_open(os.path.join(self.path, fn), framework="pt")
        return self._open[fn].get_tensor(k)


def rms(x, w, eps):
    x32 = x.float()
    return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def situ_glu(gate, up, beta, lin_beta):
    a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if lin_beta is not None:
        up = lin_beta * torch.tanh(up / lin_beta)
    return a * up


def attn_res(prefix, blocks, proj_w, norm_w, eps):
    """Attention residuals: prefix [T,D], blocks [T,B,D]; softmax over the sources, then mix them."""
    v = torch.cat([blocks, prefix.unsqueeze(1)], 1).float()
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    scores = (k * (norm_w.float() * proj_w.squeeze(0).float())).sum(-1)
    return torch.einsum("tb,tbd->td", scores.softmax(-1), v)


@dataclass
class Cfg:
    raw: dict

    def __getattr__(self, k):
        try:
            return self.raw[k]
        except KeyError as e:
            raise AttributeError(k) from e


EXPERT_RE = re.compile(r"\.experts\.\d+\.")


class KimiK31:
    P = "language_model.model."

    def __init__(self, path: str, devices: list[str], reserve_gb: float = 40.0, layers: int | None = None, log=print):
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.c = Cfg(cfg["text_config"])
        self.w = Checkpoint(path)
        self.n_layers = layers or self.c.num_hidden_layers
        self.kda = set(self.c.linear_attn_config["kda_layers"])
        self.eps = self.c.rms_norm_eps
        self.situ = (self.c.raw.get("activation_situ_beta") or 1.0, self.c.raw.get("activation_situ_linear_beta"))
        self.devices = devices
        self.dev = devices[0]
        self.res: dict[str, torch.Tensor] = {}
        self._place(reserve_gb, log)

    # ---- placement ----
    def _layer_keys(self):
        per = {i: [] for i in range(self.n_layers)}
        other = []
        for k in self.w.map:
            if not k.startswith("language_model."):
                continue
            m = re.match(r"language_model\.model\.layers\.(\d+)\.", k)
            if m:
                i = int(m.group(1))
                if i < self.n_layers:
                    per[i].append(k)
            else:
                other.append(k)
        return per, other

    def _place(self, reserve_gb, log):
        per, other = self._layer_keys()
        n_dev = len(self.devices)
        base, extra = divmod(self.n_layers, n_dev)
        self.layer_dev, i = [], 0
        for d in range(n_dev):
            self.layer_dev += [self.devices[d]] * (base + (1 if d < extra else 0))
        on_gpu_layers = 0
        for d, dev in enumerate(self.devices):
            mine = [i for i in range(self.n_layers) if self.layer_dev[i] == dev]
            keys = [k for i in mine for k in per[i] if not EXPERT_RE.search(k)]
            if d == 0:
                keys += [k for k in other if "embed_tokens" in k]
            if d == n_dev - 1:
                keys += [k for k in other if "embed_tokens" not in k]
            free = torch.cuda.mem_get_info(torch.device(dev))[0] if dev.startswith("cuda") else 1 << 62
            budget = free - reserve_gb * 2 ** 30 - sum(self.w.nbytes[k] for k in keys)
            for i in mine:
                ek = [k for k in per[i] if EXPERT_RE.search(k)]
                need = sum(self.w.nbytes[k] for k in ek)
                if ek and need <= budget:
                    keys += ek
                    budget -= need
                    on_gpu_layers += 1
            for k in keys:
                self.res[k] = self.w.get(k).to(dev)
            log(f"{dev}: layers {mine[0]}-{mine[-1]}, {len(keys)} tensors resident, "
                f"{torch.cuda.memory_allocated(torch.device(dev)) / 2**30:.0f} GiB" if dev.startswith("cuda") else dev)
        moe_layers = self.n_layers - self.c.first_k_dense_replace
        log(f"experts on GPU for {on_gpu_layers} of {moe_layers} MoE layers; the rest are read from host memory")

    # ---- weight access ----
    def _fetch(self, key):
        t = self.res.get(key)
        if t is None:
            t = self.w.get(key)
        return t.to(self.dev)

    def W(self, name, dtype=torch.float32):
        full = self.P + name
        if self.w.has(full):
            return self._fetch(full).to(dtype)
        if self.w.has(full + ".weight"):
            return self._fetch(full + ".weight").to(dtype)
        return dequant_mxfp4(self._fetch(full + ".weight_packed"), self._fetch(full + ".weight_scale"), dtype)

    def lin(self, x, name):
        return x @ self.W(name).t()

    # ---- blocks ----
    def mlp(self, x, pre):
        g = self.lin(x, pre + ".gate_proj")
        u = self.lin(x, pre + ".up_proj")
        return self.lin(situ_glu(g, u, *self.situ), pre + ".down_proj")

    def moe(self, x, pre):
        c = self.c
        logits = x.float() @ self.W(pre + ".gate.weight").t()
        scores = logits.sigmoid()
        choice = scores + self.W(pre + ".gate.e_score_correction_bias")
        idx = choice.topk(c.num_experts_per_token, -1).indices
        wts = scores.gather(1, idx)
        if c.moe_renormalize:
            wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
        wts = wts * c.routed_scaling_factor
        lat = self.lin(x, pre + ".routed_expert_down_proj")
        y = torch.zeros_like(lat)
        for e in idx.unique().tolist():
            rows, slot = (idx == e).nonzero(as_tuple=True)
            h = lat[rows]
            ep = f"{pre}.experts.{e}"
            a = situ_glu(self.lin(h, ep + ".w1"), self.lin(h, ep + ".w3"), *self.situ)
            y.index_add_(0, rows, self.lin(a, ep + ".w2") * wts[rows, slot].unsqueeze(-1))
        if c.latent_moe_use_norm:
            y = rms(y, self.W(pre + ".routed_expert_norm.weight"), self.eps)
        y = self.lin(y, pre + ".routed_expert_up_proj")
        if c.num_shared_experts:
            y = y + self.mlp(x, pre + ".shared_experts")
        return y

    def kda_attn(self, x, pre, state=None):
        """Gated delta rule (KDA), one token at a time. x [T,D]; returns out, (conv_tail, S)."""
        la = self.c.linear_attn_config
        H, dk = la["num_heads"], la["head_dim"]
        T = x.shape[0]
        K = la["short_conv_kernel_size"]
        conv_prev = state[0] if state else {n: torch.zeros(K - 1, H * dk, device=self.dev) for n in "qkv"}
        S = state[1] if state else torch.zeros(H, dk, dk, device=self.dev)
        outs, tails = {}, {}
        for n in "qkv":
            p = self.lin(x, f"{pre}.{n}_proj")
            w = self.W(f"{pre}.{n}_conv1d.weight").squeeze(1)
            full = torch.cat([conv_prev[n], p], 0)
            y = sum(full[i:i + T] * w[:, i] for i in range(K))
            outs[n] = F.silu(y)
            tails[n] = full[-(K - 1):]
        q = F.normalize(outs["q"].view(T, H, dk), dim=-1) * dk ** -0.5
        k = F.normalize(outs["k"].view(T, H, dk), dim=-1)
        v = outs["v"].view(T, H, dk)
        f = self.lin(self.lin(x, pre + ".f_a_proj"), pre + ".f_b_proj").view(T, H, dk)
        # A_log holds one decay scale per head. The checkpoint stores it padded: 128 entries for 96 heads, entries 0-95 are the
        # heads' values and the rest are zero. Entry h belongs to head h (the vLLM loader reads it the same way); reading the
        # 128 entries as per-channel values shared by all heads gives every KDA layer wrong decay rates, and the error grows
        # with context length.
        A = self.W(pre + ".A_log").flatten()
        assert A.numel() >= H, f"{pre}.A_log has {A.numel()} entries for {H} heads"
        A = A.view(1, H, dk).exp() if A.numel() == H * dk else A[:H].exp().view(1, H, 1)
        g = la["gate_lower_bound"] * torch.sigmoid(A * (f + self.W(pre + ".dt_bias").view(1, H, dk)))
        beta = torch.sigmoid(self.lin(x, pre + ".b_proj"))
        o = torch.empty(T, H, dk, device=self.dev)
        for t in range(T):
            S = S * g[t].exp().unsqueeze(-1)
            err = v[t] - torch.einsum("hk,hkv->hv", k[t], S)
            S = S + beta[t].view(H, 1, 1) * k[t].unsqueeze(-1) * err.unsqueeze(1)
            o[t] = torch.einsum("hk,hkv->hv", q[t], S)
        gate = self.lin(x, pre + ".g_proj").view(T, H, dk)
        o = rms(o, self.W(pre + ".o_norm.weight"), self.eps) * torch.sigmoid(gate)
        return self.lin(o.reshape(T, H * dk), pre + ".o_proj"), (tails, S)

    def indexer_topk(self, x, q_c, pre, kcache):
        """DSA indexer: picks the top index_topk earlier positions per query. Returns a [T, S] bool mask."""
        d = self.c.dsa_config
        Hi, di = d["index_num_heads"], d["index_head_dim"]
        T = x.shape[0]
        q = self.lin(q_c, pre + ".indexer.wq_b").view(T, Hi, di)
        k = F.layer_norm(self.lin(x, pre + ".indexer.wk"), (di,),
                         self.W(pre + ".indexer.k_norm.weight"), self.W(pre + ".indexer.k_norm.bias"), 1e-6)
        kall = torch.cat([kcache, k], 0) if kcache is not None else k
        wts = self.lin(x, pre + ".indexer.weights_proj") * Hi ** -0.5
        s = torch.einsum("thd,sd->ths", q, kall).relu() * di ** -0.5
        score = torch.einsum("ths,th->ts", s, wts)
        S = kall.shape[0]
        pos = torch.arange(S - T, S, device=self.dev).unsqueeze(1)
        causal = torch.arange(S, device=self.dev).unsqueeze(0) <= pos
        score = score.masked_fill(~causal, float("-inf"))
        kk = min(d["index_topk"], S)
        top = score.topk(kk, -1).indices
        m = torch.zeros(T, S, dtype=torch.bool, device=self.dev).scatter_(1, top, True)
        return m & causal, kall

    def mla_attn(self, x, pre, cache=None):
        c = self.c
        H = c.num_attention_heads
        dn, dr, dv, r = c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim, c.kv_lora_rank
        T = x.shape[0]
        q_c = rms(self.lin(x, pre + ".q_a_proj"), self.W(pre + ".q_a_layernorm.weight"), self.eps)
        q = self.lin(q_c, pre + ".q_b_proj").view(T, H, dn + dr)
        kv = self.lin(x, pre + ".kv_a_proj_with_mqa")
        c_kv = rms(kv[:, :r], self.W(pre + ".kv_a_layernorm.weight"), self.eps)
        lat = torch.cat([c_kv, kv[:, r:]], -1)
        lat_all = torch.cat([cache["lat"], lat], 0) if cache else lat
        S = lat_all.shape[0]
        kvb = (lat_all[:, :r] @ self.W(pre + ".kv_b_proj").t()).view(S, H, dn + dv)
        k = torch.cat([kvb[..., :dn], lat_all[:, r:].unsqueeze(1).expand(S, H, dr)], -1)
        v = kvb[..., dn:]
        sc = torch.einsum("thd,shd->hts", q, k) * (dn + dr) ** -0.5
        pos = torch.arange(S - T, S, device=self.dev).unsqueeze(1)
        mask = torch.arange(S, device=self.dev).unsqueeze(0) <= pos
        new_cache = {"lat": lat_all}
        if self.w.has(self.P + pre + ".indexer.wk.weight"):
            m, kidx = self.indexer_topk(x, q_c, pre, cache.get("idx") if cache else None)
            mask = mask & m
            new_cache["idx"] = kidx
        sc = sc.masked_fill(~mask.unsqueeze(0), float("-inf"))
        o = torch.einsum("hts,shd->thd", sc.softmax(-1), v).reshape(T, H * dv)
        o = o * torch.sigmoid(self.lin(x, pre + ".g_proj"))
        return self.lin(o, pre + ".o_proj"), new_cache

    # ---- forward ----
    @torch.no_grad()
    def forward(self, ids: torch.Tensor, state=None):
        """ids [T] -> logits of the last position [vocab], new state. state holds every layer's cache."""
        self.dev = self.devices[0]
        x = self._fetch(self.P + "embed_tokens.weight")[ids.to(self.dev)].float()
        c = self.c
        state = state if state is not None else {}
        bs = c.attn_res_block_size
        blocks = torch.zeros(x.shape[0], 0, x.shape[1], device=self.dev)
        prefix = x
        for i in range(self.n_layers):
            self.dev = self.layer_dev[i]
            blocks = blocks.to(self.dev)
            prefix = prefix.to(self.dev) if prefix is not None else None
            pre = f"layers.{i}"
            if blocks.shape[1] > 0:
                h = attn_res(prefix, blocks, self.W(pre + ".self_attention_res_proj.weight"),
                             self.W(pre + ".self_attention_res_norm.weight"), self.eps)
            else:
                h = prefix
            if i % bs == 0:
                blocks = torch.cat([blocks, prefix.unsqueeze(1)], 1)
                prefix = None
            h = rms(h, self.W(pre + ".input_layernorm.weight"), self.eps)
            if (i + 1) in self.kda:
                a, state[i] = self.kda_attn(h, pre + ".self_attn", state.get(i))
            else:
                a, state[i] = self.mla_attn(h, pre + ".self_attn", state.get(i))
            prefix = a if prefix is None else prefix + a
            h = attn_res(prefix, blocks, self.W(pre + ".mlp_res_proj.weight"),
                         self.W(pre + ".mlp_res_norm.weight"), self.eps)
            h = rms(h, self.W(pre + ".post_attention_layernorm.weight"), self.eps)
            if i >= c.first_k_dense_replace:
                m = self.moe(h, pre + ".block_sparse_moe")
            else:
                m = self.mlp(h, pre + ".mlp")
            prefix = prefix + m
        self.dev = self.devices[-1]
        prefix, blocks = prefix[-1:].to(self.dev), blocks[-1:].to(self.dev)
        if self.w.has(self.P + "output_attn_res_proj.weight"):
            prefix = attn_res(prefix, blocks, self.W("output_attn_res_proj.weight"),
                              self.W("output_attn_res_norm.weight"), self.eps)
        h = rms(prefix, self.W("norm.weight"), self.eps)
        head = self._fetch("language_model.lm_head.weight").float()
        return (h @ head.t())[0], state
