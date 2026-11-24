# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini's own drafter (DSpark, `mtp.safetensors`) on Paiton's native kernels (gfx1201).

The draft model of vLLM main (`vllm/models/inkling2/nvidia/dspark.py`) for vLLM 0.32's DSpark speculator: an anchor
and seven mask rows go through three dense layers that attend over keys made from the target's hidden states, then
a recurrent Markov head picks the seven draft tokens one after the other.

Every operation on the card is a library call:
- `PAITON_INK2_DSPARK_SO` (ABI 1 or 2): the MLP of a draft layer, the Markov head (context, step with the vocabulary
  product, soft cap and argmax, token read-out; ABI 2 also returns the probability of each draft token) and the
  fold of `hidden_out` into the unembedding;
- `PAITON_INK2_DRAFT_ATTN_SO` (else `PAITON_INK2_ATTN_SO`; attention ABI 2 with the draft entries): context keys
  and the block attention of a layer, both with their norms and RoPE;
- the operator library (`native_ops`, a build with `MixRmsNorm` and `AddRmsNorm`): RMS norms, the mix of the four
  target features and the residual add with their norms, the anchor's normed embedding, the linear layers.
PyTorch allocates buffers and indexes them. No Triton module is imported.
"""

import ctypes as C
import os
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.rotary_embedding import get_rope

from ..configs import Inkling2DSparkConfig
from . import native_ops
from .attention import Inkling2AttentionPassCache
from .native_attention import NativeAttention, _Batch, _Cache, _int32, _ok, _rows, _stream

BF16, F32 = torch.bfloat16, torch.float32

# PAITON_INK2_DRAFT_LOGITS: the draft's base logits (7 rows over the whole unembedding per request and step).
# bf16 (default): the operator library's Linear. wide: its LinearBf16 (the LM head's kernel form, other fp32
# summation order). int8: int8 rows with one scale per row (half the bytes).
DRAFT_LOGITS = os.environ.get("PAITON_INK2_DRAFT_LOGITS", "bf16")
if DRAFT_LOGITS not in ("bf16", "wide", "int8"):
    raise ValueError("PAITON_INK2_DRAFT_LOGITS must be bf16, wide or int8")


class _DraftLayer(C.Structure):  # PaitonInkling2DraftLayer
    _fields_ = [("q_norm", C.c_void_p), ("k_norm", C.c_void_p), ("k_norm_ctx", C.c_void_p),
                ("cos_sin", C.c_void_p), ("window", C.c_int)]


_libs = None


def _libraries():
    """The attention library with the draft entries and the DSpark library."""
    global _libs
    if _libs is None:
        attn_path = os.environ.get("PAITON_INK2_DRAFT_ATTN_SO") or os.environ.get("PAITON_INK2_ATTN_SO")
        dspark_path = os.environ.get("PAITON_INK2_DSPARK_SO")
        if not attn_path or not dspark_path:
            raise ValueError("the drafter needs PAITON_INK2_DSPARK_SO and PAITON_INK2_DRAFT_ATTN_SO")
        attn, dspark = C.CDLL(attn_path), C.CDLL(dspark_path)
        if attn.PaitonInkling2AttnAbiVersion() != 2 or not hasattr(attn, "PaitonInkling2DraftDecode"):
            raise RuntimeError(f"{attn_path}: need attention ABI 2 with the draft entries")
        if dspark.PaitonInkling2DsparkGetAbiVersion() not in (1, 2):
            raise RuntimeError(f"{dspark_path}: DSpark ABI {dspark.PaitonInkling2DsparkGetAbiVersion()}, need 1 or 2")
        p, i, i64, z = C.c_void_p, C.c_int, C.c_int64, C.c_size_t
        attn.PaitonInkling2AttnWorkspaceBytes.restype = z
        attn.PaitonInkling2AttnWorkspaceBytes.argtypes = [i, i, p]
        attn.PaitonInkling2DraftContext.argtypes = [C.POINTER(_DraftLayer), C.POINTER(_Cache), p, i64, p, i64, p, p, i, p]
        attn.PaitonInkling2DraftDecode.argtypes = [C.POINTER(_DraftLayer), C.POINTER(_Batch), C.POINTER(_Cache),
                                                   p, i64, p, i64, p, i64, p, p, p, p, z, i, p]
        dspark.PaitonInkling2DsparkMlp.argtypes = [p] * 6 + [i, p]
        dspark.PaitonInkling2DsparkMarkovContext.argtypes = [p] * 5 + [i, i, p]
        dspark.PaitonInkling2DsparkMarkovStep.argtypes = [p] * 9 + [z, p, p, C.c_float, i, i, p, z, p, i, p]
        dspark.PaitonInkling2DsparkTokens.argtypes = [p, p, i, p]
        dspark.PaitonInkling2DsparkFold.argtypes = [p] * 5 + [i, p]
        if hasattr(dspark, "PaitonInkling2DsparkMarkovStepProb"):  # ABI 2: the step with the probability sums
            dspark.PaitonInkling2DsparkMarkovStepProb.argtypes = [p] * 9 + [z, p, p, C.c_float, i, i, p, z, p, p, i, p]
            dspark.PaitonInkling2DsparkTokenProbs.argtypes = [p, p, C.c_float, p, p, i, p]
        _libs = attn, dspark
    return _libs


def _upload(dest: torch.Tensor, src: torch.Tensor, stage: torch.Tensor) -> None:
    """Copy a CPU tensor to the card through the pinned staging buffer (no large pageable copy)."""
    dest, src = dest.reshape(-1), src.reshape(-1)
    for start in range(0, src.numel(), stage.numel()):
        part = src[start : start + stage.numel()]
        stage[: part.numel()].copy_(part)
        dest[start : start + part.numel()].copy_(stage[: part.numel()])


def _param(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=BF16), requires_grad=False)


class Inkling2DraftLayer(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig, is_local: bool, block: int, prefix: str) -> None:
        super().__init__()
        hidden, head, mlp = config.hidden_size, config.head_dim, config.intermediate_size
        self.q_size = config.num_attention_heads * head
        self.kv_size = config.num_key_value_heads * head
        self.input_layernorm, self.post_attention_layernorm, self.dflare_norm = (_param(hidden) for _ in range(3))
        self.qkv = _param(self.q_size + 2 * self.kv_size, hidden)  # rows of q_proj, k_proj, v_proj
        self.kv_ctx = _param(2 * self.kv_size, hidden)  # rows of k_proj_ctx, v_proj_ctx
        self.o_proj = _param(hidden, self.q_size)
        self.gate_proj, self.up_proj, self.down_proj = _param(mlp, hidden), _param(mlp, hidden), _param(hidden, mlp)
        self.q_norm, self.k_norm, self.k_norm_ctx = (_param(head) for _ in range(3))
        # Owns the paged KV cache only, with the spec of the target's layers of the same kind.
        self.cache = Inkling2AttentionPassCache(
            f"{prefix}.attn", config.num_attention_heads, config.num_key_value_heads, head,
            config.sliding_window_size if is_local else None)
        # Keys a block sees: itself and, on local layers, the window before it.
        self.window = config.sliding_window_size + block if is_local else 0
        self.native: _DraftLayer | None = None


class Inkling2DSparkDecoder(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig, max_model_len: int, block: int, prefix: str) -> None:
        super().__init__()
        self.config = config
        self.block = block  # rows per request: the anchor and the masks
        hidden, rank = config.hidden_size, config.markov_rank
        self.layers = nn.ModuleList(
            Inkling2DraftLayer(config, i in config.local_layer_ids, block, f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers))
        self.norm, self.mask_embed, self.markov_dflare_norm = _param(hidden), _param(hidden), _param(hidden)
        self.fuse_weight = _param(config.num_hidden_layers + 1, len(config.target_layer_ids))
        self.markov_w1 = _param(config.padded_vocab_size, rank)
        self.rnn_proj, self.rnn_bias = _param(3 * rank, 2 * rank + 2 * hidden), _param(3 * rank)
        self.hidden_out, self.hidden_out_bias = _param(hidden, rank), _param(hidden)
        # fp32 [positions][cos 64 | sin 64], read by the attention library. Built on the host.
        # shortcut: the table has max_model_len rows (512 MB at one million positions), compute it per call in the
        # kernel before serving contexts far above 100K tokens.
        with torch.device("cpu"):
            table = get_rope(config.head_dim, max_position=max_model_len, dtype=F32,
                             rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta}).cos_sin_cache
        self.cos_sin = table.to(dtype=F32, device=self.norm.device).contiguous()
        self.fuse_alpha: torch.Tensor | None = None
        self.workspace = torch.empty(0, dtype=torch.uint8)  # of the attention library, grown on demand
        # The target's embedding (set by vLLM's load_dspark_model) and its norm (set by attach_target).
        self.embed_tokens: nn.Module | None = None
        self.embed_norm: nn.Module | None = None

    def checkpoint_map(self) -> dict[str, tuple[nn.Parameter, int]]:
        """Tensor name in `mtp.safetensors` -> (parameter, first row it fills)."""
        m = {"fuse_weight": self.fuse_weight, "mask_embed": self.mask_embed, "norm.weight": self.norm,
             "markov_head.dflare_norm.weight": self.markov_dflare_norm,
             "markov_head.markov_w1.weight": self.markov_w1, "markov_head.rnn_proj.weight": self.rnn_proj,
             "markov_head.rnn_proj.bias": self.rnn_bias, "markov_head.hidden_out.weight": self.hidden_out,
             "markov_head.hidden_out.bias": self.hidden_out_bias}
        out = {name: (param, 0) for name, param in m.items()}
        for i, layer in enumerate(self.layers):
            p, a = f"layers.{i}.", f"layers.{i}.self_attn."
            for name in ("input_layernorm", "post_attention_layernorm", "dflare_norm"):
                out[f"{p}{name}.weight"] = (getattr(layer, name), 0)
            for name in ("q_norm", "k_norm", "k_norm_ctx", "o_proj"):
                out[f"{a}{name}.weight"] = (getattr(layer, name), 0)
            for name in ("gate_proj", "up_proj", "down_proj"):
                out[f"{p}mlp.{name}.weight"] = (getattr(layer, name), 0)
            out[a + "q_proj.weight"] = (layer.qkv, 0)
            out[a + "k_proj.weight"] = (layer.qkv, layer.q_size)
            out[a + "v_proj.weight"] = (layer.qkv, layer.q_size + layer.kv_size)
            out[a + "k_proj_ctx.weight"] = (layer.kv_ctx, 0)
            out[a + "v_proj_ctx.weight"] = (layer.kv_ctx, layer.kv_size)
        return out

    def mix(self, features: torch.Tensor, row: int, weight: torch.Tensor) -> torch.Tensor:
        """RMS norm of the `fuse_alpha[row]`-weighted sum of the target features, `(T, taps * hidden)` in."""
        assert self.fuse_alpha is not None
        return native_ops.mix_rms_norm(features, self.fuse_alpha[row], weight, self.config.rms_norm_eps)

    def _add_norm(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """In place `residual += x`; returns the RMS norm of the unrounded sum."""
        return native_ops.add_rms_norm(x, residual, weight, self.config.rms_norm_eps)

    def _attention(self, layer: Inkling2DraftLayer, qkv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Block attention of one layer over its cache; `(T, q_size)`, the input of `o_proj`."""
        out = torch.zeros((qkv.shape[0], layer.q_size), dtype=BF16, device=qkv.device)
        attn_metadata = get_forward_context().attn_metadata
        kv_cache = layer.cache.kv_cache
        if not isinstance(attn_metadata, dict) or kv_cache.numel() == 0:
            return out  # memory profiling run: no cache yet
        md = attn_metadata[layer.cache.prefix]
        block = self.block
        reqs = min(qkv.shape[0] // block, md.seq_lens.shape[0])
        tokens = reqs * block
        if tokens == 0:
            return out
        lib = _libraries()[0]
        block_table, seq_lens = _int32(md.block_table[:reqs]), _int32(md.seq_lens[:reqs])
        query_start = _int32(md.query_start_loc[: reqs + 1])
        slots, positions = md.slot_mapping[:tokens], positions[:tokens]
        assert slots.dtype == positions.dtype == torch.int64 and slots.is_contiguous() and positions.is_contiguous()
        batch = _Batch(block_table.data_ptr(), seq_lens.data_ptr(), query_start.data_ptr(), block_table.shape[1],
                       reqs, tokens, block)
        splits = 0  # the library's default: a request's rows do not depend on the other requests of the call
        need = lib.PaitonInkling2AttnWorkspaceBytes(tokens, 8, None)
        if self.workspace.numel() < need:
            self.workspace = torch.empty(need, dtype=torch.uint8, device=qkv.device)
        q, k, v = qkv[:tokens].split([layer.q_size, layer.kv_size, layer.kv_size], dim=1)
        cache = NativeAttention._cache(kv_cache)
        _ok(lib.PaitonInkling2DraftDecode(
            C.byref(layer.native), C.byref(batch), C.byref(cache), *_rows(q, layer.q_size),
            *_rows(k, layer.kv_size), *_rows(v, layer.kv_size), positions.data_ptr(), slots.data_ptr(),
            out.data_ptr(), self.workspace.data_ptr(), self.workspace.numel(), splits, _stream(qkv.device)),
            "PaitonInkling2DraftDecode")
        return out

    def _mlp(self, layer: Inkling2DraftLayer, x: torch.Tensor) -> torch.Tensor:
        rows = x.shape[0]
        h = torch.empty((rows, layer.gate_proj.shape[0]), dtype=BF16, device=x.device)
        out = torch.empty((rows, layer.down_proj.shape[0]), dtype=BF16, device=x.device)
        _ok(_libraries()[1].PaitonInkling2DsparkMlp(
            x.data_ptr(), layer.gate_proj.data_ptr(), layer.up_proj.data_ptr(), layer.down_proj.data_ptr(),
            h.data_ptr(), out.data_ptr(), rows, _stream(x.device)), "PaitonInkling2DsparkMlp")
        return out

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The target's normed embedding: lookup and embed norm in one launch."""
        return native_ops.embed_norm(input_ids, self.embed_tokens.weight, self.embed_norm.weight[0],
                                     self.embed_norm.eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Hidden rows `(T, hidden)` of the blocks: per request the anchor, then the masks."""
        block = self.block
        ids = input_ids.view(-1, block)
        x = self.mask_embed.expand(ids.shape[0], block, -1).clone()
        x[:, 0] = self.embed(ids[:, 0].contiguous())
        x = x.view(-1, self.config.hidden_size)
        residual = None
        for layer in self.layers:
            if residual is None:
                residual = x
                x = native_ops.rms_norm(x, layer.input_layernorm, self.config.rms_norm_eps)
            else:
                x = self._add_norm(x, residual, layer.input_layernorm)
            x = self._attention(layer, native_ops.linear(x, layer.qkv), positions)
            x = self._add_norm(native_ops.linear(x, layer.o_proj), residual, layer.post_attention_layernorm)
            x = self._mlp(layer, x)
        return self._add_norm(x, residual, self.norm)


class Inkling2DSparkModel(nn.Module):
    """DSpark draft of Inkling 2; plugged in by `Inkling2DSparkSpeculator`."""

    allow_patterns_overrides = ["mtp.safetensors"]
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if os.environ.get("PAITON_INK2_OPS") != "native":
            raise NotImplementedError("the drafter runs on the native operators only (PAITON_INK2_OPS=native)")
        spec = vllm_config.speculative_config
        assert spec is not None
        config = spec.draft_model_config.hf_config
        assert isinstance(config, Inkling2DSparkConfig)
        # The checkpoint's block is an anchor and 7 masks. A shorter block (fewer masks) drafts fewer tokens for a
        # cheaper verify call; vLLM main allows the full block only.
        if not 1 <= spec.num_speculative_tokens < config.draft_block_size:
            raise ValueError("Inkling 2 DSpark drafts 1 to "
                             f"{config.draft_block_size - 1} tokens (the checkpoint's block size - 1)")
        if vllm_config.model_config.max_model_len > config.max_position_embeddings:
            raise ValueError(f"Inkling 2 DSpark drafts up to {config.max_position_embeddings} positions; "
                             "set --max-model-len to at most that")
        self.config = config
        # vLLM 0.32 builds the draft without a prefix: the cache layers need names apart from the target's.
        self.model = Inkling2DSparkDecoder(config, vllm_config.model_config.max_model_len,
                                           spec.num_speculative_tokens + 1, f"{prefix}dspark")
        self.lm_head: nn.Module | None = None  # the target's, set by vLLM's load_dspark_model

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        todo = self.model.checkpoint_map()
        stage = torch.empty(16 << 20, dtype=BF16).pin_memory()  # 32 MB of pinned host memory
        for name, weight in weights:
            param, row = todo.pop(name)
            dest = param.data[row : row + weight.shape[0]] if weight.dim() == 2 else param.data
            if weight.dtype != BF16 or dest.shape != weight.shape:
                raise ValueError(f"{name}: expected bf16 {tuple(dest.shape)}, got {weight.dtype} {tuple(weight.shape)}")
            _upload(dest, weight, stage)
        if todo:
            raise ValueError(f"mtp.safetensors lacks {sorted(todo)}")
        return {name for name, _ in self.named_parameters()}

    @torch.no_grad()
    def attach_target(self, target_model: nn.Module) -> None:
        """After the weights are on the card and vLLM has shared the target's embedding and LM head: derive what
        depends on them and hand the layer constants to the attention library."""
        model = self.model
        assert self.lm_head is not None and self.lm_head.weight.dtype == BF16
        assert model.embed_tokens is target_model.model.embed_tokens
        model.embed_norm = target_model.model.embed_norm
        device = model.norm.device
        model.fuse_alpha = torch.softmax(model.fuse_weight.float().cpu(), dim=-1).to(device)
        if not (native_ops.has("MixRmsNorm") and native_ops.has("AddRmsNorm")):
            raise ValueError("the drafter needs an operator library with MixRmsNorm and AddRmsNorm (PAITON_INK2_OPS_SO)")
        for layer in model.layers:
            layer.native = _DraftLayer(layer.q_norm.data_ptr(), layer.k_norm.data_ptr(), layer.k_norm_ctx.data_ptr(),
                                       model.cos_sin.data_ptr(), layer.window)
        head = self.lm_head.weight
        assert head.is_contiguous() and head.shape[0] % 16 == 0
        self.fold_weight = torch.empty((head.shape[0], self.config.markov_rank), dtype=BF16, device=device)
        self.fold_bias = torch.empty((head.shape[0],), dtype=BF16, device=device)
        _ok(_libraries()[1].PaitonInkling2DsparkFold(
            head.data_ptr(), model.hidden_out.data_ptr(), model.hidden_out_bias.data_ptr(),
            self.fold_weight.data_ptr(), self.fold_bias.data_ptr(), head.shape[0], _stream(device)),
            "PaitonInkling2DsparkFold")
        self.head_rows: tuple[torch.Tensor, torch.Tensor] | None = None
        self.__dict__["target"] = target_model  # not a submodule: only its int8 head rows are read

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.cache.prefix for layer in self.model.layers]

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states  # each draft layer mixes the raw target features itself

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        """Keys and values of the context rows (`(T, taps * hidden)` target features) into every layer's cache."""
        tokens = context_states.shape[0]
        if context_slot_mappings is None or tokens == 0:
            return
        lib = _libraries()[0]
        assert context_positions.dtype == torch.int64 and context_positions.is_contiguous()
        for i, (layer, slots) in enumerate(zip(self.model.layers, context_slot_mappings, strict=True)):
            if slots is None or layer.cache.kv_cache.numel() == 0:
                continue
            assert slots.dtype == torch.int64 and slots.is_contiguous() and slots.shape[0] == tokens
            kv = native_ops.linear(self.model.mix(context_states, i, layer.dflare_norm), layer.kv_ctx)
            k, v = kv.split(layer.kv_size, dim=1)
            cache = NativeAttention._cache(layer.cache.kv_cache)
            _ok(lib.PaitonInkling2DraftContext(
                C.byref(layer.native), C.byref(cache), *_rows(k, layer.kv_size), *_rows(v, layer.kv_size),
                context_positions.data_ptr(), slots.data_ptr(), tokens, _stream(kv.device)),
                "PaitonInkling2DraftContext")

    def compute_f_rnn(self, features: torch.Tensor) -> torch.Tensor:
        """The Markov head's context from the last committed token's features."""
        return self.model.mix(features, -1, self.model.markov_dflare_norm)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids, positions)

    def _int8_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        """int8 rows of every padded row of the unembedding: the target's when its LM head is int8, else a copy."""
        if self.head_rows is None:
            from . import model as target_module

            target, head = self.__dict__["target"], self.lm_head.weight
            if target_module.LMHEAD == "int8" and hasattr(target, "_lm_head_rows"):
                if target._lm_head_rows is None:  # the target makes them at its first logits call: the same call
                    target._lm_head_rows = native_ops.head_rows(head, "int8")
                self.head_rows = target._lm_head_rows
            if self.head_rows is None or self.head_rows[0].shape[0] != head.shape[0]:
                self.head_rows = native_ops.head_rows(head, "int8")  # an own copy, 463 MB
        return self.head_rows

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """bf16 base logits over all rows of the unembedding (a bf16 product, as the reference)."""
        if DRAFT_LOGITS == "int8":
            return native_ops.linear_rows(hidden_states, *self._int8_rows(), "int8")
        if DRAFT_LOGITS == "wide":
            return native_ops.linear_wide(hidden_states, self.lm_head.weight)
        return native_ops.linear(hidden_states, self.lm_head.weight)

    def markov_tokens(self, hidden: torch.Tensor, f_rnn: torch.Tensor, anchors: torch.Tensor,
                      probabilities: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Greedy draft tokens, int32 `(steps, requests)`. `hidden`: the mask rows in (request, step) order;
        `f_rnn` `(requests, hidden)`; `anchors`: the token before the first draft, per request. With
        `probabilities` also fp32 `(steps, requests)`: the draft distribution's probability of each draft token
        (the library's own sums over the soft-capped logits; needs DSpark ABI 2)."""
        lib, model, device = _libraries()[1], self.model, hidden.device
        if probabilities and not hasattr(lib, "PaitonInkling2DsparkMarkovStepProb"):
            raise ValueError("draft probabilities need a DSpark library with MarkovStepProb (ABI 2)")
        reqs = f_rnn.shape[0]
        steps = hidden.shape[0] // reqs
        rank, stream = self.config.markov_rank, _stream(device)
        softcap = self.config.final_logit_softcapping
        base = self.compute_logits(hidden)
        vocab_rows = base.shape[1]
        context = torch.empty((steps, reqs, 3 * rank), dtype=F32, device=device)
        _ok(lib.PaitonInkling2DsparkMarkovContext(
            hidden.data_ptr(), f_rnn.data_ptr(), model.rnn_proj.data_ptr(), model.rnn_bias.data_ptr(),
            context.data_ptr(), reqs, steps, stream), "PaitonInkling2DsparkMarkovContext")
        state = torch.zeros((2, reqs, rank), dtype=BF16, device=device)
        latent = torch.empty((reqs, rank), dtype=BF16, device=device)
        keys = torch.zeros((steps, reqs), dtype=torch.int64, device=device)  # the library's 64-bit argmax keys
        sums = torch.zeros((steps, reqs, 2), dtype=torch.int64, device=device) if probabilities else None
        tokens = torch.empty((steps, reqs), dtype=torch.int32, device=device)
        anchors = _int32(anchors)
        for i in range(steps):
            # The step reads the previous token from the keys of the step before: no host round trip.
            step = (context[i].data_ptr(), state[i % 2].data_ptr(), state[1 - i % 2].data_ptr(), latent.data_ptr(),
                    anchors.data_ptr() if i == 0 else None, keys[i - 1].data_ptr() if i else None,
                    model.markov_w1.data_ptr(), model.rnn_proj.data_ptr(), base.data_ptr() + i * vocab_rows * 2,
                    steps * vocab_rows, self.fold_weight.data_ptr(), self.fold_bias.data_ptr(),
                    softcap, self.config.vocab_size, vocab_rows, None, 0, keys[i].data_ptr())
            if sums is None:
                _ok(lib.PaitonInkling2DsparkMarkovStep(*step, reqs, stream), "PaitonInkling2DsparkMarkovStep")
            else:
                _ok(lib.PaitonInkling2DsparkMarkovStepProb(*step, sums[i].data_ptr(), reqs, stream),
                    "PaitonInkling2DsparkMarkovStepProb")
        if sums is None:
            _ok(lib.PaitonInkling2DsparkTokens(keys.data_ptr(), tokens.data_ptr(), steps * reqs, stream),
                "PaitonInkling2DsparkTokens")
            return tokens
        probs = torch.empty((steps, reqs), dtype=F32, device=device)
        _ok(lib.PaitonInkling2DsparkTokenProbs(keys.data_ptr(), sums.data_ptr(), softcap, tokens.data_ptr(),
                                               probs.data_ptr(), steps * reqs, stream), "PaitonInkling2DsparkTokenProbs")
        return tokens, probs
