# SPDX-License-Identifier: Apache-2.0
"""A DFlash2 drafter for Inkling 2 Mini on Paiton's native drafter library (gfx1201).

The draft checkpoint is a model of its own (`model.safetensors` with a `dflash_config`): context keys from four
target features, a block of the bonus token and seven mask rows through five non-causal layers with grouped
convolutions, the 16 best candidates of the target's LM head per row and a selector that walks them. The
arithmetic follows vLLM 0.32's DFlash2 draft model (`qwen3_dflash2.py`, Apache-2.0) with the soft cap after the
selector's correction, as the checkpoint's config asks.

Two library calls do the work on the card (`PAITON_INK2_DFLASH2_SO`, ABI 1): the context keys of new target rows
and one draft step for every request. The library loads the operator library (`PAITON_INK2_OPS_SO`) itself.
PyTorch allocates the buffers. No Triton module is imported.
"""

import ctypes as C
import os
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.rotary_embedding import get_rope

from ..cache_plan import FULL_VISITS
from ..configs import Inkling2DFlash2Config
from . import native_ops
from .attention import Inkling2AttentionPassCache
from .draft import DRAFT_LOGITS, _param, _upload
from .native_attention import NativeAttention, _Batch, _Cache, _int32, _ok, _stream

BF16, F32 = torch.bfloat16, torch.float32
LAYERS = 5
_P = C.c_void_p


class _Layer(C.Structure):  # PaitonInkling2Dflash2Layer
    _fields_ = [(name, _P) for name in ("input_norm", "post_norm", "attn_proj", "attn_base", "mlp_proj", "mlp_base", "qkv",
                                        "o_proj", "gate", "up", "down", "q_norm", "k_norm")]


class _Model(C.Structure):  # PaitonInkling2Dflash2Model
    _fields_ = ([("layers", _Layer * LAYERS)]
                + [(name, _P) for name in ("fc", "hidden_norm", "norm", "hidden_projection", "pred_codebook",
                                           "succ_codebook", "embed", "embed_norm", "head_codes", "head_scale",
                                           "head_bf16", "cos_sin")]
                + [("embed_rows", C.c_int), ("vocab", C.c_int), ("window", C.c_int), ("softcap", C.c_float),
                   ("eps", C.c_float)])


class _Call(C.Structure):  # PaitonInkling2Dflash2Call
    _fields_ = [("input_ids", _P), ("positions", _P), ("slots", _P * LAYERS), ("batch", _Batch * LAYERS),
                ("cache", _Cache * LAYERS), ("workspace", _P), ("workspace_bytes", C.c_size_t), ("tokens", _P),
                ("walk", _P), ("scores", _P), ("trace", _P), ("requests", C.c_int), ("block", C.c_int),
                ("ids_i64", C.c_int), ("sample", _P)]


class _Sample(C.Structure):  # PaitonInkling2Dflash2Sample (ABI 2): vLLM's buffers for sampled drafts
    _fields_ = [("req_state", _P), ("temperature", _P), ("seeds", _P), ("sample_pos", _P), ("draft_logits", _P),
                ("stride_state", C.c_int64), ("stride_step", C.c_int64), ("cached_ids", _P)]


assert C.sizeof(_Sample) == 64 and _Call.sample.offset + 8 == C.sizeof(_Call)  # the library's layout


_lib = None


def _library():
    global _lib
    if _lib is None:
        path = os.environ.get("PAITON_INK2_DFLASH2_SO")
        if not path:
            raise ValueError("the DFlash2 drafter needs PAITON_INK2_DFLASH2_SO")
        lib = C.CDLL(path)
        if lib.PaitonInkling2Dflash2AbiVersion() not in (1, 2):  # 2 adds the sampled draw (`_Call.sample`)
            raise RuntimeError(f"{path}: DFlash2 ABI {lib.PaitonInkling2Dflash2AbiVersion()}, need 1 or 2")
        lib.PaitonInkling2Dflash2Prepare.argtypes = [C.POINTER(_Model), C.c_char_p, C.POINTER(_P)]
        lib.PaitonInkling2Dflash2WorkspaceBytes.restype = C.c_size_t
        lib.PaitonInkling2Dflash2WorkspaceBytes.argtypes = [_P, C.c_int, C.c_int, C.c_int]
        lib.PaitonInkling2Dflash2Context.argtypes = [_P, _P, _P, C.POINTER(_P), C.POINTER(_Cache), C.c_int, _P,
                                                     C.c_size_t, _P]
        lib.PaitonInkling2Dflash2Draft.argtypes = [_P, C.POINTER(_Call), _P]
        _lib = lib
    return _lib


class Inkling2DFlash2Layer(nn.Module):
    def __init__(self, config: Inkling2DFlash2Config, prefix: str) -> None:
        super().__init__()
        hidden, head, mlp = config.hidden_size, config.head_dim, config.intermediate_size
        self.q_size, self.kv_size = config.num_attention_heads * head, config.num_key_value_heads * head
        coefficients = 4 * hidden // 16  # per row: 2 convolutions x 2 taps x (hidden / 16) groups
        self.input_layernorm, self.post_attention_layernorm = _param(hidden), _param(hidden)
        self.attn_proj, self.mlp_proj = _param(coefficients, hidden), _param(coefficients, hidden)
        self.attn_base, self.mlp_base = _param(2, 2, hidden), _param(2, 2, hidden)
        self.qkv = _param(self.q_size + 2 * self.kv_size, hidden)  # rows of q_proj, k_proj, v_proj
        self.o_proj = _param(hidden, self.q_size)
        self.gate_proj, self.up_proj, self.down_proj = _param(mlp, hidden), _param(mlp, hidden), _param(hidden, mlp)
        self.q_norm, self.k_norm = _param(head), _param(head)
        # Owns the paged KV cache only (bf16 whatever the target's format): a window of `sliding_window` tokens.
        self.cache = Inkling2AttentionPassCache(f"{prefix}.attn", config.num_attention_heads,
                                                config.num_key_value_heads, head, config.sliding_window, packed=False)


class Inkling2DFlash2Decoder(nn.Module):
    def __init__(self, config: Inkling2DFlash2Config, max_model_len: int, prefix: str) -> None:
        super().__init__()
        hidden, rank = config.hidden_size, 256
        self.layers = nn.ModuleList(Inkling2DFlash2Layer(config, f"{prefix}.layers.{i}")
                                    for i in range(config.num_hidden_layers))
        self.fc = _param(hidden, len(config.target_layer_ids) * hidden)
        self.hidden_norm, self.norm = _param(hidden), _param(hidden)
        self.hidden_projection = _param(rank, hidden)
        self.pred_codebook, self.succ_codebook = _param(config.vocab_size, rank), _param(config.vocab_size, rank)
        # fp32 [positions][cos 64 | sin 64], read by the library's attention. Built on the host.
        # shortcut: the table has max_model_len rows (512 MB at one million positions), as the DSpark draft's;
        # compute it per call in the kernel before serving contexts far above 100K tokens.
        with torch.device("cpu"):
            table = get_rope(config.head_dim, max_position=max_model_len, dtype=F32,
                             rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta}).cos_sin_cache
        self.cos_sin = table.to(dtype=F32, device=self.norm.device).contiguous()
        self.embed_tokens: nn.Module | None = None  # the target's, if vLLM shares it; attach_target reads the target
        # vLLM cuts every cache bucket into groups of the smallest bucket's layer count and pads the last group of
        # each (cache_plan.py). A bucket of 5 draft layers would pad the 8 full-attention visits to 10: a quarter
        # more cache per context token. Unused caches of the same spec bring this bucket to the count of the full
        # visits, so nothing is padded; they cost their window blocks per request (about 13 MB each).
        layer = self.layers[0].cache
        self.pad_caches = nn.ModuleList(
            Inkling2AttentionPassCache(f"{prefix}.pad.{i}.attn", layer.num_heads, layer.num_kv_heads, layer.head_dim,
                                       layer.sliding_window, packed=False)
            for i in range(FULL_VISITS - len(self.layers)))

    def checkpoint_map(self) -> dict[str, tuple[nn.Parameter, int]]:
        """Tensor name in the draft checkpoint -> (parameter, first row it fills)."""
        m = {"fc.weight": self.fc, "hidden_norm.weight": self.hidden_norm, "norm.weight": self.norm,
             "candidate_selector.hidden_projection.weight": self.hidden_projection,
             "candidate_selector.predecessor_codebook": self.pred_codebook,
             "candidate_selector.successor_codebook": self.succ_codebook}
        out = {name: (param, 0) for name, param in m.items()}
        for i, layer in enumerate(self.layers):
            p, a = f"layers.{i}.", f"layers.{i}.self_attn."
            for name in ("input_layernorm", "post_attention_layernorm"):
                out[f"{p}{name}.weight"] = (getattr(layer, name), 0)
            for kind in ("attn", "mlp"):
                conv = "attention_conv" if kind == "attn" else "mlp_conv"
                out[f"{p}{conv}.kernel_projection.weight"] = (getattr(layer, kind + "_proj"), 0)
                out[f"{p}{conv}.base_kernel"] = (getattr(layer, kind + "_base"), 0)
            for name in ("gate_proj", "up_proj", "down_proj"):
                out[f"{p}mlp.{name}.weight"] = (getattr(layer, name), 0)
            for name in ("q_norm", "k_norm", "o_proj"):
                out[f"{a}{name}.weight"] = (getattr(layer, name), 0)
            out[a + "q_proj.weight"] = (layer.qkv, 0)
            out[a + "k_proj.weight"] = (layer.qkv, layer.q_size)
            out[a + "v_proj.weight"] = (layer.qkv, layer.q_size + layer.kv_size)
        return out


class Inkling2DFlash2Model(nn.Module):
    """DFlash2 draft for Inkling 2 Mini; plugged in by `Inkling2DFlash2Speculator`."""

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
        assert isinstance(config, Inkling2DFlash2Config)
        # The layers are not causal inside a block and were trained on the whole block: always draft all of it.
        # The server may check fewer of the drafts (num_speculative_tokens_per_batch_size).
        if spec.num_speculative_tokens != config.draft_block_size - 1:
            raise ValueError(f"this DFlash2 draft needs num_speculative_tokens {config.draft_block_size - 1} "
                             "(its block size - 1)")
        if vllm_config.model_config.max_model_len > config.max_position_embeddings:
            raise ValueError(f"this DFlash2 draft covers {config.max_position_embeddings} positions; "
                             "set --max-model-len to at most that")
        self.config = config
        self.block = config.draft_block_size
        self.model = Inkling2DFlash2Decoder(config, vllm_config.model_config.max_model_len, f"{prefix}dflash2")
        self.lm_head: nn.Module | None = None  # the target's, if vLLM shares it; attach_target reads the target
        self.handle = _P()
        self.workspace = torch.empty(0, dtype=torch.uint8)  # of the library, grown on demand

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
            raise ValueError(f"the DFlash2 checkpoint lacks {sorted(todo)}")
        return {name for name, _ in self.named_parameters()}

    @torch.no_grad()
    def attach_target(self, target: nn.Module) -> None:
        """After the weights are on the card: hand the library every pointer, with the target's embedding, embed
        norm and LM head rows. `target`: the Inkling 2 language model."""
        from . import model as target_module

        model, config = self.model, self.config
        embed, embed_norm, head = target.model.embed_tokens.weight, target.model.embed_norm, target.lm_head.weight
        vocab = target.text_config.vocab_size  # the candidates come from the rows the target itself samples from
        assert embed.dtype == head.dtype == BF16 and embed.is_contiguous() and head.is_contiguous()
        assert vocab <= head.shape[0] and vocab <= config.vocab_size and config.mask_token_id < embed.shape[0]
        assert embed_norm.eps == config.rms_norm_eps  # the library has one epsilon for every RMS norm
        native = _Model()
        for slot, layer in zip(native.layers, model.layers):
            slot.input_norm, slot.post_norm = layer.input_layernorm.data_ptr(), layer.post_attention_layernorm.data_ptr()
            slot.attn_proj, slot.attn_base = layer.attn_proj.data_ptr(), layer.attn_base.data_ptr()
            slot.mlp_proj, slot.mlp_base = layer.mlp_proj.data_ptr(), layer.mlp_base.data_ptr()
            slot.qkv, slot.o_proj = layer.qkv.data_ptr(), layer.o_proj.data_ptr()
            slot.gate, slot.up, slot.down = layer.gate_proj.data_ptr(), layer.up_proj.data_ptr(), layer.down_proj.data_ptr()
            slot.q_norm, slot.k_norm = layer.q_norm.data_ptr(), layer.k_norm.data_ptr()
        native.fc, native.hidden_norm, native.norm = model.fc.data_ptr(), model.hidden_norm.data_ptr(), model.norm.data_ptr()
        native.hidden_projection = model.hidden_projection.data_ptr()
        native.pred_codebook, native.succ_codebook = model.pred_codebook.data_ptr(), model.succ_codebook.data_ptr()
        self._embed_norm = embed_norm.weight[0].contiguous()
        native.embed, native.embed_norm, native.embed_rows = embed.data_ptr(), self._embed_norm.data_ptr(), embed.shape[0]
        # PAITON_INK2_DRAFT_LOGITS=int8: the candidates from int8 rows of the LM head (the target's own rows when its
        # head is int8, else a copy of 463 MB); anything else: from the bf16 rows.
        self._head_rows = None
        if DRAFT_LOGITS == "int8":
            if target_module.LMHEAD == "int8":
                if target._lm_head_rows is None:  # the target makes them at its first logits call: the same call
                    target._lm_head_rows = native_ops.head_rows(head, "int8")
                self._head_rows = target._lm_head_rows
                target.release_bf16_head()  # only the int8 rows are read from here on: 0.86 GiB back for the cache
            else:
                self._head_rows = native_ops.head_rows(head, "int8")
            native.head_codes, native.head_scale = self._head_rows[0].data_ptr(), self._head_rows[1].data_ptr()
        else:
            native.head_bf16 = head.data_ptr()
        native.cos_sin = model.cos_sin.data_ptr()
        native.vocab, native.window = vocab, config.sliding_window
        native.softcap, native.eps = config.final_logit_softcapping, config.rms_norm_eps
        _ok(_library().PaitonInkling2Dflash2Prepare(C.byref(native), os.environ["PAITON_INK2_OPS_SO"].encode(),
                                                    C.byref(self.handle)), "PaitonInkling2Dflash2Prepare")
        self.__dict__["target"] = target  # not a submodule: its embedding and head rows must outlive the handle
        self.abi = _library().PaitonInkling2Dflash2AbiVersion()

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.cache.prefix for layer in self.model.layers]

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states  # the library mixes the raw target features itself

    def _workspace(self, tokens: int, requests: int, device: torch.device) -> torch.Tensor:
        need = _library().PaitonInkling2Dflash2WorkspaceBytes(self.handle, tokens, requests, self.block)
        if self.workspace.numel() < need:
            self.workspace = torch.empty(need, dtype=torch.uint8, device=device)
        return self.workspace

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        """Keys and values of the context rows (`(T, 4 * hidden)` target features) into every layer's cache."""
        tokens = context_states.shape[0]
        if context_slot_mappings is None or tokens == 0:
            return
        assert context_states.dtype == BF16 and context_states.is_contiguous()
        assert context_positions.dtype == torch.int64 and context_positions.is_contiguous()
        slots, caches = (_P * LAYERS)(), (_Cache * LAYERS)()
        for i, (layer, mapping) in enumerate(zip(self.model.layers, context_slot_mappings, strict=True)):
            if mapping is None or layer.cache.kv_cache.numel() == 0:
                continue  # a null pointer: the library skips the layer
            assert mapping.dtype == torch.int64 and mapping.is_contiguous() and mapping.shape[0] == tokens
            slots[i], caches[i] = mapping.data_ptr(), NativeAttention._cache(layer.cache.kv_cache)
        workspace = self._workspace(tokens, 0, context_states.device)
        _ok(_library().PaitonInkling2Dflash2Context(
            self.handle, context_states.data_ptr(), context_positions.data_ptr(), slots, caches, tokens,
            workspace.data_ptr(), workspace.numel(), _stream(context_states.device)), "PaitonInkling2Dflash2Context")

    def draft(self, input_ids: torch.Tensor, positions: torch.Tensor, requests: int,
              attn_metadata: dict, sample: tuple | None = None) -> torch.Tensor:
        """Draft tokens, int32 `(requests, block - 1)`, of the blocks `input_ids` (per request the bonus token,
        then the mask token) at `positions`. `sample` (vLLM's sample_idx_mapping, temperature, seeds, sample_pos,
        draft_logits and an int32 `(request states, block - 1, 16)` buffer): requests at a temperature above 0 get
        sampled drafts, and every request's 16 scores per row go to its rows of draft_logits."""
        block, tokens = self.block, requests * self.block
        assert input_ids.dtype in (torch.int32, torch.int64) and input_ids.is_contiguous()
        assert positions.dtype == torch.int64 and positions.is_contiguous() and input_ids.shape[0] >= tokens
        out = torch.empty((requests, block - 1), dtype=torch.int32, device=input_ids.device)
        workspace = self._workspace(0, requests, input_ids.device)
        call = _Call(input_ids=input_ids.data_ptr(), positions=positions.data_ptr(), workspace=workspace.data_ptr(),
                     workspace_bytes=workspace.numel(), tokens=out.data_ptr(), requests=requests, block=block,
                     ids_i64=input_ids.dtype == torch.int64)
        if sample is not None:
            state, temperature, seeds, sample_pos, logits, cached = sample
            assert state.dtype == seeds.dtype == sample_pos.dtype == torch.int64 and cached.dtype == torch.int32
            assert temperature.dtype == logits.dtype == torch.float32 and logits.dim() == 3
            assert state.shape[0] >= requests * (block - 1) <= sample_pos.shape[0] and cached.shape[:2] == logits.shape[:2]
            native = _Sample(state.data_ptr(), temperature.data_ptr(), seeds.data_ptr(), sample_pos.data_ptr(),
                             logits.data_ptr(), logits.stride(0), logits.stride(1), cached.data_ptr())
            call.sample = C.addressof(native)  # read during the call
        held, batches = [], {}  # tensors the call points at; one batch per cache group's metadata
        for i, layer in enumerate(self.model.layers):
            md = attn_metadata[layer.cache.prefix]
            if id(md) not in batches:
                table, seq_lens = _int32(md.block_table[:requests]), _int32(md.seq_lens[:requests])
                query_start, slots = _int32(md.query_start_loc[: requests + 1]), md.slot_mapping[:tokens]
                assert slots.dtype == torch.int64 and slots.is_contiguous()
                held += [table, seq_lens, query_start, slots]
                batches[id(md)] = (_Batch(table.data_ptr(), seq_lens.data_ptr(), query_start.data_ptr(), table.shape[1],
                                          requests, tokens, block), slots.data_ptr())
            call.batch[i], call.slots[i] = batches[id(md)]
            call.cache[i] = NativeAttention._cache(layer.cache.kv_cache)
        _ok(_library().PaitonInkling2Dflash2Draft(self.handle, C.byref(call), _stream(input_ids.device)),
            "PaitonInkling2Dflash2Draft")
        return out

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        target = self.__dict__["target"]
        return native_ops.embed_norm(input_ids, target.model.embed_tokens.weight, self._embed_norm,
                                     self.config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("the DFlash2 draft step is one library call: Inkling2DFlash2Model.draft")
