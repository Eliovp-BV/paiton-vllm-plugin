# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini text forward as one native call (gfx1201).

Binding of `inkling2_step_gfx1201.so` (path in `PAITON_INK2_STEP_SO`, ABI
version 3): the library runs the 48 layer visits of a looped forward by calling
the operator, expert and attention libraries in the order `loop.py` calls them,
so Python makes one call per forward instead of about 1,200. `PAITON_INK2_STEP=native`
(read in `amd/model.py`) turns it on; it needs `PAITON_INK2_OPS=native`.

`forward` returns None when the forward is not one it covers (no attention
metadata or unbound caches in a profile run, padded rows, the per-visit tap of
the bring-up checks): the caller then runs the Python loop.
"""

import ctypes as C
import os

import torch

from . import native_ops
from .experts_native import _Weights
from .native_attention import KV, NativeAttention, _int32, _Layer

BF16, F32 = torch.bfloat16, torch.float32
# 1: one attention call per decode visit; 2: the operator library's fused launches; 4: in a forward of more than
# 64 rows the loop controller's two products are linear calls (sequencer build 7 on). Same bits with any of them.
FLAGS = int(os.environ.get("PAITON_INK2_STEP_FLAGS", "0"))
TIME = bool(os.environ.get("PAITON_INK2_TIME"))  # bring-up: print the library calls per forward
_P = C.c_void_p


class _StepLayer(C.Structure):  # PaitonInkling2StepLayer
    _fields_ = [(n, _P) for n in (
        "qkvr", "wo", "attn_norm", "mlp_norm", "k_conv", "v_conv", "k_dyn", "v_dyn", "attn_conv", "mlp_conv",
        "attn_query", "mlp_query", "gate_weight", "gate_bias", "ctrl_down", "ctrl_up", "message_query",
        "qkvr_codes", "qkvr_scale", "wo_codes", "wo_scale", "kdyn_codes", "kdyn_scale", "vdyn_codes", "vdyn_scale",
    )] + [("attn", _Layer), ("moe", _Weights), ("route_scale", C.c_float), ("is_local", C.c_int)]


class _Model(C.Structure):  # PaitonInkling2StepModel
    _fields_ = [("layers", _StepLayer * 24)] + [(n, _P) for n in ("embed", "embed_norm", "final_norm", "output_query")] + [
        ("eps", C.c_float)] + [(n, C.c_int) for n in (
            "vocab", "num_layers", "block_size", "core_start", "core_stop", "num_passes", "normalize_values",
            "proj_format")]


class _AttnGroup(C.Structure):  # PaitonInkling2StepAttnGroup
    _fields_ = [(n, _P) for n in ("block_table", "seq_lens", "query_start", "slot_mapping")] + [
        (n, C.c_int) for n in ("block_table_cols", "requests", "max_query_len")]


class _ConvGroup(C.Structure):  # PaitonInkling2StepConvGroup
    _fields_ = [(n, _P) for n in ("block_table", "seq_idx", "slot_mapping", "query_start")] + [("max_blocks", C.c_int)]


class _Visit(C.Structure):  # PaitonInkling2StepVisit
    _fields_ = [("kv", _P)] + [(n, C.c_int64) for n in ("kv_stride_block", "kv_stride_head", "kv_stride_slot")] + [
        ("conv", _P), ("attn_group", C.c_int), ("conv_group", C.c_int), ("kv_format", C.c_int), ("reserved", C.c_int),
        ("kv_v_offset", C.c_int64), ("conv_stride_block", C.c_int64)]


class _Call(C.Structure):  # PaitonInkling2StepCall
    _fields_ = [(n, _P) for n in ("ids", "embeds", "positions", "visits", "attn_groups", "conv_groups", "hidden", "taps",
                                  "tap_visits", "workspace")] + [("workspace_bytes", C.c_size_t)] + [
        (n, C.c_int) for n in ("tokens", "num_taps", "moe_flags", "attn_flags", "flags")]


_lib = None


def _library():
    global _lib
    if _lib is None:
        path = os.environ.get("PAITON_INK2_STEP_SO")
        if not path:
            raise ValueError("PAITON_INK2_STEP=native needs PAITON_INK2_STEP_SO")
        lib = C.CDLL(path)
        if lib.PaitonInkling2StepAbiVersion() != 3:
            raise RuntimeError(f"{path}: sequencer ABI {lib.PaitonInkling2StepAbiVersion()}, need 3")
        sizes = (C.c_int * 6)()
        lib.PaitonInkling2StepSizes(sizes)
        ours = [C.sizeof(t) for t in (_Model, _StepLayer, _Visit, _AttnGroup, _ConvGroup, _Call)]
        if list(sizes) != ours:
            raise RuntimeError(f"{path}: struct sizes {list(sizes)}, the binding has {ours}")
        lib.PaitonInkling2StepPrepare.argtypes = [C.POINTER(_Model), C.c_char_p, C.c_char_p, C.c_char_p, C.POINTER(_P)]
        lib.PaitonInkling2StepVisits.argtypes = [_P]
        lib.PaitonInkling2StepVisitOf.argtypes = [_P, C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int)]
        lib.PaitonInkling2StepVisitOf.restype = None
        lib.PaitonInkling2StepWorkspaceBytes.argtypes = [_P, C.c_int, C.c_int]
        lib.PaitonInkling2StepWorkspaceBytes.restype = C.c_size_t
        lib.PaitonInkling2StepCalls.argtypes = [_P, C.POINTER(C.c_int64)]
        lib.PaitonInkling2StepCalls.restype = None
        lib.PaitonInkling2StepForward.argtypes = [_P, C.POINTER(_Call), _P]
        _lib = lib
    return _lib


class NativeStep:
    """The sequencer of one loaded `Inkling2LoopModel`. Built at the first
    forward with caches: the weights and the attention tables are on the card
    by then."""

    def __init__(self, model) -> None:
        lib = _library()
        ptr = native_ops._ptr
        core = model.loop_core
        m = _Model(eps=model.norm.eps, vocab=model.embed_tokens.weight.shape[0], num_layers=len(model.layers),
                   block_size=model.block_size, core_start=core.start, core_stop=core.stop,
                   num_passes=model.config.num_loop_passes, normalize_values=int(model.normalize_values),
                   proj_format=("bf16", "fp8", "int8").index(native_ops.PROJ))
        m.embed = ptr(model.embed_tokens.weight, BF16, "embedding table")
        m.embed_norm = ptr(model.embed_norm.weight, BF16, "embed norm")
        m.final_norm = ptr(model.norm.weight, BF16, "final norm")
        m.output_query = ptr(model.attn_res.output_query, F32, "output query")
        for i, layer in enumerate(model.layers):
            attn, experts, l = layer.attn, layer.mlp.experts, m.layers[i]
            if experts._w is None:
                experts._w = experts._weights()
            l.attn, l.moe, l.route_scale, l.is_local = attn.native.layer, experts._w, layer.mlp.route_scale, layer.is_local
            weights = dict(
                qkvr=attn.qkvr.weight, wo=attn.wo_ud.weight, attn_norm=layer.attn_norm.weight, mlp_norm=layer.mlp_norm.weight,
                k_conv=attn.k_sconv.conv_weight, v_conv=attn.v_sconv.conv_weight, attn_conv=layer.attn_sconv.conv_weight,
                mlp_conv=layer.mlp_sconv.conv_weight, gate_weight=layer.mlp.gate.weight)
            if not layer.is_local:
                weights.update(k_dyn=attn.k_sconv.weight_proj, v_dyn=attn.v_sconv.weight_proj)
            if i in core:
                weights.update(ctrl_down=layer.controller_down, ctrl_up=layer.controller_gain_forget)
                l.message_query = ptr(model.message_queries[i - core.start], F32, "message queries")
            for name, tensor in weights.items():
                setattr(l, name, ptr(tensor, BF16, f"layer {i} {name}"))
            if m.proj_format:  # PAITON_INK2_PROJ: the 8-bit rows the Python path uses (made at first use, kept there)
                rows = [("qkvr", attn.qkvr.weight), ("wo", attn.wo_ud.weight)]
                if m.proj_format == 2 and not layer.is_local:   # int8: the tap projections too
                    rows += [("kdyn", attn.k_sconv.weight_proj), ("vdyn", attn.v_sconv.weight_proj)]
                for name, weight in rows:
                    codes, scale = native_ops._rows_of(weight)
                    setattr(l, name + "_codes", codes.data_ptr())
                    setattr(l, name + "_scale", scale.data_ptr())
            l.gate_bias = ptr(layer.mlp.gate.bias, F32, "gate bias")
            l.mlp_query = ptr(model.attn_res.mlp_queries[i], F32, "mlp query")
            if i:
                l.attn_query = ptr(model.attn_res.attn_queries[i - 1], F32, "attention queries")
        self.model, self.handle = m, _P()
        rc = lib.PaitonInkling2StepPrepare(C.byref(m), *(os.environ[n].encode() for n in (
            "PAITON_INK2_OPS_SO", "PAITON_INK2_MOE_SO", "PAITON_INK2_ATTN_SO")), C.byref(self.handle))
        if rc != 0:
            raise RuntimeError(f"PaitonInkling2StepPrepare failed with HIP error {rc}")
        self.moe_flags, self.attn_flags = model.layers[0].mlp.experts.flags, model.layers[0].attn.native.flags
        # Per visit: its KV cache and conv-state cache (bound once by vLLM), and which metadata record of a forward
        # it reads. Layers of one cache group share one record, so a forward fills one struct per group.
        count, layer_id, pass_id = lib.PaitonInkling2StepVisits(self.handle), C.c_int(), C.c_int()
        self.visits = (_Visit * count)()
        self.attn_names: list[str] = []   # one layer name per attention group
        self.conv_names: list[str] = []
        self.pairs = []                   # per visit: (attention layer name, conv state name)
        self.packed: dict[str, tuple] = {}  # attention layer name of a packed visit: (cache tensor, bytes per token)
        for v in range(count):
            lib.PaitonInkling2StepVisitOf(self.handle, v, C.byref(layer_id), C.byref(pass_id))
            layer = model.layers[layer_id.value]
            name, kv = layer.attn._pass_cache(pass_id.value)
            conv = layer.conv_states[pass_id.value]
            fmt = layer.attn.native.format   # None, or a packed format of the full visits (PAITON_INK2_KV)
            cache = NativeAttention._cache(kv, fmt)   # checks the tensor's shape and strides
            conv_ptr, conv_values = native_ops.conv_block(conv.kv_cache)   # checks the tensor; padded blocks or not
            self.visits[v] = _Visit(cache.k, cache.stride_block, cache.stride_head, cache.stride_slot, conv_ptr, 0, 0,
                                    {None: 0, 512: 2, 2048: 4}[fmt and fmt[1]], 0, cache.v - cache.k, conv_values * 2)
            if fmt is not None:
                self.packed[name] = (kv, fmt[0])
            self.pairs.append((name, conv.prefix))
        self.grouped = False
        self.workspace = torch.empty(0, dtype=torch.uint8)
        self.calls = (C.c_int64 * 4)()
        self.forwards, self.counted, self.padded = 0, [0, 0, 0, 0], 0
        print(f"INK2 STEP: forward sequencer active, {count} visits, flags {FLAGS}, projections on {native_ops.PROJ} rows, "
              f"full-visit cache {KV}", flush=True)

    def _group(self, attn_metadata: dict) -> None:
        """Visits to metadata records, by identity of the records of the first forward."""
        for names, slot, field in ((self.attn_names, 0, "attn_group"), (self.conv_names, 1, "conv_group")):
            seen: dict[int, int] = {}
            for v, pair in enumerate(self.pairs):
                key = id(attn_metadata[pair[slot]])
                if key not in seen:
                    seen[key] = len(names)
                    names.append(pair[slot])
                setattr(self.visits[v], field, seen[key])
                # A cache group holds visits of one format: its one record is handed over in that format's units.
                assert slot or (pair[0] in self.packed) == (names[seen[key]] in self.packed), pair
        self.records = len({id(md) for md in attn_metadata.values()})
        self.grouped = True

    def forward(self, input_ids, positions, inputs_embeds, attn_metadata: dict, aux_layers: tuple[int, ...]):
        """`aux_layers`: the visits whose input state the drafter reads (ids past the last visit name the
        final output, which is always appended)."""
        tokens = positions.shape[0]
        if not self.grouped:
            self._group(attn_metadata)
        # The grouping is vLLM's cache groups, fixed at start-up: the record count must not change.
        assert len({id(md) for md in attn_metadata.values()}) == self.records
        keep = []
        attn_groups = (_AttnGroup * len(self.attn_names))()
        for g, name in enumerate(self.attn_names):
            md = attn_metadata[name]
            if md.num_actual_tokens != tokens:
                self.padded += 1
                if self.padded == 1:
                    print(f"INK2 STEP: a forward of {tokens} rows with {md.num_actual_tokens} request rows runs on "
                          "the Python loop (padded rows)", flush=True)
                return None
            tensors = (_int32(md.block_table), _int32(md.seq_lens), _int32(md.query_start_loc), md.slot_mapping)
            if name in self.packed:   # library pages and slots, as NativeAttention.forward and .write hand them over
                kv, size = self.packed[name]
                tensors = (NativeAttention._page_table(tensors[0], kv, size), *tensors[1:3],
                           NativeAttention._slots(tensors[3], kv, size))
            assert tensors[3].dtype == torch.int64 and tensors[3].is_contiguous()
            assert tensors[0].shape[0] >= tensors[1].shape[0] < tensors[2].shape[0]
            keep.append(tensors)
            attn_groups[g] = _AttnGroup(*(t.data_ptr() for t in tensors), tensors[0].shape[1], tensors[1].shape[0],
                                        md.max_query_len)
        conv_groups = (_ConvGroup * len(self.conv_names))()
        for g, name in enumerate(self.conv_names):
            md = attn_metadata[name]
            tensors = (native_ops._as(md.block_table, torch.int32), native_ops._as(md.seq_idx, torch.int32),
                       native_ops._as(md.slot_mapping, torch.int64), native_ops._as(md.query_start, torch.int32))
            assert min(t.shape[0] for t in tensors[1:]) >= tokens
            keep.append(tensors)
            conv_groups[g] = _ConvGroup(*(t.data_ptr() for t in tensors), tensors[0].shape[1])
        device = positions.device
        lib = _library()
        need = lib.PaitonInkling2StepWorkspaceBytes(self.handle, tokens, self.moe_flags)
        if self.workspace.numel() < need:
            self.workspace = torch.empty(need, dtype=torch.uint8, device=device)
        hidden = torch.empty((tokens, 2304), dtype=BF16, device=device)
        call = _Call(positions=native_ops._as(positions, torch.int64).data_ptr(), visits=C.addressof(self.visits),
                     attn_groups=C.addressof(attn_groups), conv_groups=C.addressof(conv_groups), hidden=hidden.data_ptr(),
                     workspace=self.workspace.data_ptr(), workspace_bytes=self.workspace.numel(), tokens=tokens,
                     moe_flags=self.moe_flags, attn_flags=self.attn_flags, flags=FLAGS)
        if inputs_embeds is None:
            ids = input_ids.reshape(-1)
            keep.append(ids := native_ops._as(ids, torch.int64))
            call.ids = ids.data_ptr()
        else:
            call.embeds = native_ops._ptr(inputs_embeds, BF16, "input embeddings")
        aux = ()
        taps = sorted(t for t in set(aux_layers) if t < len(self.visits))
        if taps:
            aux = torch.empty((len(taps), tokens, 2304), dtype=BF16, device=device)
            tap_visits = (C.c_int32 * len(taps))(*taps)
            call.taps, call.tap_visits, call.num_taps = aux.data_ptr(), C.addressof(tap_visits), len(taps)
        rc = lib.PaitonInkling2StepForward(self.handle, C.byref(call), torch.cuda.current_stream(device).cuda_stream)
        if rc != 0:
            raise RuntimeError(f"PaitonInkling2StepForward failed with HIP error {rc}")
        self.forwards += 1
        if TIME and self.forwards % 100 == 0:
            counts = self.call_counts()
            mean = [(now - before) / 100 for now, before in zip(counts, self.counted)]
            self.counted = counts
            print(f"INK2 STEP forwards {self.forwards}: library calls per forward: operators {mean[0]:.1f}, experts "
                  f"{mean[1]:.1f}, attention {mean[2]:.1f}, device copies {mean[3]:.1f} (mean of 100)", flush=True)
        return (hidden, [*aux, hidden]) if aux_layers else hidden

    def call_counts(self) -> list[int]:
        """Library calls since the sequencer was built: operators, experts, attention, device copies."""
        _library().PaitonInkling2StepCalls(self.handle, self.calls)
        return list(self.calls)


def forward(model, input_ids, positions, inputs_embeds):
    """The model's output for a text forward, or None when the Python loop
    has to run it. `model` is the `Inkling2LoopModel`."""
    from vllm.forward_context import get_forward_context

    attn_metadata = get_forward_context().attn_metadata
    if (not isinstance(attn_metadata, dict) or getattr(model, "tap", None) is not None or positions.shape[0] == 0
            or not hasattr(model, "loop_core")):
        return None
    step = getattr(model, "_native_step", None)
    if step is None:
        if any(layer.attn._pass_cache(p)[1].numel() == 0 or layer.conv_states[p].kv_cache.numel() == 0
               for layer in model.layers for p in range(len(layer.conv_states))):
            return None  # caches not bound yet (profile run)
        step = model._native_step = NativeStep(model)
    return step.forward(input_ids, positions, inputs_embeds, attn_metadata, tuple(model.aux_hidden_state_layers))
