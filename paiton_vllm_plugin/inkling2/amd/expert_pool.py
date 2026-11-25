# SPDX-License-Identifier: Apache-2.0
"""Expert pool for 16 GB cards (PAITON_INK2_EXPERT_POOL_GIB > 0, set by the launcher's 16 GB profile).

Every layer's 128 experts live in one pinned host allocation (hipHostMalloc, mapped), each expert in the slot layout of
the expert library's ABI 2 (w13, w13_scale, w2, w2_scale back to back). The checkpoint loads straight into it, so the
card never holds all experts at once. The card holds two pools of slots, one per expert format (NVFP4 layers 0-19,
MXFP8 layers 20-23, 20 % of the bytes for MXFP8), seeded from the shipped plan (`expert_plan.json`, best reads per MB
first). Per visit the library's Fetch applies the rule of native/inkling2_pool (ABI 2) to the router's ids, writes the
layer's expert table and copies the misses from the host copy before the expert GEMMs: every routed expert is
computed, so the output stays exact. Structs: the 16 GB plan's FABLE-BRIEF.md 3.2 to 3.4 and k12/ABI.h.

Hosts that cannot pin every copy (32 GB hosts: GTT is half the RAM): PAITON_INK2_POOL_RESIDENT=<NVFP4 layers> keeps
those layers whole on the card (all 128 experts in pinned slots at the front of the NVFP4 pool, loaded straight from
the checkpoint). They never miss, so they need no host copy; the rest of the pool is the same LRU pool over the other
layers. Still exact.

Opt-in, not exact: PAITON_INK2_EXPERT_SKIP=<gate> (expert library ABI 3). In forwards of 8 rows or fewer, an expert
that is not on the card at the visit's start is skipped when every row routes it with less than gate x that row's
top-8 total; those rows' other weights are scaled back to the row's total (16 GB plan NOTES.md, 25 Nov 18:40).
"""
import ctypes as C
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from . import host_embed

GIB = float(os.environ.get("PAITON_INK2_EXPERT_POOL_GIB") or 0)  # GiB of expert slots and ring on the card
ENABLED = GIB > 0
RING_SLOTS = int(os.environ.get("PAITON_INK2_POOL_RING", "16"))  # staged experts of a visit copied to the card; 0: read in place
MXFP8_SHARE = 0.20  # replay of the c10 routes: flat between 0.15 and 0.25
SKIP = float(os.environ.get("PAITON_INK2_EXPERT_SKIP") or 0)  # 0: exact (the default)
RESIDENT = [int(x) for x in os.environ.get("PAITON_INK2_POOL_RESIDENT", "").split(",") if x.strip()]  # set by the launcher
STATS = os.environ.get("PAITON_INK2_POOL_STATS") == "1"  # G3: misses and MB per forward on stderr every 200 forwards
LAYERS, EXPERTS, ROWS = 24, 128, 2304  # w13 has 2 x 1152 rows, w2 2304: both 2304
PARTS = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
ROW_BYTES = {True: (1152, 144, 576, 72), False: (2304, 72, 1152, 36)}  # per part, NVFP4 / MXFP8
EXPERT_BYTES = {fmt: ROWS * sum(rows) for fmt, rows in ROW_BYTES.items()}
OFFSETS = {fmt: np.cumsum((0,) + tuple(ROWS * r for r in rows[:-1])) for fmt, rows in ROW_BYTES.items()}
MAX_SLOTS = 8192
PLAN = Path(__file__).resolve().parents[1] / "expert_plan.json"
assert EXPERT_BYTES == {True: 4_478_976, False: 8_211_456}


class _Group(C.Structure):  # PaitonInkling2PoolGroupState; pointers are device memory
    _fields_ = [("slot_key", C.c_void_p), ("stamp", C.c_void_p), ("pinned", C.c_void_p), ("slots", C.c_int),
                ("slot_bytes", C.c_uint64), ("pool", C.c_void_p)]


class _State(C.Structure):  # PaitonInkling2PoolState (a host struct)
    _fields_ = [("groups", _Group * 2), ("num_groups", C.c_int), ("group_of", C.c_int32 * LAYERS),
                ("key_slot", C.c_void_p), ("seen", C.c_void_p), ("clock", C.c_void_p),
                ("host", C.c_void_p * LAYERS), ("expert_bytes", C.c_uint64 * LAYERS),
                ("skip_gate", C.c_float), ("skip_stats", C.c_void_p)]  # ABI 3 (an ABI 2 library reads 608 bytes)


class _Ring(C.Structure):  # PaitonInkling2StagingRing (a host struct)
    _fields_ = [("slots", C.c_void_p), ("ready", C.c_void_p), ("ring_slots", C.c_int)]


assert (C.sizeof(_Group), C.sizeof(_State), C.sizeof(_Ring)) == (48, 624, 24) and _State.skip_gate.offset == 608
_HOST: dict[int, tuple] = {}  # layer -> (ctypes buffer, [128, expert bytes] pinned tensor, device address)
POOL = None  # the model's pool once built
_POOLS: list = []  # the two groups' slot memory on the card, made at the first need (a resident layer loads into it)


def _slots() -> tuple[list[int], int]:
    """([NVFP4 slots, MXFP8 slots], ring slots): resident layers first in the NVFP4 group, the rest of the budget
    80 / 20 by bytes after the ring."""
    if len(set(RESIDENT)) != len(RESIDENT) or not all(0 <= x < 20 for x in RESIDENT):
        raise ValueError(f"PAITON_INK2_POOL_RESIDENT={RESIDENT}: distinct NVFP4 layers 0-19")
    ring_slots = max(0, RING_SLOTS)
    budget = GIB * 2 ** 30 - ring_slots * EXPERT_BYTES[False] - len(RESIDENT) * EXPERTS * EXPERT_BYTES[True]
    slots = [len(RESIDENT) * EXPERTS + min((20 - len(RESIDENT)) * EXPERTS, max(0, int(budget * (1 - MXFP8_SHARE) // EXPERT_BYTES[True]))),
             min(4 * EXPERTS, max(0, int(budget * MXFP8_SHARE // EXPERT_BYTES[False])))]
    if not all(1 <= s <= MAX_SLOTS for s in slots) or slots[0] == len(RESIDENT) * EXPERTS:
        raise RuntimeError(f"expert pool: {GIB} GiB with {len(RESIDENT)} resident layers gives {slots} slots; "
                           f"each group needs 1 to {MAX_SLOTS} slots besides the resident ones")
    return slots, ring_slots


def _pools() -> list:
    if not _POOLS:
        dev = torch.device("cuda", torch.cuda.current_device())
        _POOLS.extend(torch.empty((s, EXPERT_BYTES[g == 0]), dtype=torch.uint8, device=dev) for g, s in enumerate(_slots()[0]))
    return _POOLS


def host_params(layer: int, nvfp4: bool, shapes: dict[str, tuple[tuple[int, ...], torch.dtype]]) -> dict:
    """The layer's four expert weights as views into one pinned host allocation in slot layout, for the checkpoint to
    load into. `shapes`: part name -> (shape, dtype) of the checkpoint tensor. A resident layer gets its slots on the
    card instead (the same layout), so it loads straight there."""
    if layer in RESIDENT:
        r = RESIDENT.index(layer)
        host = _pools()[0][r * EXPERTS:(r + 1) * EXPERTS]
        _HOST[layer] = (None, host, host.data_ptr())  # also its "host" address: a miss (impossible) would copy it onto itself
        return _views(host, nvfp4, shapes)
    size = EXPERTS * EXPERT_BYTES[nvfp4]
    hip = host_embed._hip()
    ptr, dev = C.c_void_p(), C.c_void_p()
    err = hip.hipHostMalloc(C.byref(ptr), C.c_size_t(size), C.c_uint(host_embed._MAPPED | host_embed._NON_COHERENT))
    if err != 0 or not ptr.value:
        raise RuntimeError(f"expert pool: hipHostMalloc of {size} B for layer {layer} failed (hipError {err})")
    err = hip.hipHostGetDevicePointer(C.byref(dev), ptr, C.c_uint(0))
    if err != 0 or not dev.value:
        raise RuntimeError(f"expert pool: no device address for the host copy of layer {layer} (hipError {err})")
    buf = (C.c_uint8 * size).from_address(ptr.value)
    host = torch.frombuffer(buf, dtype=torch.uint8).view(EXPERTS, EXPERT_BYTES[nvfp4])
    _HOST[layer] = (buf, host, dev.value)
    return _views(host, nvfp4, shapes)


def _views(host: torch.Tensor, nvfp4: bool, shapes: dict) -> dict:
    views = {}
    for name, offset, rows in zip(PARTS, OFFSETS[nvfp4], ROW_BYTES[nvfp4]):
        shape, dtype = shapes[name]
        part = host[:, offset:offset + ROWS * rows].view(dtype).view(shape)  # 1-byte types: a view, no copy
        assert part.data_ptr() == host.data_ptr() + offset
        views[name] = part
    return views


class Pool:
    """Device state of the pool. `state`, `ring` and `miss_logs` are the addresses the sequencer (ABI 4) takes."""

    def __init__(self, nvfp4_of: dict[int, bool]) -> None:
        if not 0 <= SKIP <= 0.12:  # the library refuses a larger gate in Fetch (a row could lose all 8 pairs), failing the forward
            raise ValueError(f"PAITON_INK2_EXPERT_SKIP={SKIP}: use 0 (exact) or a gate up to 0.12")
        if sorted(nvfp4_of) != list(range(LAYERS)) or sorted(_HOST) != list(range(LAYERS)):
            raise RuntimeError("expert pool: every layer's experts must be pooled")
        if any(nvfp4_of[layer] != (layer < 20) for layer in range(LAYERS)):
            raise RuntimeError("expert pool: the two groups assume NVFP4 layers 0-19 and MXFP8 layers 20-23")
        dev = torch.device("cuda", torch.cuda.current_device())
        slots, ring_slots = _slots()
        plan = json.loads(PLAN.read_text())
        order = plan["order"]
        if plan.get("format") != 1 or len({tuple(p) for p in order}) != LAYERS * EXPERTS:
            raise RuntimeError(f"{PLAN}: not a plan of all {LAYERS * EXPERTS} experts (format 1)")
        size = [EXPERT_BYTES[True], EXPERT_BYTES[False]]
        self.pools = _pools()
        slot_key = [np.full(s, -1, np.int32) for s in slots]
        key_slot = np.full(LAYERS * EXPERTS, -1, np.int32)
        # Every table entry starts as the expert's address in the host copy (readable in place), then the seeded ones
        # point at their slot. Table of a layer: w13[128], w13_scale[128], w2[128], w2_scale[128].
        table = np.zeros((LAYERS, 4, EXPERTS), np.uint64)
        for layer in range(LAYERS):
            nv = layer < 20
            table[layer] = (_HOST[layer][2] + np.arange(EXPERTS, dtype=np.uint64)[None] * EXPERT_BYTES[nv]
                            + OFFSETS[nv].astype(np.uint64)[:, None])
        used = [len(RESIDENT) * EXPERTS, 0]
        pinned = np.zeros(slots[0], np.uint8)
        for r, layer in enumerate(RESIDENT):  # loaded in place: slots r x 128 + expert, pinned for good
            s = r * EXPERTS + np.arange(EXPERTS)
            slot_key[0][s] = layer * EXPERTS + np.arange(EXPERTS)
            key_slot[layer * EXPERTS:(layer + 1) * EXPERTS] = s
            pinned[s] = 1
        for layer, expert in order:
            g = int(layer >= 20)
            if layer in RESIDENT:
                continue
            if used[g] == slots[g]:
                continue
            s = used[g]
            used[g] += 1
            self.pools[g][s].copy_(_HOST[layer][1][expert], non_blocking=True)
            slot_key[g][s] = layer * EXPERTS + expert
            key_slot[layer * EXPERTS + expert] = s
            table[layer, :, expert] = self.pools[g][s].data_ptr() + OFFSETS[layer < 20].astype(np.uint64)
        i32 = dict(dtype=torch.int32, device=dev)
        self.slot_key = [torch.from_numpy(k).to(dev) for k in slot_key]
        self.stamp = [torch.zeros(s, **i32) for s in slots]  # uint32 in the library; 0 = never used
        self.pinned = [torch.from_numpy(pinned).to(dev), torch.zeros(slots[1], dtype=torch.uint8, device=dev)]
        self.key_slot = torch.from_numpy(key_slot).to(dev)
        self.seen = torch.zeros(LAYERS * EXPERTS, **i32)
        # shortcut: a 32-bit clock admits nothing after 2^32 visits (about a month of nonstop decoding; still exact),
        # reset stamps and clock between requests if servers run that long.
        self.clock = torch.zeros(1, **i32)
        self.tables = torch.from_numpy(table.view(np.int64)).to(dev)
        self.miss_log = torch.zeros((48, 2 + 2 * EXPERTS), **i32)  # one row per visit of a forward
        self.ring_buf = torch.empty((ring_slots, EXPERT_BYTES[False]), dtype=torch.uint8, device=dev)
        torch.cuda.synchronize()
        st = self._state = _State(num_groups=2)
        for g in range(2):
            st.groups[g] = _Group(self.slot_key[g].data_ptr(), self.stamp[g].data_ptr(), self.pinned[g].data_ptr(),
                                  slots[g], size[g], self.pools[g].data_ptr())
        for layer in range(LAYERS):
            st.group_of[layer] = int(layer >= 20)
            st.host[layer] = _HOST[layer][2]
            st.expert_bytes[layer] = EXPERT_BYTES[layer < 20]
        st.key_slot, st.seen, st.clock = self.key_slot.data_ptr(), self.seen.data_ptr(), self.clock.data_ptr()
        self._ring = _Ring(self.ring_buf.data_ptr() if ring_slots else None, None, ring_slots)
        self.state, self.ring = C.addressof(st), C.addressof(self._ring)
        self.miss_logs = self.miss_log.data_ptr()
        self.skip_stats = torch.zeros(2, dtype=torch.float32, device=dev)  # experts skipped, sum of dropped / row total
        self.skip_sum = torch.zeros(2, dtype=torch.float64, device=dev)  # fp32 counts are exact to 2^24: moved here per forward
        st.skip_gate, st.skip_stats = SKIP, self.skip_stats.data_ptr()
        # [decode (8 rows or fewer), prefill] x [forwards, NVFP4 misses, MXFP8 misses, staged, rows]
        self.counts = torch.zeros((2, 5), dtype=torch.int64, device=dev)
        self.forwards = 0
        self.slots = slots
        sys.stderr.write(f"[paiton.inkling2] expert pool: {slots[0]} NVFP4 + {slots[1]} MXFP8 slots "
                         f"({sum(s * b for s, b in zip(slots, size)) / 2 ** 30:.2f} GiB), ring {ring_slots}, "
                         f"seeded {used[0]} + {used[1]}, resident layers {RESIDENT or 'none'}; host copies "
                         f"{sum(h[1].numel() for h in _HOST.values() if h[0] is not None) / 2 ** 30:.2f} GiB\n")

    def account(self, tokens: int, layer: int | None = None) -> None:
        """Add the miss log to the counters: one visit (`layer`, the Python expert path, row 0) or a whole forward
        of the sequencer (rows 0-47; visits 44-47 run the MXFP8 layers 20-23)."""
        c, log = self.counts[int(tokens > 8)], self.miss_log
        if layer is None:
            c[1] += log[:44, 0].sum()
            c[2] += log[44:, 0].sum()
            c[3] += log[:, 1].sum()
        else:
            c[1 + int(layer >= 20)] += log[0, 0]
            c[3] += log[0, 1]
        if layer not in (None, 0):
            return
        c[0] += 1
        c[4] += tokens
        if SKIP:
            self.skip_sum += self.skip_stats
            self.skip_stats.zero_()
        self.forwards += 1
        if self.forwards % 200 == 0:
            for kind, (f, nv, mx, staged, rows) in zip(("decode", "prefill"), self.counts.tolist()):
                skip = ""
                if SKIP and kind == "decode":  # the skip runs in decode forwards only; 48 visits per row
                    n, share = self.skip_sum.tolist()
                    skip = f" skipped {n:.0f} dropped weight {100 * share / max(48 * rows, 1):.3f} %"
                sys.stderr.write(f"[paiton.inkling2] expert pool stats {kind}: forwards {f} misses {nv + mx} "
                                 f"({(nv * EXPERT_BYTES[True] + mx * EXPERT_BYTES[False]) / 1e6:.0f} MB) staged {staged}{skip}\n")

    def table(self, layer: int) -> int:
        """Device address of the layer's expert table."""
        return self.tables[layer].data_ptr()


def build(experts) -> Pool:
    """After every layer loaded (the model's process_weights_after_loading). `experts`: the pooled expert modules."""
    global POOL
    POOL = Pool({m.layer_id: m.nvfp4 for m in experts})
    return POOL
