"""Host KV tier hit alignment of the offloading-connector overlay (run inside the serving image: needs vllm).

KV4 long-context mode with prefix caching: 2 full-attention groups and 6 GDN (Mamba, align mode) groups with
1,600-token chunks next to the DFlash drafter's sliding-window group with 800-token chunks, which the connector treats
as the draft (EAGLE) group. A hit must end on a 1,600-token block: the recurrent state loaded for each GDN group is the
one at the end of the hit. The drafter group drops its volatile last chunk and used to move the end onto an 800-token
step between two recurrent states (a 258,000-token prompt hit at 256,800 and resumed from the state at 257,600).
"""
import importlib.util
import random
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.v1.kv_offload.base import LookupResult  # noqa: E402

OVERLAY = (Path(__file__).resolve().parents[1] / "paiton_vllm_plugin/_vendor/qwen38/compat/overlays/vllm/distributed/"
           "kv_transfer/kv_connector/v1/offloading/scheduler.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


conn = _load("paiton_overlay_offloading_scheduler", OVERLAY)
SCHED = conn.OffloadingConnectorScheduler
ALIGN = 1600
DRAFT_CHUNK = 800
DRAFT_WINDOW = 3          # 2,048-token sliding window in 800-token chunks
# group_idx: (tokens_per_chunk, sliding_window_size_in_chunks, is_eagle_group)
GROUPS = {0: (1600, None, False), 1: (1600, None, False), **{i: (1600, 1, False) for i in range(2, 8)},
          8: (DRAFT_CHUNK, DRAFT_WINDOW, True)}


class Store:
    def __init__(self, stored):
        self.stored = stored            # set of (group_idx, chunk_idx)

    def lookup(self, key, ctx):
        return LookupResult.HIT if key in self.stored else LookupResult.MISS


def _scheduler(stored, aligned=True):
    configs = tuple(types.SimpleNamespace(group_idx=g, tokens_per_chunk=c, sliding_window_size_in_chunks=w,
                                          is_eagle_group=e) for g, (c, w, e) in GROUPS.items())
    full = [g for g, (_, w, _) in GROUPS.items() if w is None]
    sliding = sorted((g for g, (_, w, _) in GROUPS.items() if w is not None), key=lambda g: GROUPS[g][1], reverse=True)
    s = types.SimpleNamespace(config=types.SimpleNamespace(kv_group_configs=configs), manager=Store(stored),
                              _sliding_window_groups=tuple(sliding), _lookup_groups=tuple(full) + tuple(sliding),
                              _mamba_align_size=ALIGN, _chunks_being_loaded=set(), _paiton_unaligned_hits=0,
                              _events_tracker=types.SimpleNamespace(record_lookup=lambda *a: None))
    for name in ("_lookup_complete_chunks", "_maximal_prefix_lookup", "_sliding_window_lookup",
                 "_paiton_mamba_aligned"):
        setattr(s, name, types.MethodType(getattr(SCHED, name), s))
    if not aligned:                     # the connector before the fix
        s._paiton_mamba_aligned = lambda tokens: tokens
    return s


def _keys(prompt):
    return [[(g, i) for i in range(prompt // c)] for g, (c, _, _) in GROUPS.items()]


def _status(prompt, local=0):
    req = types.SimpleNamespace(num_tokens=prompt, request_id="r")
    return types.SimpleNamespace(num_locally_computed_tokens=local, req=req, req_context=None,
                                 group_states=[types.SimpleNamespace(offload_keys=k) for k in _keys(prompt)])


def _everything_stored(prompt):
    return {key for keys in _keys(prompt) for key in keys}


def _serves(stored, local, hit):
    """Every chunk a hit ending at local + hit needs is stored: the full-attention prefix after the local hit, the
    recurrent state at the end, the drafter's window before the end."""
    end = local + hit
    for g, (c, w, _) in GROUPS.items():
        if w is None:
            need = range(local // c, end // c)
        else:   # the window is clipped where the lookup starts (prompt start or the local hit)
            need = range(max(local // c, end // c - w), end // c)
        if any((g, i) not in stored for i in need):
            return False
    return True


@pytest.mark.parametrize("prompt", [32000, 32001, 32400, 32799, 32800, 33552, 128000, 128400, 256800, 256801,
                                    258000, 262143])
def test_hit_ends_on_a_recurrent_state_block(prompt):
    stored = _everything_stored(prompt)
    hit = _scheduler(stored)._lookup_complete_chunks(_status(prompt))
    assert hit % ALIGN == 0 and hit > 0
    assert _serves(stored, 0, hit)
    # the drafter's last stored chunk is volatile and stays out of the hit
    assert hit + DRAFT_CHUNK <= prompt // DRAFT_CHUNK * DRAFT_CHUNK


def test_258k_hits_where_the_gpu_prefix_cache_hits():
    stored = _everything_stored(258000)
    assert _scheduler(stored)._lookup_complete_chunks(_status(258000)) == 256000
    # before the fix the same lookup ended between two recurrent states; the guard now recomputes instead
    old = _scheduler(stored, aligned=False)
    assert old._lookup_complete_chunks(_status(258000)) == 0
    assert old._paiton_unaligned_hits == 1


def test_aligning_the_end_rechecks_the_drafter_window():
    # 258,000: the drafter verifies chunks up to 256,800 and the end is aligned down to 256,000; its window before
    # 256,000 starts one chunk earlier than the verified run (chunk 317), which is missing here, so the hit has to move
    # further down to an end whose window is stored
    stored = _everything_stored(258000) - {(8, 256000 // DRAFT_CHUNK - DRAFT_WINDOW)}
    hit = _scheduler(stored)._lookup_complete_chunks(_status(258000))
    assert hit % ALIGN == 0 and 0 < hit < 256000
    assert _serves(stored, 0, hit)


def test_debug_switch_exercises_the_hit_end_guard(monkeypatch):
    # PAITON_HOST_KV_DEBUG_UNALIGNED=1 (validation only) skips the alignment; the guard then recomputes the request
    monkeypatch.setattr(conn, "_PAITON_DEBUG_UNALIGNED", True)
    s = _scheduler(_everything_stored(258000))
    assert s._lookup_complete_chunks(_status(258000)) == 0
    assert s._paiton_unaligned_hits == 1
    # an aligned prompt has nothing to recompute
    assert _scheduler(_everything_stored(32000))._lookup_complete_chunks(_status(32000)) % ALIGN == 0


def test_hit_after_a_local_gpu_hit_stays_aligned():
    stored = _everything_stored(200000)
    hit = _scheduler(stored)._lookup_complete_chunks(_status(200000, local=96000))
    assert (96000 + hit) % ALIGN == 0 and hit > 0
    assert _serves(stored, 96000, hit)


def test_no_hit_without_an_aligned_recurrent_state():
    prompt = 64000
    stored = _everything_stored(prompt)
    # keep only GDN states that end off the hit range: every state chunk below 1,600 x 10 dropped except odd ones
    stored = {k for k in stored if not (2 <= k[0] <= 7)} | {(g, 3) for g in range(2, 8)}
    hit = _scheduler(stored)._lookup_complete_chunks(_status(prompt))
    assert hit in (0, 4 * ALIGN)
    if hit:
        assert _serves(stored, 0, hit)


def test_random_sparse_stores_never_give_an_unaligned_or_unserved_hit():
    rng = random.Random(20261004)
    for _ in range(400):
        prompt = rng.randint(1, 262144)
        full = _everything_stored(prompt)
        density = rng.choice((0.3, 0.7, 0.95, 1.0))
        stored = {k for k in full if GROUPS[k[0]][1] is None or rng.random() < density}
        s = _scheduler(stored)
        hit = s._lookup_complete_chunks(_status(prompt))
        assert hit is not None
        assert hit % ALIGN == 0, (prompt, hit)
        if hit:
            assert _serves(stored, 0, hit), (prompt, hit)
        assert s._paiton_unaligned_hits == 0


def test_partial_tail_hits_stay_off_with_a_draft_group():
    # the hit-end guard covers the complete-chunk lookup only; partial-tail hits are disabled while an EAGLE group
    # exists, so they cannot bypass it for this model
    source = OVERLAY.read_text()
    assert "and not any(config.is_eagle_group for config in kv_group_configs)" in source


# Phase B: only draft-group chunks that can serve a hit are stored (sparse Mamba retention, PAITON_HOST_KV_DRAFTER_TRIM)
RET = 32000


def _retained(prompt):
    """Boundaries with a retained recurrent state for one prompt: multiples of the retention interval, the replay
    boundary and the tail checkpoint."""
    tail = prompt // ALIGN * ALIGN - ALIGN
    return {k * RET for k in range(1, prompt // RET + 1)} | {(prompt - 1) // ALIGN * ALIGN} | ({tail} if tail > 0 else set())


def _trim_scheduler():
    s = types.SimpleNamespace(_paiton_retention=RET, _mamba_align_size=ALIGN)
    s._paiton_drafter_chunk_reachable = types.MethodType(SCHED._paiton_drafter_chunk_reachable, s)
    return s


def _req(prompt):
    return types.SimpleNamespace(num_prompt_tokens=prompt, paiton_eagle_ckpt=prompt // ALIGN * ALIGN - ALIGN,
                                 shared_prefix_boundary=None)


@pytest.mark.parametrize("prompt", [3308, 32000, 33552, 64801, 130001, 256801, 258000, 262143])
def test_trim_keeps_exactly_the_reachable_drafter_chunks(prompt):
    s, req = _trim_scheduler(), _req(prompt)
    kept = {i for i in range(prompt // DRAFT_CHUNK) if s._paiton_drafter_chunk_reachable(req, i, DRAFT_CHUNK, DRAFT_WINDOW)}
    want = {i for b in _retained(prompt) for i in range(b // DRAFT_CHUNK - DRAFT_WINDOW, b // DRAFT_CHUNK + 1)
            if 0 <= i < prompt // DRAFT_CHUNK}
    assert kept == want
    if prompt > 200000:
        assert len(kept) < 0.2 * (prompt // DRAFT_CHUNK)       # most drafter chunks are no longer stored


def test_trim_leaves_every_host_hit_unchanged():
    rng = random.Random(4)
    s_trim = _trim_scheduler()
    for _ in range(300):
        prompt = rng.randint(1600, 262144)
        req = _req(prompt)
        keys = _keys(prompt)
        states = {b // ALIGN - 1 for b in _retained(prompt)}
        full = {k for ks in keys for k in ks
                if GROUPS[k[0]][1] is None or (GROUPS[k[0]][1] == 1 and k[1] in states) or k[0] == 8}
        trimmed = {k for k in full
                   if k[0] != 8 or s_trim._paiton_drafter_chunk_reachable(req, k[1], DRAFT_CHUNK, DRAFT_WINDOW)}
        a = _scheduler(full)._lookup_complete_chunks(_status(prompt))
        b = _scheduler(trimmed)._lookup_complete_chunks(_status(prompt))
        assert a == b, (prompt, a, b)
        if b:
            assert _serves(trimmed, 0, b)
