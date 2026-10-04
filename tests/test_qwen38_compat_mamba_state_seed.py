"""Align-mode Mamba (GDN) state seeding of the V2 GPU model runner overlay (run inside the serving image: needs vllm and
torch).

KV4 long-context mode with prefix caching: the target attention and GDN groups use 1,600-token blocks, the DFlash2
drafter's sliding-window group 800, and cache_config.block_size is the smallest group block (800). The upstream
MambaHybridModelState.add_request seeded a resumed request's running state column as
(num_computed_tokens - 1) // cache_config.block_size: a 6,400-token prefix-cache hit seeded column 7 instead of 3, the
pre-copy migrated an unrelated block into the running state, and the request continued from it (garbage logits, then
the GDN-norm nonfinite engine error at the first verify step; reproduced with 8K- and 16K-token code prompts, its
identical repeat). The column must count the Mamba group's own block, which the pre-copy kernel also uses.
"""
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

COMPAT = Path(__file__).resolve().parents[1] / "paiton_vllm_plugin/_vendor/qwen38/compat"
REL = "vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
TARGET = "/usr/local/lib/python3.12/dist-packages/" + REL


def _load():
    spec = importlib.util.spec_from_file_location("paiton_overlay_mamba_hybrid", COMPAT / "overlays" / REL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _state(module, block_size, mamba_block_size):
    cls = module.MambaHybridModelState
    state = cls.__new__(cls)
    state.cache_config = types.SimpleNamespace(block_size=block_size, mamba_block_size=mamba_block_size,
                                               mamba_cache_mode="align")
    state._align_mode = True
    state.num_accepted_tokens_gpu = torch.full((4,), 5, dtype=torch.int32)
    state._mamba_state_idx_gpu = torch.zeros(4, dtype=torch.int32)
    return state


@pytest.mark.parametrize("computed, column", [(0, -1), (1, 0), (1600, 0), (1601, 1), (6400, 3), (256000, 159)])
def test_add_request_seeds_the_state_column_in_mamba_blocks(monkeypatch, computed, column):
    module = _load()
    monkeypatch.setattr(module.DefaultModelState, "add_request", lambda self, idx, data: None)
    state = _state(module, block_size=800, mamba_block_size=1600)        # the KV4 long-mode layout
    state.add_request(2, types.SimpleNamespace(num_computed_tokens=computed))
    assert int(state._mamba_state_idx_gpu[2]) == column
    assert int(state.num_accepted_tokens_gpu[2]) == 1


def test_equal_blocks_are_unchanged(monkeypatch):
    module = _load()
    monkeypatch.setattr(module.DefaultModelState, "add_request", lambda self, idx, data: None)
    for mamba_block_size in (880, None):                                   # fp8 long mode; unset -> block_size
        state = _state(module, block_size=880, mamba_block_size=mamba_block_size)
        state.add_request(0, types.SimpleNamespace(num_computed_tokens=31680))
        assert int(state._mamba_state_idx_gpu[0]) == 31679 // 880


def test_manifest_pins_the_overlay():
    entry = json.loads((COMPAT / "compat_manifest.json").read_text())["files"][TARGET]
    overlay = COMPAT / entry["overlay"]
    assert entry["overlay"] == "overlays/" + REL
    assert hashlib.sha256(overlay.read_bytes()).hexdigest() == entry["overlay_sha256"]
    original = Path(TARGET)
    if original.exists():                                                  # inside the serving image
        assert hashlib.sha256(original.read_bytes()).hexdigest() == entry["original_sha256"]
