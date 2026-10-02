"""VRAM headroom cap in the vendored vLLM worker overlay (run inside the serving image: needs vllm and torch).

On the R9700 the KFD admits allocations up to tens of MiB past the physically free VRAM and then evicts the process's
own buffers into system memory instead of failing (measured with a native hipMalloc probe: the last 64 MiB step with
45 MiB free succeeded and evicted queues). PyTorch's caching allocator only gives memory back after a failed
allocation, so a long-running server settles at that edge; the 4-bit long-context mode ran the 15.5 GiB host out of
memory twice that way. After warm-up the worker caps the allocator at what it holds plus what is free, less a
headroom, so PyTorch frees cached blocks before the driver has to evict.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

OVERLAY = (Path(__file__).resolve().parents[1]
           / "paiton_vllm_plugin/_vendor/qwen38/compat/overlays/vllm/v1/worker/gpu_worker.py")
GIB, MIB = 1 << 30, 1 << 20


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "vllm.v1.worker"   # the overlay's relative imports resolve against the installed package
    sys.modules[name] = module   # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


worker = _load("paiton_overlay_gpu_worker", OVERLAY)


def test_cap_lets_the_allocator_grow_into_free_memory_less_the_headroom():
    fraction = worker._paiton_vram_cap_fraction(512 * MIB, reserved=30 * GIB, free=1536 * MIB, total=32 * GIB)
    assert round(fraction * 32 * GIB) == 30 * GIB + 1024 * MIB


def test_no_cap_when_the_headroom_is_not_set():
    assert worker._paiton_vram_cap_fraction(0, reserved=30 * GIB, free=GIB, total=32 * GIB) is None


def test_no_cap_when_less_than_the_headroom_is_free():
    # capping below what the allocator already holds would turn cached blocks into request-time OOMs
    assert worker._paiton_vram_cap_fraction(512 * MIB, reserved=31 * GIB, free=256 * MIB, total=32 * GIB) is None


def _fake_card(root, total, used, address="0000:03:00.0"):
    node = root / address
    node.mkdir(parents=True)
    (node / "mem_info_vram_total").write_text(f"{total}\n")
    (node / "mem_info_vram_used").write_text(f"{used}\n")


def test_physical_free_comes_from_the_devices_own_amdgpu_sysfs_node(tmp_path):
    # hipMemGetInfo reported 327 MiB free on the R9700 while the card had ~1.6 GiB unused (2 Oct, rc-t3 smoke)
    _fake_card(tmp_path, total=32 * GIB, used=30 * GIB + 512 * MIB)
    _fake_card(tmp_path, total=2 * GIB, used=GIB, address="0000:0c:00.0")   # a second (display) card
    assert worker._paiton_physical_free_bytes(0, 3, 0, root=tmp_path) == GIB + 512 * MIB


def test_physical_free_is_unknown_without_the_sysfs_node(tmp_path):
    assert worker._paiton_physical_free_bytes(0, 3, 0, root=tmp_path) is None


def test_launcher_fraction_is_read_from_the_allocator_setting(monkeypatch):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:64,per_process_memory_fraction:0.95")
    assert worker._paiton_launcher_fraction() == 0.95
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:64")
    assert worker._paiton_launcher_fraction() is None
    monkeypatch.delenv("PYTORCH_ALLOC_CONF")
    assert worker._paiton_launcher_fraction() is None


def test_warm_up_cap_only_lowers_the_launcher_cap():
    # rc-t4: the warm-up cap (0.9565) raised the launcher's 0.95 and the card peaked 0.17 GiB below the KFD limit
    assert worker._paiton_lower_only(0.9565, 0.95) == 0.95
    assert worker._paiton_lower_only(0.941, 0.95) == 0.941
    assert worker._paiton_lower_only(0.941, None) == 0.941
    assert worker._paiton_lower_only(None, 0.95) is None


def test_headroom_comes_from_the_launcher_environment(monkeypatch):
    monkeypatch.delenv("PAITON_VRAM_HEADROOM_MIB", raising=False)
    assert worker._paiton_vram_headroom_bytes() == 0
    monkeypatch.setenv("PAITON_VRAM_HEADROOM_MIB", "512")
    assert worker._paiton_vram_headroom_bytes() == 512 * MIB
