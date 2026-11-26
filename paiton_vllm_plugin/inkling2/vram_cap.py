# SPDX-License-Identifier: Apache-2.0
"""Allocator cap after warm-up, the 27B release's (`_vendor/qwen38/compat/overlays/vllm/v1/worker/gpu_worker.py`).

On the R9700 the KFD admits allocations past the physically free VRAM and then evicts to system memory instead of
failing, and PyTorch's caching allocator only gives cached blocks back after a failed allocation; so a server whose
forwards grow (longer prompts) creeps to the VRAM edge. After warm-up the worker caps the allocator at what it holds
plus what is free, less PAITON_VRAM_HEADROOM_MIB (0 = off). The pure helpers are copied, not imported: the 27B module is
a whole vLLM worker overlay.
"""

import logging
import os

logger = logging.getLogger("vllm.paiton.inkling2.vram_cap")  # under "vllm": printed by vLLM's log handler


def headroom_bytes() -> int:
    return int(os.environ.get("PAITON_VRAM_HEADROOM_MIB", "0") or 0) << 20


def cap_fraction(headroom: int, reserved: int, free: int, total: int) -> float | None:
    """None when no headroom is asked for or less than it is free (a cap below what the allocator holds would turn
    cached blocks into request-time OOMs)."""
    if headroom <= 0 or total <= 0 or free <= headroom:
        return None
    return min(1.0, (reserved + free - headroom) / total)


def launcher_fraction() -> float | None:
    for item in os.environ.get("PYTORCH_ALLOC_CONF", "").split(","):
        key, _, value = item.partition(":")
        if key.strip() == "per_process_memory_fraction":
            try:
                return float(value)
            except ValueError:
                return None
    return None


def lower_only(fraction: float | None, launcher: float | None) -> float | None:
    if fraction is None or launcher is None:
        return fraction
    return min(fraction, launcher)


def physical_free_bytes(domain: int, bus: int, dev: int, root: str = "/sys/bus/pci/devices") -> int | None:
    """Unused VRAM from the card's amdgpu sysfs node; hipMemGetInfo under-reports free memory on the R9700."""
    node = f"{root}/{domain:04x}:{bus:02x}:{dev:02x}.0/mem_info_vram_"
    try:
        with open(node + "total") as total, open(node + "used") as used:
            return int(total.read()) - int(used.read())
    except (OSError, ValueError):
        return None


def apply(device) -> None:
    headroom = headroom_bytes()
    if headroom <= 0:
        return
    import torch

    props = torch.cuda.get_device_properties(device)
    try:
        free = physical_free_bytes(props.pci_domain_id, props.pci_bus_id, props.pci_device_id)
    except AttributeError:
        free = None
    reserved = torch.cuda.memory_reserved(device)
    launcher = launcher_fraction()
    fraction = lower_only(None if free is None else cap_fraction(headroom, reserved, free, props.total_memory),
                          launcher)
    if fraction is not None:
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    logger.info(
        "[paiton.vram_cap] %s",
        {"headroom_mib": headroom >> 20, "reserved_mib": reserved >> 20,
         "physical_free_mib": None if free is None else free >> 20,
         "hip_free_mib": torch.cuda.mem_get_info(device)[0] >> 20, "total_mib": props.total_memory >> 20,
         "launcher_fraction": launcher, "fraction": fraction,
         "cap_mib": None if fraction is None else int(fraction * props.total_memory) >> 20},
    )


def install() -> None:
    """Run `apply` at the end of vLLM's warm-up, in the worker (the 27B overlay calls it at the same place)."""
    from vllm.v1.worker.gpu_worker import Worker

    original = Worker.compile_or_warm_up_model
    if getattr(original, "_paiton_vram_cap", False):
        return

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        apply(self.device)
        return result

    compile_or_warm_up_model._paiton_vram_cap = True
    Worker.compile_or_warm_up_model = compile_or_warm_up_model
