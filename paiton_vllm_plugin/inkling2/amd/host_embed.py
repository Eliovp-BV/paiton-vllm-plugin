"""Input embedding in pinned host memory (opt-in, PAITON_INK2_HOST_EMBED=1; for 16 GB cards).

The target's bf16 embedding (padded vocabulary x 2,304, 0.86 GiB) moves into one hipHostMalloc allocation (mapped,
coarse-grained: GTT memory, which the kernel does not evict the way it evicts registered user pointers) and the
module reads it through a zero-copy device view. A step reads only the rows of its tokens (4,608 B each) over PCIe:
the same values, so the same tokens. The drafter shares the module object and follows. The pattern of the 27B line's
`_vendor/qwen38/paiton_host_embed.py`, which is not in the Inkling image. offload() runs in the target's
process_weights_after_loading: before the drafter loads and before the sequencer takes the table's address at the
first forward. The buffer is kept for the life of the process.
"""
import ctypes
import os
import sys

import torch

ENABLED = os.environ.get("PAITON_INK2_HOST_EMBED", "0") == "1"
_MAPPED, _NON_COHERENT = 0x2, 0x80000000
_KEEP = []


def _hip():
    """The HIP runtime this process already uses: the image also carries an old system libamdhip64.so.5."""
    with open("/proc/self/maps") as maps:
        paths = {line.split()[-1] for line in maps if "/libamdhip64.so" in line}
    if len(paths) != 1:
        raise RuntimeError(f"host embedding: expected one loaded libamdhip64, found {sorted(paths)}")
    return ctypes.CDLL(paths.pop())


def offload(layer: torch.nn.Module) -> None:
    """Swap `layer.weight` for a zero-copy view of a pinned host copy (no-op unless enabled)."""
    if not ENABLED:
        return
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    weight = layer.weight
    if weight.device.type != "cuda" or weight.dim() != 2 or not weight.is_contiguous():
        raise RuntimeError(f"host embedding: unexpected weight {tuple(weight.shape)} on {weight.device}")
    nbytes = weight.numel() * weight.element_size()
    ptr = ctypes.c_void_p()
    err = _hip().hipHostMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(_MAPPED | _NON_COHERENT))
    if err != 0 or not ptr.value:
        raise RuntimeError(f"host embedding: hipHostMalloc of {nbytes} B failed (hipError {err})")
    buf = (ctypes.c_uint8 * nbytes).from_address(ptr.value)
    host = torch.frombuffer(buf, dtype=torch.uint8).view(weight.dtype).view(weight.shape)
    if not host.is_pinned():
        raise RuntimeError("host embedding: the hipHostMalloc buffer is not recognised as pinned")
    host.copy_(weight)  # device to pinned host
    torch.cuda.synchronize()
    view = get_accelerator_view_from_cpu_tensor(host)
    if view.device.type != "cuda" or view.data_ptr() == weight.data_ptr() or not torch.equal(view, weight):
        raise RuntimeError("host embedding: no zero-copy device view equal to the card's table")
    param = torch.nn.Parameter(view, requires_grad=False)
    for key, value in vars(weight).items():  # weight_loader and the other loader attributes
        setattr(param, key, value)
    _KEEP.append((buf, host, view))
    layer.weight = param
    del weight
    torch.cuda.empty_cache()
    sys.stderr.write(f"[paiton.inkling2] input embedding in pinned host memory: {nbytes} B, {tuple(view.shape)}\n")
