"""Host-memory vision encoder blocks (opt-in prototype, PAITON_HOST_VISION=1).

The Qwen3.8 vision tower (27 blocks of ~30 MB bf16, 0.82 GB of its 0.86 GiB) runs only for image requests, eagerly
and outside the captured graphs. Its block weights move into one pinned host arena (hipHostMalloc, mapped,
coarse-grained); every block's parameters become views into ONE device staging buffer the size of the largest block,
and a forward pre-hook copies that block's weights into it (a stream-ordered, non-blocking H2D copy on the current
stream, a few milliseconds per block over PCIe). The patch embedding, position embedding
and mergers stay resident. Unlike --cpu-offload-params visual, the matrix multiplies never read weights over PCIe.
offload() runs after load_model, next to paiton_host_embed.offload(); it is a no-op unless enabled or without a vision
tower (--language-model-only).
"""
import json
import os
import sys
import time

import torch

import paiton_host_embed

ENABLED = os.environ.get("PAITON_HOST_VISION", "0") == "1"
ALIGN = 256
_KEEP = []


def _log(msg):
    sys.stderr.write(f"[paiton.host_vision] {msg}\n")
    sys.stderr.flush()


def _tensors(block):
    named = list(block.named_parameters(recurse=True)) + list(block.named_buffers(recurse=True))
    return [(n, t) for n, t in named if t.device.type == "cuda" and t.numel() > 0]


def offload(model):
    if not ENABLED:
        return
    visual = getattr(model, "visual", None)
    blocks = getattr(visual, "blocks", None)
    if visual is None or blocks is None or not isinstance(blocks, torch.nn.ModuleList) or len(blocks) == 0:
        _log("no vision blocks (language-model-only?): skipped")
        return
    t0 = time.time()
    layout, offsets, total, largest = [], [], 0, 0
    for block in blocks:
        entries, size = [], 0
        for name, t in _tensors(block):
            if not t.is_contiguous():
                raise RuntimeError(f"host vision: non-contiguous {name}")
            entries.append((name, t, size))
            size += (t.numel() * t.element_size() + ALIGN - 1) // ALIGN * ALIGN
        layout.append(entries)
        offsets.append((total, size))
        total += size
        largest = max(largest, size)
    import ctypes
    buf = (ctypes.c_uint8 * total).from_address(paiton_host_embed._host_alloc(total))
    arena = torch.frombuffer(buf, dtype=torch.uint8)
    if not arena.is_pinned():
        raise RuntimeError("host vision: the hipHostMalloc arena is not recognised as pinned")
    staging = torch.empty(largest, dtype=torch.uint8, device="cuda")
    for (base, size), entries in zip(offsets, layout):
        for _, t, off in entries:
            n = t.numel() * t.element_size()
            arena[base + off:base + off + n].copy_(t.detach().reshape(-1).view(torch.uint8))
    torch.cuda.synchronize()
    for (base, size), entries, block in zip(offsets, layout, blocks):
        for _, t, off in entries:
            n = t.numel() * t.element_size()
            t.data = staging[off:off + n].view(t.dtype).view(t.shape)
        host = arena[base:base + size]

        def pre_hook(module, args, _host=host, _size=size):
            staging[:_size].copy_(_host, non_blocking=True)

        block.register_forward_pre_hook(pre_hook)
    _KEEP.append((buf, arena, staging))
    torch.cuda.empty_cache()
    _log(json.dumps({"blocks": len(blocks), "host_bytes": total, "staging_bytes": largest,
                     "seconds": round(time.time() - t0, 2)}))
