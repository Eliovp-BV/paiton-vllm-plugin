"""Host-memory input embedding (opt-in prototype, PAITON_HOST_EMBED=1).

Moves the target's bf16 input embedding (248,320 x 5,120 = 2.54 GB) into pinned host memory and serves the lookups
through a zero-copy GPU view, so its VRAM can go to the KV pool. A decode step reads only the rows of its tokens
(10,240 B each) over PCIe; the table never moves again.

The table lives in one hipHostMalloc allocation (mapped, coarse-grained: GPU reads may be cached; GTT memory, which
the kernel does not evict the way it evicts registered user pointers) rather than torch's caching host allocator,
which rounds sizes up to the next power of two (4 GiB here). The lookup stays a plain F.embedding read of a CUDA
tensor, so torch.compile and the captured graphs keep working, and the DFlash2 drafter, which shares the module
object, follows. offload() must run after load_model and before the drafter is loaded and graphs are captured: the
weight address is baked into the graphs. The buffer is kept for the life of the process.
"""
import ctypes
import json
import os
import sys
import time

import torch

ENABLED = os.environ.get("PAITON_HOST_EMBED", "0") == "1"
HIP_HOST_MALLOC_MAPPED = 0x2
HIP_HOST_MALLOC_NON_COHERENT = 0x80000000
_KEEP = []


def _log(msg):
    sys.stderr.write(f"[paiton.host_embed] {msg}\n")
    sys.stderr.flush()


def _host_alloc(nbytes):
    hip = None
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        try:
            hip = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if hip is None:
        raise RuntimeError("host embedding: libamdhip64 not found")
    ptr = ctypes.c_void_p()
    err = hip.hipHostMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes),
                            ctypes.c_uint(HIP_HOST_MALLOC_MAPPED | HIP_HOST_MALLOC_NON_COHERENT))
    if err != 0 or not ptr.value:
        raise RuntimeError(f"host embedding: hipHostMalloc of {nbytes} B failed (hipError {err})")
    return ptr.value


def _embedding(model):
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
    found = [(n, m) for n, m in model.named_modules()
             if isinstance(m, VocabParallelEmbedding) and not isinstance(m, ParallelLMHead)
             and n.endswith("embed_tokens")]
    if len(found) != 1:
        raise RuntimeError(f"host embedding: expected one input embedding, found {[n for n, _ in found]}")
    return found[0]


def offload(model):
    """Replace the input embedding's weight by a zero-copy view of a pinned host copy (no-op unless enabled)."""
    if not ENABLED:
        return
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    t0 = time.time()
    name, layer = _embedding(model)
    weight = layer.weight
    if weight.device.type != "cuda" or weight.dim() != 2 or not weight.is_contiguous():
        raise RuntimeError(f"host embedding: unexpected weight {tuple(weight.shape)} on {weight.device}")
    nbytes = weight.numel() * weight.element_size()
    buf = (ctypes.c_uint8 * nbytes).from_address(_host_alloc(nbytes))
    host = torch.frombuffer(buf, dtype=torch.uint8).view(weight.dtype).view(weight.shape)
    if not host.is_pinned():
        raise RuntimeError("host embedding: the hipHostMalloc buffer is not recognised as pinned")
    host.copy_(weight)          # device -> pinned host (pageable device-to-host copies are unsafe on gfx1201)
    torch.cuda.synchronize()
    view = get_accelerator_view_from_cpu_tensor(host)
    if view.device.type != "cuda" or view.data_ptr() == weight.data_ptr():
        raise RuntimeError("host embedding: no zero-copy device view")
    if not torch.equal(view, weight):
        raise RuntimeError("host embedding: the zero-copy view differs from the GPU table")
    param = torch.nn.Parameter(view, requires_grad=False)
    for key, value in vars(weight).items():     # weight_loader and other loader attributes
        setattr(param, key, value)
    _KEEP.append((buf, host, view))
    layer.weight = param
    del weight
    torch.cuda.empty_cache()
    _log(json.dumps({"module": name, "bytes": nbytes, "shape": list(view.shape), "dtype": str(view.dtype),
                     "host_ptr": hex(host.data_ptr()), "device_ptr": hex(view.data_ptr()),
                     "seconds": round(time.time() - t0, 2), "check": "equal"}))
