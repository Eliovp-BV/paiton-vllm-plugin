# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini vision tower on Paiton's native vision library (gfx1201).

With `PAITON_INK2_OPS=native` (read in `amd/model.py`) images are encoded by
`inkling2_vision_gfx1201.so` (path in `PAITON_INK2_VISION_SO`): pixel
normalization, patch and position embedding, the 23 layers and the projector
into the decoder width, one library call per image. The library has no
framework dependency; this module only hands it device pointers. No Triton, no
upstream operator module and no PyTorch operator is used here.
"""

import ctypes as C
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn

BLOCKS, WIDTH, FF, CELL, MERGED = 23, 1024, 4096, 40, 4096
MAX_CELLS = 16383  # the library's limit per image (about 5,100 x 5,100 pixels)
# Library field: (checkpoint name, shape). q, k and v are one matrix in the library.
_LAYER: dict[str, tuple[str | None, tuple[int, ...]]] = {
    "ln1_w": ("ln_1.weight", (WIDTH,)),
    "ln1_b": ("ln_1.bias", (WIDTH,)),
    "qkv_w": (None, (3 * WIDTH, WIDTH)),
    "qkv_b": (None, (3 * WIDTH,)),
    "out_w": ("self_attn.out_proj.weight", (WIDTH, WIDTH)),
    "out_b": ("self_attn.out_proj.bias", (WIDTH,)),
    "ls1": ("ls_1.gamma", (WIDTH,)),
    "ln2_w": ("ln_2.weight", (WIDTH,)),
    "ln2_b": ("ln_2.bias", (WIDTH,)),
    "fc_w": ("mlp.c_fc.weight", (FF, WIDTH)),
    "fc_b": ("mlp.c_fc.bias", (FF,)),
    "proj_w": ("mlp.c_proj.weight", (WIDTH, FF)),
    "proj_b": ("mlp.c_proj.bias", (WIDTH,)),
    "ls2": ("ls_2.gamma", (WIDTH,)),
}
_TOP = {
    "patch_w": ("backbone.embeddings.patch_embedding.weight", (WIDTH, 1200)),
    "pos": ("backbone.embeddings.position_embedding", (1024, WIDTH)),
    "cls": ("backbone.embeddings.class_embedding", (WIDTH,)),
    "cls_pos": ("backbone.embeddings.cls_position_embedding", (WIDTH,)),
    "pre_w": ("backbone.ln_pre.weight", (WIDTH,)),
    "pre_b": ("backbone.ln_pre.bias", (WIDTH,)),
    "proj_in": ("proj_in.weight", (MERGED, MERGED)),
    "proj_out": ("proj_out.weight", (0, MERGED)),  # rows = the decoder width
}


class _Layer(C.Structure):  # PaitonInkling2VisionLayer, ABI version 1
    _fields_ = [(n, C.c_void_p) for n in _LAYER]


class _Weights(C.Structure):  # PaitonInkling2VisionWeights, ABI version 1
    _fields_ = [(n, C.c_void_p) for n in _TOP] + [("layers", _Layer * BLOCKS)]


_lib = None


def _library():
    global _lib
    if _lib is None:
        path = os.environ.get("PAITON_INK2_VISION_SO")
        if not path:
            raise ValueError(
                "images with PAITON_INK2_OPS=native need PAITON_INK2_VISION_SO "
                "(inkling2_vision_gfx1201.so); without it serve with "
                '--limit-mm-per-prompt \'{"image": 0}\''
            )
        _lib = C.CDLL(path)
        if _lib.PaitonInkling2VisionGetAbiVersion() != 1:
            raise ValueError(f"{path}: vision library ABI version 1 expected")
        _lib.PaitonInkling2VisionWorkspaceBytes.restype = C.c_size_t
        _lib.PaitonInkling2VisionWorkspaceBytes.argtypes = [C.c_int, C.c_int]
        _lib.PaitonInkling2VisionEncode.restype = C.c_int
        _lib.PaitonInkling2VisionEncode.argtypes = (
            [C.POINTER(_Weights), C.c_void_p, C.c_int, C.c_int]
            + [C.c_void_p] * 3
            + [C.c_size_t, C.c_int, C.c_void_p]
        )
        print(f"INK2 VISION: native vision tower {path}", flush=True)
    return _lib


class Inkling2NativeVision(nn.Module):
    """Raw `[N, 40, 40, 3]` bf16 pixel cells (-1 = padding) to `[N, hidden]`
    decoder embeddings, the interface of `vision.Inkling2VisionEncoder`."""

    def __init__(self, config: Mapping[str, Any], text_hidden_size: int) -> None:
        super().__init__()
        # The library's dimensions are fixed to Inkling 2 Mini.
        fixed = {
            "hidden_size": WIDTH,
            "native_patch_size": CELL // 2,
            "adapter_merge_size": 2,
            "posemb_grid_size": 32,
            "projection_hidden_dim": MERGED,
            "projector_bias": False,
        }
        differ = {k: config[k] for k, v in fixed.items() if k in config and config[k] != v}
        if differ or text_hidden_size != 2304:
            raise ValueError(f"the native vision library is built for Inkling 2 Mini: {differ}")
        _library()
        self.hidden = text_hidden_size
        # Checkpoint name -> (parameter, third of the fused q/k/v matrix or None).
        self._names: dict[str, tuple[str, int | None]] = {}
        for field, (name, shape) in _TOP.items():
            self._raw(field, (shape[0] or text_hidden_size, *shape[1:]))
            self._names[name] = (field, None)
        for i in range(BLOCKS):
            for field, (name, shape) in _LAYER.items():
                self._raw(f"layer{i}_{field}", shape)
                if name is not None:
                    self._names[f"backbone.layers.{i}.{name}"] = (f"layer{i}_{field}", None)
            for part, proj in enumerate("qkv"):
                for kind, field in (("weight", "qkv_w"), ("bias", "qkv_b")):
                    name = f"backbone.layers.{i}.self_attn.{proj}_proj.{kind}"
                    self._names[name] = (f"layer{i}_{field}", part)
        self._w: _Weights | None = None
        self._workspace: torch.Tensor | None = None

    def _raw(self, name: str, shape: tuple[int, ...]) -> None:
        self.register_parameter(
            name, nn.Parameter(torch.empty(shape, dtype=torch.bfloat16), requires_grad=False)
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load `model.visual.*` checkpoint tensors with that prefix stripped."""
        params = dict(self.named_parameters())
        filled: Counter[str] = Counter()
        for name, weight in weights:
            if name not in self._names:
                raise ValueError(f"Unexpected vision checkpoint tensor: {name}")
            target, part = self._names[name]
            data = params[target].data
            if part is not None:
                third = data.shape[0] // 3
                data = data[part * third : (part + 1) * third]
            if weight.dtype != torch.bfloat16 or weight.numel() != data.numel():
                raise ValueError(f"{name}: {weight.dtype} {tuple(weight.shape)} does not fit")
            data.copy_(weight.reshape(data.shape))
            filled[target] += 1
        return {n for n, count in filled.items() if count == (3 if "_qkv_" in n else 1)}

    def _weights(self) -> _Weights:
        # Built at the first call: the tensors are on the card by then.
        if self._w is None:
            ptr = lambda n: getattr(self, n).data_ptr()  # noqa: E731
            w = _Weights(**{n: ptr(n) for n in _TOP})
            for i in range(BLOCKS):
                w.layers[i] = _Layer(**{n: ptr(f"layer{i}_{n}") for n in _LAYER})
            self._w = w
        return self._w

    def forward(self, pixels: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        grid = grid_thw.tolist()
        assert pixels.dtype == torch.bfloat16 and pixels.is_cuda
        assert pixels.shape[1:] == (CELL, CELL, 3)
        assert all(t == 1 for t, _, _ in grid)
        assert sum(h * w for _, h, w in grid) == pixels.shape[0]
        if any(h * w > MAX_CELLS for _, h, w in grid):
            raise ValueError(f"an image has more than {MAX_CELLS} cells of 40 x 40 pixels")
        lib, weights, pixels = _library(), self._weights(), pixels.contiguous()
        # One workspace for all images: the calls of one stream run in order.
        need = max((lib.PaitonInkling2VisionWorkspaceBytes(h, w) for _, h, w in grid), default=0)
        if self._workspace is None or self._workspace.numel() < need:
            self._workspace = torch.empty(need, dtype=torch.uint8, device=pixels.device)
        out = torch.empty((pixels.shape[0], self.hidden), dtype=torch.bfloat16, device=pixels.device)
        stream = torch.cuda.current_stream(pixels.device).cuda_stream
        cell = 0
        for _, rows, cols in grid:
            rc = lib.PaitonInkling2VisionEncode(
                C.byref(weights),
                pixels.data_ptr() + cell * CELL * CELL * 3 * 2,
                rows,
                cols,
                out.data_ptr() + cell * self.hidden * 2,
                None,
                self._workspace.data_ptr(),
                self._workspace.numel(),
                0,
                stream,
            )
            if rc != 0:
                raise RuntimeError(f"PaitonInkling2VisionEncode failed with HIP error {rc}")
            cell += rows * cols
        return out
