# SPDX-License-Identifier: Apache-2.0
"""3-bit experts for the fast profile (W3X; opt-in, not exact). PAITON_INK2_W3X (the launcher's --w3x) names a folder
from the W3X export: manifest.json and one layer_LL.safetensors per 3-bit layer. Per layer the manifest says "w3"
(that file's codes, scales and factor) or "checkpoint" (the checkpoint's own NVFP4 / MXFP8 experts, today's path).

File layout (notes/W3X-LAYOUT.md): weight = (u - 4) x e4m3(scale) x factor, u in 0..7 as 3 bits LSB-first per row
(32 codes in 3 dwords), one e4m3 scale per 64 inputs, an fp32 factor per expert matrix. The expert library (ABI 4,
TUNE's W3X-ENTRY.md) repacks the codes into its card layout on the host (PaitonInkling2MoeW3Repack); scales and
factors load as stored.
No torch here: the launcher reads the manifest before vLLM starts."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import re
from functools import cache
from pathlib import Path

GROUP, EXPERTS, LAYERS = 64, 128, 24
SHAPES = {"w13": (2304, 2304), "w2": (2304, 1152)}  # (rows n, inputs k) of one expert matrix, checkpoint row order
MXFP8_LAYERS = range(20, 24)  # the checkpoint's MXFP8 experts; layers 0-19 are NVFP4
CHECKPOINT_EXPERT_BYTES = {False: 4_478_976, True: 8_211_456}  # one expert, both matrices: NVFP4, MXFP8
_PARTS = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.(?:w13|w2)_weight")


def codes_bytes(n: int, k: int) -> int:
    return n * k * 3 // 8


def expert_bytes() -> int:
    """One 3-bit expert on the card: codes, scales and the two matrix factors (3,110,408 B)."""
    return sum(codes_bytes(n, k) + n * k // GROUP + 4 for n, k in SHAPES.values())


def layers(folder: str | None = None) -> dict[int, Path]:
    """{layer: file} of the 3-bit layers in the manifest of `folder` (default PAITON_INK2_W3X); {} without one."""
    return {layer: path for layer, (path, _) in _entries(folder).items()}


def denominator(layer: int, m: str) -> float:
    """The struct's denominator of matrix `m` of a 3-bit layer: 1.0 (the factor is the whole outer scale) unless the
    manifest entry gives "denominator": {"w13": .., "w2": ..} (a bit gate against a vehicle checkpoint: factor = its
    amax, denominator = its outer-scale denominator, as the NVFP4 kernels take them)."""
    return _entries()[layer][1].get(m, 1.0)


def _entries(folder: str | None = None) -> dict[int, tuple[Path, dict]]:
    folder = folder or os.environ.get("PAITON_INK2_W3X")
    return _layers(folder) if folder else {}


@cache
def _layers(folder: str) -> dict[int, tuple[Path, dict]]:
    root = Path(folder)
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"PAITON_INK2_W3X={folder}: no readable manifest.json ({error})") from None
    head = {key: manifest.get(key) for key in ("format", "version", "group", "shapes")}
    if head != dict(format="W3X", version=1, group=GROUP, shapes={m: list(s) for m, s in SHAPES.items()}):
        raise ValueError(f"{root}/manifest.json: not a W3X v1 file of this model: {head}")
    found: dict[int, tuple[Path, dict]] = {}
    seen = set()
    for entry in manifest.get("layers", []):
        layer, kind = entry.get("layer"), entry.get("format")
        if not isinstance(layer, int) or not 0 <= layer < LAYERS or layer in seen or kind not in ("w3", "checkpoint"):
            raise ValueError(f"{root}/manifest.json: bad layer entry {entry}")
        seen.add(layer)
        if kind == "checkpoint":
            continue
        # shortcut: per-row factors (stride 1) are the reserved formats 4 / 5 (lead, 26 Nov 10:48); load them once built
        stride, name = entry.get("factor_stride"), entry.get("file", "")
        if stride != 0 or isinstance(stride, bool):
            raise ValueError(f"{root}: layer {layer} has factor_stride {stride!r}; the expert library takes 0 (per matrix)")
        if Path(name).name != name or not (root / name).is_file():
            raise ValueError(f"{root}: layer {layer} names file {name!r}, not a file in that folder "
                             "(or a link to a place that is not mounted)")
        den = entry.get("denominator", {})
        if not isinstance(den, dict) or not set(den) <= set(SHAPES) or not all(
                type(v) in (int, float) and 0 < v < math.inf for v in den.values()):
            raise ValueError(f"{root}: layer {layer} has denominator {den!r}, not {{matrix: number above 0}}")
        found[layer] = root / name, {m: float(v) for m, v in den.items()}
    return found


def experts_gib(folder: str | None = None, w3=None) -> float:
    """GiB of routed experts on the card with the 3-bit layers of `folder`, or the layer numbers `w3` (14.59 exact,
    8.90 all 3-bit)."""
    w3 = layers(folder) if w3 is None else w3
    return EXPERTS * sum(expert_bytes() if layer in w3 else CHECKPOINT_EXPERT_BYTES[layer in MXFP8_LAYERS]
                         for layer in range(LAYERS)) / 2 ** 30


def replaces(name: str) -> bool:
    """True for a checkpoint expert tensor (renamed) of a 3-bit layer: the W3X file replaces it."""
    match = _PARTS.search(name)
    return match is not None and int(match.group(1)) in layers()


def read(layer: int, repack, card_bytes) -> dict:
    """Host tensors of one 3-bit layer: {"w13" | "w2": (card codes [128, card_bytes(n, k)] u8, scales [128, n, k/64]
    u8, factor [128] f32)}. `repack(codes_ptr, n, k, out_ptr)` is the library's PaitonInkling2MoeW3Repack:
    thread-safe, one expert matrix per call (about 5 ns per weight on one core), so the experts run on several threads."""
    import torch
    from safetensors import safe_open

    path, out = layers()[layer], {}
    workers = min(16, len(os.sched_getaffinity(0)))
    with safe_open(str(path), framework="pt") as f:
        for m, (n, k) in SHAPES.items():
            want = {f"{m}.codes": ([EXPERTS, n, codes_bytes(1, k)], "U8"), f"{m}.scales": ([EXPERTS, n, k // GROUP], "U8"),
                    f"{m}.factor": ([EXPERTS], "F32")}
            for key, (shape, dtype) in want.items():
                part = f.get_slice(key) if key in f.keys() else None
                if part is None or list(part.get_shape()) != shape or part.get_dtype() != dtype:
                    raise ValueError(f"{path}: {key} should be {dtype} {shape}")
            codes = f.get_tensor(f"{m}.codes").contiguous()
            card = torch.empty((EXPERTS, card_bytes(n, k)), dtype=torch.uint8)
            with ThreadPoolExecutor(workers) as pool:
                refused = [e for e, rc in enumerate(pool.map(
                    lambda e: repack(codes[e].data_ptr(), n, k, card[e].data_ptr()), range(EXPERTS))) if rc != 0]
            if refused:
                raise RuntimeError(f"{path}: the expert library refused the {m} codes of experts {refused} ({n} x {k})")
            out[m] = card, f.get_tensor(f"{m}.scales").contiguous(), f.get_tensor(f"{m}.factor").contiguous()
    return out
