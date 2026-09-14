"""Bounded, framework-free GGUF metadata inspection; never loads tensor data.

This is an inventory tool, not authorization to execute a GGUF with a Paiton
artifact. Unknown tensor encodings fail closed instead of guessing their size.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct


# GGML type ID: (name, elements per block, bytes per block).
FORMATS = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34),
    12: ("Q4_K", 256, 144), 14: ("Q6_K", 256, 210),
    30: ("BF16", 1, 2),
}
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
           6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class InventoryError(ValueError):
    pass


class _Reader:
    def __init__(self, source, budget):
        self.source = source
        self.budget = budget

    def read(self, count):
        if count < 0 or self.source.tell() + count > self.budget:
            raise InventoryError("GGUF metadata exceeds byte budget")
        data = self.source.read(count)
        if len(data) != count:
            raise InventoryError("truncated GGUF metadata")
        return data

    def scalar(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self):
        try:
            return self.read(self.scalar("Q")).decode("utf-8")
        except UnicodeDecodeError as error:
            raise InventoryError("invalid UTF-8 in GGUF metadata") from error

    def value(self, kind):
        if kind in SCALARS:
            return self.scalar(SCALARS[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            subtype, count = self.scalar("I"), self.scalar("Q")
            if subtype == 9 or subtype not in (*SCALARS, 8):
                raise InventoryError("unsupported GGUF array element type")
            if count > 1_000_000:
                raise InventoryError("GGUF array exceeds element budget")
            return [self.value(subtype) for _ in range(count)]
        raise InventoryError(f"unsupported GGUF metadata type {kind}")


def inspect_gguf(path, *, file_size=None, metadata_budget=64 * 1024 * 1024):
    """Inspect a complete file or a header range with an explicit total size.

    Offsets are absolute byte offsets into the original file. Passing file_size
    is for remote header inspection only: it does not verify the missing bytes.
    Tensor dimensions retain GGUF order (contiguous dimension first).
    """
    path = Path(path)
    local_size = path.stat().st_size
    if file_size is None:
        file_size = local_size
    if type(file_size) is not int or file_size < local_size:
        raise InventoryError("invalid declared file size")
    with path.open("rb") as source:
        r = _Reader(source, metadata_budget)
        if r.read(4) != b"GGUF" or r.scalar("I") not in (2, 3):
            raise InventoryError("expected little-endian GGUF v2/v3")
        nt, nk = r.scalar("Q"), r.scalar("Q")
        if nt > 100_000 or nk > 10_000:
            raise InventoryError("GGUF table exceeds entry budget")
        metadata = {}
        for _ in range(nk):
            key = r.string()
            if key in metadata:
                raise InventoryError(f"duplicate metadata key {key}")
            metadata[key] = r.value(r.scalar("I"))
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or not 1 <= alignment <= 4096 or alignment & (alignment - 1):
            raise InventoryError("invalid GGUF alignment")
        tensors = []
        names = set()
        for _ in range(nt):
            name, rank = r.string(), r.scalar("I")
            if name in names or not name or not 1 <= rank <= 4:
                raise InventoryError("duplicate/empty tensor name or invalid rank")
            names.add(name)
            dims = [r.scalar("Q") for _ in range(rank)]
            kind, offset = r.scalar("I"), r.scalar("Q")
            if kind not in FORMATS:
                raise InventoryError(f"unsupported tensor encoding {kind} for {name}")
            fmt, block, size = FORMATS[kind]
            if not all(0 < d < 2**31 for d in dims) or dims[0] % block:
                raise InventoryError(f"invalid block dimensions for {name}")
            length = math.prod(dims) // block * size
            if offset % alignment:
                raise InventoryError(f"unaligned tensor offset for {name}")
            tensors.append(dict(name=name, dimensions=dims, type_id=kind,
                                format=fmt, offset=offset, size_bytes=length))
        data_offset = (source.tell() + alignment - 1) // alignment * alignment
        if data_offset > file_size:
            raise InventoryError("tensor data offset exceeds file size")
        previous_end = data_offset
        for t in sorted(tensors, key=lambda t: t["offset"]):
            t["offset"] += data_offset
            end = t["offset"] + t["size_bytes"]
            if t["offset"] < previous_end or end > file_size:
                raise InventoryError(f"overlapping or out-of-range tensor {t['name']}")
            previous_end = end
    return dict(metadata=metadata, tensors=tensors, data_offset=data_offset,
                file_size=file_size, header_only=local_size < file_size,
                format_counts=dict(Counter(t["format"] for t in tensors)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--file-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = inspect_gguf(args.path, file_size=args.file_size)
    args.output.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
