"""Pinned, bounded reads of original GGUF tensor bytes, without dequantization."""

import hashlib
import os
from pathlib import Path

from .gguf_inventory import InventoryError, inspect_gguf


class GGUFTensorSource:
    """Read from one verified descriptor; no weight-format or dtype conversion.

    This source is not a vLLM model loader. Callers must additionally validate
    a model-specific tensor/architecture contract before binding any artifact.
    The verified checkpoint must remain immutable throughout its use.
    """

    def __init__(self, path, *, sha256, size_bytes, max_read_bytes=64 * 1024 * 1024):
        if not isinstance(sha256, str) or len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError("expected lowercase SHA256")
        if type(size_bytes) is not int or size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if type(max_read_bytes) is not int or max_read_bytes <= 0:
            raise ValueError("max_read_bytes must be positive")
        self.path = Path(path)
        self.max_read_bytes = max_read_bytes
        self.source = self.path.open("rb")
        try:
            if os.fstat(self.source.fileno()).st_size != size_bytes:
                raise InventoryError("GGUF checkpoint size mismatch")
            digest = hashlib.sha256()
            count = 0
            for block in iter(lambda: self.source.read(min(max_read_bytes, 8 * 1024 * 1024)), b""):
                digest.update(block)
                count += len(block)
            if count != size_bytes or digest.hexdigest() != sha256:
                raise InventoryError("GGUF checkpoint size or SHA256 mismatch")
            # /proc/self/fd pins parsing to the descriptor that was hashed,
            # including when a cache symlink is replaced during verification.
            self.inventory = inspect_gguf(f"/proc/self/fd/{self.source.fileno()}")
            self.tensors = {t["name"]: t for t in self.inventory["tensors"]}
        except BaseException:
            self.source.close()
            raise

    def read_rows(self, name, start, count):
        """Return whole packed rows; GGUF dimension zero is the row width."""
        tensor = self.tensors[name]
        if len(tensor["dimensions"]) != 2:
            raise InventoryError("read_rows requires a matrix tensor")
        rows = tensor["dimensions"][1]
        if type(start) is not int or type(count) is not int or start < 0 or count <= 0 or start + count > rows:
            raise InventoryError("GGUF row range is out of bounds")
        row_bytes = tensor["size_bytes"] // rows
        length = count * row_bytes
        if length > self.max_read_bytes:
            raise InventoryError("GGUF row read exceeds staging budget")
        result = os.pread(self.source.fileno(), length, tensor["offset"] + start * row_bytes)
        if len(result) != length:
            raise InventoryError("GGUF checkpoint truncated after verification")
        return result

    def close(self):
        self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
