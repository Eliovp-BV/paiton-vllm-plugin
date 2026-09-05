"""Losslessly split the public Ornith safetensors checkpoint into small shards."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any, BinaryIO


COPY_BYTES = 16 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(COPY_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _read_header(source: Path) -> tuple[int, dict[str, str], list[tuple[str, dict]]]:
    with source.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError("checkpoint is too short for a safetensors header")
        header_length = struct.unpack("<Q", raw_length)[0]
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError("checkpoint has a truncated safetensors header")
        header = json.loads(raw_header)

    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("safetensors metadata must be an object")
    tensors = sorted(header.items(), key=lambda item: item[1]["data_offsets"][0])
    cursor = 0
    for name, spec in tensors:
        start, end = spec["data_offsets"]
        if start != cursor or end < start:
            raise ValueError(f"invalid or non-contiguous tensor range for {name}")
        cursor = end
    if source.stat().st_size != 8 + header_length + cursor:
        raise ValueError("safetensors file size does not match its header")
    return header_length, metadata, tensors


def _partition(
    tensors: list[tuple[str, dict]], limit: int
) -> list[list[tuple[str, dict]]]:
    if limit <= 0:
        raise ValueError("maximum shard size must be positive")
    shards: list[list[tuple[str, dict]]] = []
    current: list[tuple[str, dict]] = []
    current_bytes = 0
    for item in tensors:
        start, end = item[1]["data_offsets"]
        size = end - start
        if current and current_bytes + size > limit:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def _encoded_header(
    metadata: dict[str, str], tensors: list[tuple[str, dict]]
) -> bytes:
    output: dict[str, Any] = {"__metadata__": metadata}
    cursor = 0
    for name, source_spec in tensors:
        start, end = source_spec["data_offsets"]
        size = end - start
        output[name] = {
            "dtype": source_spec["dtype"],
            "shape": source_spec["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    raw = json.dumps(output, separators=(",", ":")).encode("utf-8")
    return raw + b" " * ((8 - len(raw) % 8) % 8)


def _copy_exact(
    source: BinaryIO, target: BinaryIO, count: int, digest: Any
) -> None:
    remaining = count
    while remaining:
        chunk = source.read(min(COPY_BYTES, remaining))
        if not chunk:
            raise EOFError(f"checkpoint ended with {remaining} bytes left to copy")
        target.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


def reshard_checkpoint(
    source: Path,
    output_dir: Path,
    *,
    max_shard_bytes: int = 4_000_000_000,
) -> dict[str, Any]:
    """Copy tensor payloads byte for byte into smaller safetensors shards."""
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_header_bytes, metadata, tensors = _read_header(source)
    shards = _partition(tensors, max_shard_bytes)
    shard_count = len(shards)
    weight_map: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    source_digest = hashlib.sha256()
    logical_payload_digest = hashlib.sha256()

    with source.open("rb") as source_handle:
        for chunk in iter(lambda: source_handle.read(COPY_BYTES), b""):
            source_digest.update(chunk)
        source_handle.seek(0)
        source_handle.read(8 + source_header_bytes)
        payload_base = 8 + source_header_bytes
        for shard_number, shard_tensors in enumerate(shards, start=1):
            filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
            partial = output_dir / f".{filename}.partial"
            final = output_dir / filename
            header = _encoded_header(metadata, shard_tensors)
            source_start = shard_tensors[0][1]["data_offsets"][0]
            source_end = shard_tensors[-1][1]["data_offsets"][1]
            payload_bytes = source_end - source_start
            source_handle.seek(payload_base + source_start)
            shard_digest = hashlib.sha256()
            with partial.open("xb") as target:
                prefix = struct.pack("<Q", len(header)) + header
                target.write(prefix)
                shard_digest.update(prefix)
                _copy_exact(
                    source_handle,
                    target,
                    payload_bytes,
                    logical_payload_digest,
                )
                target.flush()
                os.fsync(target.fileno())
            os.replace(partial, final)
            records.append(
                {
                    "filename": filename,
                    "sha256": sha256_file(final),
                    "file_bytes": final.stat().st_size,
                    "payload_bytes": payload_bytes,
                    "tensor_count": len(shard_tensors),
                }
            )
            for name, _ in shard_tensors:
                weight_map[name] = filename

    total_payload_bytes = sum(
        spec["data_offsets"][1] - spec["data_offsets"][0]
        for _, spec in tensors
    )
    index = {"metadata": {"total_size": total_payload_bytes}, "weight_map": weight_map}
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "source_sha256": source_digest.hexdigest(),
        "source_file_bytes": source.stat().st_size,
        "source_header_bytes": source_header_bytes,
        "logical_payload_sha256": logical_payload_digest.hexdigest(),
        "logical_payload_bytes": total_payload_bytes,
        "tensor_count": len(tensors),
        "max_shard_bytes": max_shard_bytes,
        "shards": records,
    }
    (output_dir / "reshard-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
