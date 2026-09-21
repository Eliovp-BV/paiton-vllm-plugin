"""Runtime validation for versioned Paiton artifact manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


ARTIFACT_MANIFEST_VERSION = 1
ARTIFACT_BINARY_ABI_VERSION = 1
ARTIFACT_CAPABILITY_VERSION = 1
_VALIDATED_CHECKSUMS: set[tuple] = set()


class ArtifactCompatibilityError(RuntimeError):
    """Raised before loading an incompatible compiled artifact."""


def normalize_gpu_arch(arch: str) -> str:
    match = re.search(r"gfx[0-9a-f]+", arch.lower())
    if match is None:
        raise ArtifactCompatibilityError(f"invalid AMD GPU architecture: {arch!r}")
    return match.group(0)


def detect_runtime_gpu_arch() -> str:
    override = os.environ.get("PAITON_GPU_ARCH")
    if override:
        return normalize_gpu_arch(override)
    try:
        import torch

        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        arch = getattr(properties, "gcnArchName", None)
        if arch:
            return normalize_gpu_arch(arch)
    except Exception as error:
        raise ArtifactCompatibilityError(
            "could not determine the selected AMD GPU architecture; set PAITON_GPU_ARCH"
        ) from error
    raise ArtifactCompatibilityError(
        "selected device did not report an AMD gcnArchName; set PAITON_GPU_ARCH"
    )


def manifest_path_for(artifact_path: str | Path) -> Path:
    return Path(artifact_path).with_suffix(".manifest.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(stat: os.stat_result) -> tuple[int, ...]:
    # Reading may update atime. Only changes to the object or its contents
    # invalidate verification; an access-time update is expected here.
    return (stat.st_dev, stat.st_ino, stat.st_mode, stat.st_ctime_ns,
            stat.st_mtime_ns, stat.st_size)


def load_and_validate_artifact_manifest(
    artifact_path: str | Path,
    *,
    expected_arch: str,
    expected_tp_size: int,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    artifact_path = Path(artifact_path)
    manifest_path = manifest_path_for(artifact_path)
    if not manifest_path.is_file():
        raise ArtifactCompatibilityError(
            f"versioned artifact is missing mandatory manifest: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactCompatibilityError(
            f"could not read artifact manifest {manifest_path}: {error}"
        ) from error

    for key, expected in (
        ("manifest_version", ARTIFACT_MANIFEST_VERSION),
        ("binary_abi_version", ARTIFACT_BINARY_ABI_VERSION),
        ("capability_version", ARTIFACT_CAPABILITY_VERSION),
    ):
        if manifest.get(key) != expected:
            raise ArtifactCompatibilityError(
                f"unsupported {key}={manifest.get(key)!r}; runtime requires {expected}"
            )

    target = manifest.get("target")
    if not isinstance(target, Mapping):
        raise ArtifactCompatibilityError("artifact manifest is missing target metadata")
    selected_arch = normalize_gpu_arch(expected_arch)
    if target.get("arch") != selected_arch:
        raise ArtifactCompatibilityError(
            f"artifact targets {target.get('arch')}, but selected device is {selected_arch}"
        )
    expected_wave = 32 if selected_arch.startswith("gfx12") else 64
    if target.get("wave_size") != expected_wave:
        raise ArtifactCompatibilityError(
            f"artifact wave size {target.get('wave_size')} is invalid for {selected_arch}"
        )

    parallelism = manifest.get("parallelism")
    if not isinstance(parallelism, Mapping) or parallelism.get("tp_size") != expected_tp_size:
        raise ArtifactCompatibilityError(
            f"artifact TP contract does not match requested tp_size={expected_tp_size}"
        )

    artifact = manifest.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ArtifactCompatibilityError("artifact manifest is missing checksum metadata")
    if artifact.get("filename") != artifact_path.name:
        raise ArtifactCompatibilityError("manifest filename does not match shared library")
    if artifact.get("size_bytes") != artifact_path.stat().st_size:
        raise ArtifactCompatibilityError("artifact size does not match manifest")
    stat = artifact_path.stat()
    validation_key = (
        str(artifact_path.resolve()),
        stat.st_dev,
        stat.st_ino,
        stat.st_ctime_ns,
        stat.st_mtime_ns,
        stat.st_size,
        artifact.get("sha256"),
        selected_arch,
        expected_tp_size,
    )
    if verify_checksum and validation_key not in _VALIDATED_CHECKSUMS:
        if artifact.get("sha256") != _sha256_file(artifact_path):
            raise ArtifactCompatibilityError("artifact checksum does not match manifest")
        if _file_identity(artifact_path.stat()) != _file_identity(stat):
            raise ArtifactCompatibilityError("artifact changed during validation")
        _VALIDATED_CHECKSUMS.add(validation_key)
    return manifest


def verify_file(path: Path, declaration: Mapping[str, Any]) -> None:
    """Shared payload verifier; trusted callers supply the expected digest."""
    if path.is_symlink() or not path.is_file():
        raise ArtifactCompatibilityError(f"expected regular execution-package file: {path}")
    before = path.stat()
    digest = declaration.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ArtifactCompatibilityError("invalid expected SHA256")
    if before.st_size != declaration.get("size_bytes"):
        raise ArtifactCompatibilityError(f"artifact size mismatch: {path.name}")
    key = (str(path.resolve()), before.st_dev, before.st_ino, before.st_ctime_ns,
           before.st_mtime_ns, before.st_size, digest)
    if key not in _VALIDATED_CHECKSUMS:
        if _sha256_file(path) != digest:
            raise ArtifactCompatibilityError(f"artifact checksum mismatch: {path.name}")
        if _file_identity(path.stat()) != _file_identity(before):
            raise ArtifactCompatibilityError(f"artifact changed during validation: {path.name}")
        _VALIDATED_CHECKSUMS.add(key)


def validate_execution_package(root: Path, manifest: Mapping[str, Any], *,
                               expected_arch: str, expected_tp_size: int) -> None:
    """Execution schema v1 composes files; binary manifest v1 is unchanged.

    Extension binaries with their own ABI receipts retain those receipts and
    additionally receive this package's exact file/target/parallelism checks.
    """
    if manifest.get("execution_schema") != 1:
        raise ArtifactCompatibilityError("unsupported execution-package schema; expected 1")
    target = manifest.get("target", {})
    arch = normalize_gpu_arch(expected_arch)
    if target != {"arch": arch, "wave_size": 32 if arch.startswith("gfx12") else 64}:
        raise ArtifactCompatibilityError("execution-package GPU/wave mismatch")
    if manifest.get("binary_abi_version") != ARTIFACT_BINARY_ABI_VERSION:
        raise ArtifactCompatibilityError("execution-package ABI mismatch")
    if manifest.get("tp_size") != expected_tp_size:
        raise ArtifactCompatibilityError("execution-package parallelism mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ArtifactCompatibilityError("execution package has no file inventory")
    root = root.resolve(strict=True)
    for name, record in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
            raise ArtifactCompatibilityError("unsafe execution-package path")
        path = root / relative
        if any(p.is_symlink() for p in (path, *path.parents) if p != root and root in p.parents):
            raise ArtifactCompatibilityError("execution package contains a symlink")
        verify_file(path, record)
        if record.get("binary_manifest"):
            load_and_validate_artifact_manifest(path, expected_arch=arch,
                                               expected_tp_size=expected_tp_size)
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(files) | {"execution.json"}:
        raise ArtifactCompatibilityError("execution-package inventory mismatch")
