#!/usr/bin/env python3
"""Verify a Paiton MiniMax H3 Ref2VA release directory before loading the artifact.

Metadata mode is CPU-only and does not import Torch, ComfyUI or the runtime.
It is the CI gate; ``--checkpoints`` adds the pinned weight hashes for the release
host. Exit codes: 0 pass, 2 mismatch, 3 missing file, 4 usage error.

Draft for v1.1.0. The release lock is the single source of truth: no artifact
hash is hard-coded in this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
LOCK_NAME = "release.lock.json"
MAX_TOKENS = 49152
PACKED_LAYOUT = ["text", "refs", "audio", "video"]
REF_LIMITS = {"images": 9, "videos": 3, "audios": 3}
ALLOWED_SCOPES = {
    "fl2va_pruned_int8_accumulation_projections",
    "ref2va_pruned_int8_accumulation_projections",
}
REQUIRED_LICENSES = (
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "MODIFICATIONS.md",
    "LICENSES/MiniMax-H3-Community.txt",
    "LICENSES/comfy-kitchen-Apache-2.0.txt",
    "LICENSES/ComfyUI-GPL-3.0.txt",
    "LICENSES/Triton-MIT.txt",
)
NOTICE_SENTENCE = (
    "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, "
    "Copyright \u00a9 2026 MiniMax. All Rights Reserved."
)
FORBIDDEN_SNIPPETS = (
    "/home/",
    "paiton-compiler",
    "compile_lib",
    "native/minimax_h3",
    "keys/",
    "notes/",
)
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class VerificationError(RuntimeError):
    """A release invariant failed."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise VerificationError(3, f"missing file: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(2, f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(2, f"expected a JSON object in {path}")
    return value


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise VerificationError(2, f"{where}: missing required field {key!r}")
    return mapping[key]


def _expect(value: Any, expected: Any, where: str) -> None:
    if value != expected:
        raise VerificationError(2, f"{where}: expected {expected!r}, found {value!r}")


def check_manifest(manifest: dict[str, Any], artifact_name: str) -> None:
    for key, expected in (
        ("manifest_version", 1),
        ("binary_abi_version", 1),
        ("capability_version", 1),
        ("h3_projection_abi_version", 1),
        ("h3_contract_version", 1),
        ("h3_max_tokens", MAX_TOKENS),
    ):
        _expect(manifest.get(key), expected, f"manifest.{key}")

    target = _require(manifest, "target", "manifest")
    _expect(target.get("arch"), "gfx1201", "manifest.target.arch")
    _expect(target.get("wave_size"), 32, "manifest.target.wave_size")
    parallelism = _require(manifest, "parallelism", "manifest")
    _expect(parallelism.get("tp_size"), 1, "manifest.parallelism.tp_size")

    model = _require(manifest, "model", "manifest")
    scope = model.get("scope")
    if scope not in ALLOWED_SCOPES:
        raise VerificationError(2, f"manifest.model.scope: unsupported scope {scope!r}")
    if scope.startswith("ref2va"):
        _expect(
            manifest.get("h3_packed_layout"),
            PACKED_LAYOUT,
            "manifest.h3_packed_layout",
        )
        policy = _require(manifest, "h3_ref_policy", "manifest")
        for field in ("order", "timestep", "audio_timestep", "audio_before_video", "rope"):
            _require(policy, field, "manifest.h3_ref_policy")
        _expect(manifest.get("h3_ref_limits"), REF_LIMITS, "manifest.h3_ref_limits")
        capabilities = manifest.get("h3_capabilities")
        if capabilities is None or "ref2va" not in capabilities:
            raise VerificationError(
                2, "manifest.h3_capabilities: Ref2VA release requires 'ref2va'"
            )

    artifact = _require(manifest, "artifact", "manifest")
    _expect(artifact.get("filename"), artifact_name, "manifest.artifact.filename")
    size = artifact.get("size_bytes")
    if not isinstance(size, int) or size <= 0:
        raise VerificationError(2, "manifest.artifact.size_bytes must be a positive integer")
    digest = artifact.get("sha256")
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise VerificationError(2, "manifest.artifact.sha256 is not a SHA-256 hex digest")


def check_lock(lock: dict[str, Any]) -> None:
    _expect(lock.get("schema_version"), SCHEMA_VERSION, "lock.schema_version")
    release = _require(lock, "release", "lock")
    for field in ("id", "version", "tag"):
        _require(release, field, "lock.release")
    artifact = _require(lock, "artifact", "lock")
    for field in ("file", "manifest", "size_bytes", "sha256"):
        _require(artifact, field, "lock.artifact")
    if not SHA256_PATTERN.fullmatch(str(artifact["sha256"])):
        raise VerificationError(2, "lock.artifact.sha256 is not a SHA-256 hex digest")
    allowlist = lock.get("artifact_allowlist")
    if not isinstance(allowlist, list) or not allowlist:
        raise VerificationError(2, "lock.artifact_allowlist must be a non-empty list")
    for entry in allowlist:
        if not isinstance(entry, str) or not entry:
            raise VerificationError(2, "lock.artifact_allowlist entries must be strings")
        path = Path(entry)
        if path.is_absolute() or ".." in path.parts or "\\" in entry:
            raise VerificationError(2, f"lock.artifact_allowlist entry is unsafe: {entry!r}")
    image = _require(lock, "image", "lock")

    for field in ("tag", "digest", "base_digest"):
        _require(image, field, "lock.image")
    for field in ("digest", "base_digest"):
        if not DIGEST_PATTERN.fullmatch(str(image[field])):
            raise VerificationError(2, f"lock.image.{field} is not an immutable sha256 digest")
    contract = _require(lock, "contract", "lock")
    _expect(contract.get("h3_contract_version"), 1, "lock.contract.h3_contract_version")
    _expect(contract.get("h3_max_tokens"), MAX_TOKENS, "lock.contract.h3_max_tokens")
    _expect(contract.get("ref_limits"), REF_LIMITS, "lock.contract.ref_limits")
    _expect(contract.get("packed_layout"), PACKED_LAYOUT, "lock.contract.packed_layout")
    publication = _require(lock, "publication", "lock")
    for field in ("github_repository", "huggingface_repository", "huggingface_revision"):
        _require(publication, field, "lock.publication")


def check_artifact(release_dir: Path, lock: dict[str, Any], require: bool) -> dict[str, Any]:
    artifact = lock["artifact"]
    path = release_dir / artifact["file"]
    manifest_path = release_dir / artifact["manifest"]
    manifest = load_json(manifest_path)
    check_manifest(manifest, Path(artifact["file"]).name)
    manifest_artifact = manifest["artifact"]
    if (
        manifest_artifact["sha256"] != artifact["sha256"]
        or manifest_artifact["size_bytes"] != artifact["size_bytes"]
    ):
        raise VerificationError(2, "manifest artifact hash/size does not match the lock")
    if not path.is_file():
        if require:
            raise VerificationError(3, f"missing artifact: {path}")
        # Image-only distribution: the reviewed .so is inside the container image.
        return {
            "artifact_present": False,
            "artifact_sha256": artifact["sha256"],
            "artifact_bytes": artifact["size_bytes"],
            "artifact_source": "image",
        }
    size = path.stat().st_size
    if size != artifact["size_bytes"]:
        raise VerificationError(
            2, f"artifact size {size} != lock size {artifact['size_bytes']}"
        )
    digest = sha256_file(path)
    if digest != artifact["sha256"]:
        raise VerificationError(2, f"artifact sha256 {digest} != lock {artifact['sha256']}")
    if manifest_artifact["sha256"] != digest or manifest_artifact["size_bytes"] != size:
        raise VerificationError(2, "manifest artifact hash/size does not match the artifact")
    return {
        "artifact_present": True,
        "artifact_sha256": digest,
        "artifact_bytes": size,
        "artifact_source": "file",
    }


def check_allowlist(release_dir: Path, lock: dict[str, Any]) -> list[str]:
    """Only the reviewed `.so` and its manifest may live under artifacts/."""
    allowlist = lock["artifact_allowlist"]
    artifact = lock["artifact"]
    required = {artifact["file"], artifact["manifest"]}
    missing = sorted(required - set(allowlist))
    if missing:
        raise VerificationError(
            2, f"lock.artifact_allowlist does not include: {', '.join(missing)}"
        )
    artifacts_dir = release_dir / "artifacts"
    if artifacts_dir.is_dir():
        for path in sorted(artifacts_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix != ".so" and not path.name.endswith(".manifest.json"):
                continue
            relative = path.relative_to(release_dir).as_posix()
            if relative not in allowlist:
                raise VerificationError(2, f"unlisted runtime artifact: {relative}")
    for entry in allowlist:
        if not (release_dir / entry).is_file() and entry not in required:
            raise VerificationError(3, f"allowlisted file is missing: {entry}")
    return allowlist


def check_image(lock: dict[str, Any], image_ref: str) -> dict[str, Any]:
    """Check the image digest and artifact/model labels (image-only distribution)."""
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", image_ref],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise VerificationError(4, "docker is required for --image verification") from error
    except subprocess.CalledProcessError as error:
        message = (error.stderr or error.stdout or "").strip()
        raise VerificationError(3, f"image not found: {image_ref}: {message}") from error
    try:
        data = json.loads(completed.stdout)[0]
    except (json.JSONDecodeError, IndexError, TypeError) as error:
        raise VerificationError(2, f"could not parse docker image inspect output: {error}") from error
    labels = (data.get("Config") or {}).get("Labels") or {}
    expected_sha = lock["artifact"]["sha256"]
    if labels.get("dev.paiton.artifact.sha256") != expected_sha:
        raise VerificationError(
            2,
            "image dev.paiton.artifact.sha256 label does not match the locked artifact hash",
        )
    if labels.get("dev.paiton.model") != "EliovpAI/MiniMax-H3-W4A8-Paiton-RDNA4":
        raise VerificationError(2, "image model label is missing or unexpected")
    image_id = data.get("Id")
    expected_ids = {
        value
        for value in (lock["image"].get("local_image_id"), lock["image"].get("digest"))
        if value
    }
    if image_id not in expected_ids:
        raise VerificationError(
            2, f"image id {image_id} does not match the locked digest/local_image_id"
        )
    return {
        "image_verified": True,
        "image_id": image_id,
        "artifact_label": labels.get("dev.paiton.artifact.sha256"),
    }


def check_licenses(release_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in REQUIRED_LICENSES:
        path = release_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise VerificationError(3, f"missing or empty licence/notice file: {relative}")
        if relative.endswith(".txt") or relative in ("LICENSE", "NOTICE", "MODIFICATIONS.md"):
            hashes[relative] = sha256_file(path)
    notice = (release_dir / "NOTICE").read_text(encoding="utf-8")
    normalized_notice = " ".join(notice.split())
    if " ".join(NOTICE_SENTENCE.split()) not in normalized_notice:
        raise VerificationError(2, "NOTICE is missing the exact MiniMax H3 sentence")
    third_party = (release_dir / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for needle in ("MiniMax H3", "comfy-kitchen", "ComfyUI", "AIMDO"):
        if needle not in third_party:
            raise VerificationError(2, f"THIRD_PARTY_NOTICES.md does not mention {needle!r}")
    return hashes


def check_forbidden_paths(release_dir: Path) -> None:
    for name in (LOCK_NAME, "README.md", "THIRD_PARTY_NOTICES.md", "MODIFICATIONS.md"):
        path = release_dir / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                raise VerificationError(2, f"{name}: forbidden private path/reference {snippet!r}")


def check_readme_digest(release_dir: Path, lock: dict[str, Any], required: bool) -> None:
    """The image digest is authoritative in the lock/label; README pinning is optional."""
    path = release_dir / "README.md"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    tag = lock["image"]["tag"]
    digest = lock["image"]["digest"]
    if tag not in text:
        print(f"warning: README.md does not reference the image tag {tag}", file=sys.stderr)
        return
    if f"{tag}@{digest}" not in text and f"{tag.split(':')[-1]}@{digest}" not in text:
        print(f"warning: README.md does not pin {tag}@{digest}", file=sys.stderr)


def check_checksums(release_dir: Path) -> int:
    path = release_dir / "SHA256SUMS"
    if not path.is_file():
        return 0
    listed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, relative = line.partition("  ")
        if not SHA256_PATTERN.fullmatch(digest) or not relative:
            raise VerificationError(2, f"malformed SHA256SUMS line: {line!r}")
        target = release_dir / relative
        if not target.is_file():
            raise VerificationError(3, f"SHA256SUMS lists missing file: {relative}")
        if sha256_file(target) != digest:
            raise VerificationError(2, f"SHA256SUMS mismatch: {relative}")
        listed += 1
    return listed


def check_checkpoints(lock: dict[str, Any], directory: Path) -> int:
    entries = lock.get("checkpoints", {}).get("files", [])
    if not entries:
        raise VerificationError(2, "lock has no checkpoints.files to verify")
    verified = 0
    for entry in entries:
        relative = entry.get("file")
        expected = entry.get("sha256")
        if not relative or not expected:
            raise VerificationError(2, f"malformed checkpoint entry: {entry!r}")
        path = directory / relative
        if not path.is_file():
            raise VerificationError(3, f"missing checkpoint: {path}")
        if path.stat().st_size != entry.get("bytes", path.stat().st_size):
            raise VerificationError(2, f"checkpoint size mismatch: {relative}")
        if sha256_file(path) != expected:
            raise VerificationError(2, f"checkpoint sha256 mismatch: {relative}")
        verified += 1
    return verified


def verify(
    release_dir: Path,
    *,
    lock_path: Path | None = None,
    artifact: Path | None = None,
    checkpoints: Path | None = None,
    image: str | None = None,
    require_artifact: bool = False,
    require_digest: bool = False,
) -> dict[str, Any]:
    if not release_dir.is_dir():
        raise VerificationError(4, f"release directory not found: {release_dir}")
    lock = load_json(lock_path or release_dir / LOCK_NAME)
    check_lock(lock)
    artifact_allowlist = check_allowlist(release_dir, lock)
    if artifact is not None:
        lock = json.loads(json.dumps(lock))
        lock["artifact"]["file"] = str(artifact)
        lock["artifact"]["manifest"] = str(artifact.with_suffix(".manifest.json"))
    result: dict[str, Any] = {
        "status": "passed",
        "release": lock["release"],
        "image": lock["image"],
        "artifact_allowlist": artifact_allowlist,
    }
    result.update(check_artifact(release_dir, lock, require_artifact or artifact is not None))
    result["licenses"] = check_licenses(release_dir)
    check_forbidden_paths(release_dir)
    check_readme_digest(release_dir, lock, require_digest)
    result["checksums_checked"] = check_checksums(release_dir)
    if image is not None:
        result.update(check_image(lock, image))
    if checkpoints is not None:
        result["checkpoints_checked"] = check_checkpoints(lock, checkpoints)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--lock", type=Path, default=None)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--checkpoints", type=Path, default=None)
    parser.add_argument("--image", default=None, help="verify this local image ref/digest and its artifact label")
    parser.add_argument("--require-artifact", action="store_true", help="deprecated: require a local .so (image-only releases do not ship it)")
    parser.add_argument("--require-digest", action="store_true")
    parser.add_argument("--metadata-only", action="store_true", help="CI mode (default)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = verify(
            args.release_dir.resolve(),
            lock_path=args.lock,
            artifact=args.artifact,
            checkpoints=args.checkpoints,
            image=args.image,
            require_artifact=args.require_artifact,
            require_digest=args.require_digest,
        )
    except VerificationError as error:
        if args.json:
            print(json.dumps({"status": "failed", "error": str(error), "code": error.code}))
        else:
            print(f"FAIL: {error}", file=sys.stderr)
        return error.code

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"OK: {result['release']['id']}")
        print(f"  image:    {result['image']['tag']}@{result['image']['digest']}")
        source = "file" if result.get("artifact_present") else "image-only"
        print(f"  artifact: {result.get('artifact_sha256', 'unknown')} ({source})")
        print(f"  licences: {len(result['licenses'])} hashed")
        print(f"  sums:     {result['checksums_checked']} files")
        if "checkpoints_checked" in result:
            print(f"  weights:  {result['checkpoints_checked']} verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
