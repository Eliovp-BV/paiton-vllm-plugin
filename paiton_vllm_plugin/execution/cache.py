"""Content-addressed preparation; installed catalogue is the executable trust root."""

from __future__ import annotations
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request
from ..artifact_manifest import validate_execution_package, verify_file
from .catalogue import manifest, package_digest
from .contracts import PreparationError, canonical


def cache_root(value: str | Path | None = None) -> Path:
    return (
        Path(value).expanduser()
        if value
        else Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "paiton"
    ).resolve()


@contextmanager
def locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def destination(root: Path, package: dict) -> Path:
    return root / "executions" / package_digest(package)


def verify_package(root: Path, package: dict) -> None:
    try:
        actual = json.loads((root / "execution.json").read_text())
    except (OSError, ValueError) as exc:
        raise PreparationError(
            "manifest", "Execution-package manifest is missing or invalid."
        ) from exc
    expected = manifest(package)
    if actual != expected:
        raise PreparationError(
            "manifest-trust",
            "Execution manifest differs from the pinned installed catalogue.",
        )
    validate_execution_package(
        root, expected, expected_arch=package["runtime"]["gpu_arch"], expected_tp_size=1
    )


def extract_archive(archive: Path, dest: Path, inventory: dict) -> None:
    """Only declared regular files. No links, devices, traversal or tarfile.extract."""
    expected = dict(inventory["files"])
    seen = set()
    with tarfile.open(archive, "r:*") as stream:
        for member in stream:
            name = member.name
            relative = Path(name)
            if (
                not member.isfile()
                or relative.is_absolute()
                or ".." in relative.parts
                or str(relative) != name
                or name in seen
            ):
                raise PreparationError(
                    "archive", "Unsafe or duplicate archive member: " + name
                )
            if name not in expected and name != "execution.json":
                raise PreparationError("archive", "Undeclared archive member: " + name)
            bound = expected[name]["size_bytes"] if name in expected else 1024 * 1024
            if member.size > bound:
                raise PreparationError(
                    "archive", "Archive member exceeds declared size: " + name
                )
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            if source is None:
                raise PreparationError("archive", "Missing archive member data.")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            seen.add(name)
    if seen != set(expected) | {"execution.json"}:
        raise PreparationError("archive", "Incomplete archive.")


def prepare_package(
    root: Path, package: dict, *, source: Path | None = None, offline: bool = False
) -> Path:
    final = destination(root, package)
    with locked(root / "locks" / (package_digest(package) + ".lock")):
        if final.exists():
            try:
                verify_package(final, package)
            except Exception as exc:
                raise PreparationError(
                    "cache-corrupt",
                    "Prepared execution package is corrupt. Use a fresh --cache-dir; live package files will not be replaced.",
                ) from exc
            return final
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix="." + final.name + ".partial-", dir=final.parent)
        )
        try:
            if source is not None:
                source = source.resolve(strict=True)
                verify_package(source, package)
                for name in manifest(package)["files"]:
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / name, target)
                shutil.copyfile(source / "execution.json", staging / "execution.json")
            else:
                remote = package.get("archive")
                if offline and remote:
                    raise PreparationError(
                        "payload-unavailable",
                        "The pinned bundle is not in this cache. Offline mode cannot fetch it; prepare online first or supply a verified local --bundle.",
                    )
                if offline or not remote:
                    raise PreparationError(
                        "payload-unavailable",
                        "The pinned native bundle is not prepared. This bundle is unpublished; supply its verified local directory with --bundle and --prepare-only.",
                    )
                if not remote["url"].startswith("https://"):
                    raise PreparationError(
                        "origin", "Artifact origin must be pinned HTTPS metadata."
                    )
                archive = staging / ".download.partial"
                with (
                    urllib.request.urlopen(remote["url"], timeout=60) as response,
                    archive.open("xb") as output,
                ):
                    if not response.geturl().startswith("https://"):
                        raise PreparationError(
                            "origin", "Artifact redirect downgraded HTTPS."
                        )
                    remaining = remote["size_bytes"]
                    while chunk := response.read(min(1024 * 1024, remaining + 1)):
                        remaining -= len(chunk)
                        if remaining < 0:
                            raise PreparationError(
                                "download", "Artifact exceeds declared size."
                            )
                        output.write(chunk)
                verify_file(archive, remote)
                extract_archive(archive, staging, manifest(package))
                archive.unlink()
            verify_package(staging, package)
            os.rename(staging, final)
            return final
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def write_json_immutable(path: Path, value: dict) -> None:
    """Identical writes are idempotent; different content never replaces a plan."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with locked(path.parent / ("." + path.name + ".lock")):
        if path.exists():
            if canonical(json.loads(path.read_text())) != canonical(value):
                raise PreparationError(
                    "immutable-file",
                    "Existing plan/lock differs; choose a new output path.",
                )
            return
        fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as out:
                out.write(canonical(value) + "\n")
                out.flush()
                os.fsync(out.fileno())
            os.rename(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
