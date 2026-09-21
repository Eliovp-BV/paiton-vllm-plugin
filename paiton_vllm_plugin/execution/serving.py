"""Verified release configuration beside unchanged source checkpoint files."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import tempfile

from .cache import locked, write_json_immutable
from .catalogue import manifest
from .contracts import PreparationError, digest
from .inspection import fingerprint, sha256, validate_source


def inventory(package, model, bundle):
    """Return the trusted file set, excluding support files unused by model loaders."""
    source = Path(model.path)
    result = {
        name: (source / name, record, name.endswith((".safetensors", ".gguf")))
        for name, record in package["model"]["files"].items()
    }
    for name, record in manifest(package)["files"].items():
        if not name.startswith("support/"):
            result[name] = (bundle / name, record, False)
    if package.get("serving_layout") == "ornith-resharded":
        result.pop("model.safetensors")
        for name, record in package["resharded_files"].items():
            result[name] = (None, record, False)
    return result


def prepare(root, package, model, bundle):
    layout = package.get("serving_layout")
    if layout is None:
        return None, ()
    if layout == "bundle":
        return str(bundle), ()
    if layout not in ("overlay", "ornith-resharded"):
        raise PreparationError("serving-layout", "Unknown model preparation layout.")
    key = digest([asdict(model), manifest(package), layout])
    target = root / "serving-models" / key
    expected = inventory(package, model, bundle)
    with locked(target.with_suffix(".lock")):
        if not target.exists():
            temp = Path(tempfile.mkdtemp(prefix=".prepare-", dir=target.parent))
            try:
                if layout == "ornith-resharded":
                    from ..ornith_reshard import reshard_checkpoint

                    reshard_checkpoint(Path(model.path) / "model.safetensors", temp)
                    # The public resharder emits a local report; the verified
                    # model directory has only the declared engine inputs.
                    (temp / "reshard-report.json").unlink(missing_ok=True)
                for name, (source, record, link) in expected.items():
                    path = temp / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if source is not None:
                        if link:
                            path.symlink_to(source.resolve())
                        else:
                            shutil.copy2(source, path)
                validate_source(model)
                records = verify(temp, expected, full=True)
                os.replace(temp, target)
                _remember_verification(root, target, records)
            except BaseException:
                shutil.rmtree(temp, ignore_errors=True)
                raise
        else:
            records = verify(target, expected)
            receipt, value = _verification_receipt(root, target, records)
            try:
                reused = receipt.is_file() and json.loads(receipt.read_text()) == value
            except (OSError, ValueError) as error:
                raise PreparationError(
                    "serving-receipt",
                    "Prepared-model verification receipt is unreadable; use a fresh --cache-dir.",
                ) from error
            if not reused:
                records = verify(target, expected, full=True)
                verify(target, expected, records=records)
                _remember_verification(root, target, records)
    return str(target), records


def _verification_receipt(root, target, records):
    # Like source-weight receipts, this private-cache entry binds trusted
    # digests to device/inode/ctime/mtime/size. Reused Ornith shards must not
    # require another complete 23 GB read on every server launch.
    value = {
        "serving_verification_schema": 1,
        "directory": str(target.resolve()),
        "files": [list(row) for row in records],
    }
    return root / "verified-serving-models" / (digest(value) + ".json"), value


def _remember_verification(root, target, records):
    path, value = _verification_receipt(root, target, records)
    write_json_immutable(path, value)


def verify(root, expected, *, full=False, records=()):
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() or p.is_symlink()
    }
    if root.is_symlink() or actual != set(expected):
        raise PreparationError(
            "serving-inventory", "Prepared model directory changed; use a fresh cache."
        )
    previous = {r[0]: r for r in records}
    result = []
    for name, (source, record, link) in expected.items():
        path = root / name
        if any(p.is_symlink() for p in path.parents if root in p.parents):
            raise PreparationError(
                "serving-link", "Prepared model contains an unexpected directory link."
            )
        if link:
            if not path.is_symlink() or path.resolve() != source.resolve():
                raise PreparationError(
                    "serving-link", "Prepared checkpoint link changed: " + name
                )
        elif path.is_symlink():
            raise PreparationError(
                "serving-link", "Prepared runtime file became a link: " + name
            )
        row = (name, record["sha256"], *fingerprint(path))
        if row[4] != record["size_bytes"] or (records and previous.get(name) != row):
            raise PreparationError(
                "serving-changed", "Prepared model file changed: " + name
            )
        if full and not link and sha256(path) != record["sha256"]:
            raise PreparationError(
                "serving-digest", "Prepared model checksum mismatch: " + name
            )
        result.append(row)
    return tuple(result)


def validate(plan, package):
    layout = package.get("serving_layout")
    if layout is None:
        if plan.serving_model is not None or plan.serving_files:
            raise PreparationError(
                "serving-layout", "Unexpected model overlay in this plan."
            )
    elif layout == "bundle":
        if plan.serving_model != plan.bundle or plan.serving_files:
            raise PreparationError(
                "serving-layout",
                "This package serves its verified configuration bundle.",
            )
    else:
        if not plan.serving_model or not plan.serving_files:
            raise PreparationError("serving-layout", "Missing prepared model overlay.")
        verify(
            Path(plan.serving_model),
            inventory(package, plan.resolution.model, Path(plan.bundle)),
            records=plan.serving_files,
        )
