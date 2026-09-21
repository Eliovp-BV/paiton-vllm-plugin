"""Installed, versioned trust root. No mutable remote catalogue."""

import json
from pathlib import Path
from .contracts import PreparationError, digest

DATA = Path(__file__).with_name("data")


def catalogue() -> dict:
    value = json.loads((DATA / "catalogue.json").read_text())
    if value.get("catalogue_schema") != 1:
        raise PreparationError(
            "catalogue-schema", "Unsupported installed catalogue schema."
        )
    return value


def package_by_id(identifier: str) -> dict:
    found = [p for p in catalogue()["packages"] if p["id"] == identifier]
    if len(found) != 1:
        raise PreparationError(
            "package", "Package is absent or ambiguous in the installed catalogue."
        )
    return found[0]


def manifest(package: dict) -> dict:
    name = package["manifest"]
    if Path(name).name != name:
        raise PreparationError("catalogue-path", "Invalid installed manifest path.")
    return json.loads((DATA / name).read_text())


def package_digest(package: dict) -> str:
    return digest(manifest(package))
