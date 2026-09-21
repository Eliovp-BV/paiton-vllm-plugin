"""Named, explicit serving profiles; paths and repository IDs keep their meaning."""

from .catalogue import catalogue
from .contracts import PreparationError
from .cache import cache_root, locked
import json
import os
import re
from pathlib import Path
import tempfile


def presets() -> dict[str, tuple[dict, str]]:
    result = {}
    for package in catalogue()["packages"]:
        for name, profile in package.get("presets", {}).items():
            if (
                not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name)
                or name in result
                or profile not in package["profiles"]
            ):
                raise PreparationError("catalogue", "Invalid serving preset: " + name)
            result[name] = (package, profile)
    return result


def select(
    reference: str, profile: str | None, model_dir=None, *, cache_dir=None
) -> tuple[str, str | None]:
    """A named preset explicitly selects its disclosed serving settings."""
    # Preserve the original CLI meaning of an existing relative directory,
    # including one whose name happens to be a preset. './NAME' is always a path.
    entry = (
        None
        if model_dir is None and Path(reference).expanduser().is_dir()
        else presets().get(reference)
    )
    if entry is None:
        if model_dir is not None:
            raise PreparationError(
                "model-dir", "--model-dir requires a named preset from paiton models."
            )
        return reference, profile
    package, selected = entry
    if profile is not None and profile != selected:
        raise PreparationError(
            "profile-conflict",
            f"Preset {reference} selects {selected}; remove --profile.",
        )
    if model_dir is None:
        binding = cache_root(cache_dir) / "presets" / (reference + ".json")
        if binding.exists():
            try:
                value = json.loads(binding.read_text())
                if (
                    not isinstance(value, dict)
                    or not isinstance(value.get("model_dir"), str)
                    or not value["model_dir"]
                ):
                    raise ValueError("Expected a model directory in the stored preset")
                if {
                    k: value.get(k)
                    for k in ("binding_schema", "profile", "repository", "revision")
                } != {
                    "binding_schema": 1,
                    "profile": selected,
                    "repository": package["model"]["repository"],
                    "revision": package["model"]["revision"],
                }:
                    raise ValueError(
                        "The stored preset refers to a different checkpoint/profile"
                    )
                model_dir = value["model_dir"]
            except (OSError, ValueError, KeyError) as error:
                raise PreparationError(
                    "preset-binding",
                    "Stored preset is invalid; prepare again with --model-dir: "
                    + str(error),
                ) from error
    if model_dir is not None:
        path = Path(model_dir).expanduser().resolve()
        if not path.is_dir():
            raise PreparationError(
                "model-dir", "Model directory does not exist: " + str(path)
            )
        return str(path), selected
    return package["model"]["repository"], selected


def remember(name, plan, *, cache_dir=None):
    """Remember an explicitly supplied local checkpoint after successful verification.

    This is user configuration, separate from immutable launch plans/locks.
    It stores no arguments or credentials and never bypasses model inspection.
    """
    package, profile = presets()[name]
    if (
        package["id"] != plan.resolution.package_id
        or profile != plan.resolution.profile
        or not plan.resolution.model.verified
    ):
        raise PreparationError(
            "preset-binding", "Only the verified matching preset can be remembered."
        )
    path = cache_root(cache_dir) / "presets" / (name + ".json")
    value = {
        "binding_schema": 1,
        "profile": profile,
        "repository": plan.resolution.model.repository,
        "revision": plan.resolution.model.revision,
        "model_dir": plan.resolution.model.path,
    }
    with locked(path.with_suffix(".lock")):
        fd, temporary = tempfile.mkstemp(prefix=".preset-", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(value, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
