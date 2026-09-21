"""Shared model discovery; no framework imports, downloads or device probes."""

from dataclasses import asdict
from .catalogue import catalogue, package_digest
from .cache import cache_root, destination, verify_package
from .inspection import inspect_environment
from .resolver import check_environment
from .contracts import PreparationError


def discover(cache_dir=None) -> dict:
    environment = inspect_environment()
    rows = []
    cat = catalogue()
    for package in cat["packages"]:
        path = destination(cache_root(cache_dir), package)
        payload = "unavailable"
        reasons = []
        if path.is_dir():
            try:
                verify_package(path, package)
                payload = "prepared"
            except Exception as error:
                payload = "corrupt"
                reasons.append(str(error))
        runtime = "metadata-compatible"
        try:
            check_environment(environment, package["runtime"], require_gpu=False)
        except PreparationError as error:
            runtime = "incompatible"
            reasons.append(str(error))
        rows.append(
            {
                "package": package["id"],
                "presets": package.get("presets", {}),
                "repository": package["model"]["repository"],
                "revision": package["model"]["revision"],
                "profiles": {
                    k: v["description"] for k, v in package["profiles"].items()
                },
                "status": "incompatible" if runtime == "incompatible" else payload,
                "payload_status": payload,
                "runtime_status": runtime,
                "qualification": "pinned release payload; GPU/checkpoint verification required before launch",
                "qualified_runtime": package["runtime"],
                "release_status": package["release_status"],
                "package_digest": package_digest(package),
                "reasons": reasons,
            }
        )
    return {
        "catalogue_version": cat["catalogue_version"],
        "environment": asdict(environment),
        "gpu_probe": "not requested",
        "models": rows,
    }
