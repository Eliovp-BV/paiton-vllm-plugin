"""Reject inherited execution overrides outside the selected profile."""

import os
from ..activation import activate
from .contracts import PreparationError

TRANSPORT = {
    "VLLM_PLUGINS",
    "VLLM_LOGGING_LEVEL",
    "VLLM_CONFIGURE_LOGGING",
    "VLLM_LOGGING_CONFIG_PATH",
    "VLLM_LOGGING_PREFIX",
    "VLLM_NO_USAGE_STATS",
    "VLLM_DO_NOT_TRACK",
    "VLLM_CACHE_ROOT",
    "VLLM_HOST_IP",
    "VLLM_PORT",
    "VLLM_RPC_TIMEOUT",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_TARGET_DEVICE",
}


def validate(package: dict, profile: str, *, offline: bool, inherited=None) -> None:
    inherited = os.environ if inherited is None else inherited
    from .adapters import PLATFORM

    activate(
        "native", dict(inherited), requires_platform=package["adapter"] in PLATFORM
    )
    expected = package["profiles"][profile]["environment"]
    special = {"PAITON_PLUGIN_MODE", "PAITON_GPU_ARCH"}
    if offline:
        special.add("PAITON_OFFLINE_ACTIVE")
    for key, value in inherited.items():
        if key in expected:
            required = expected[key]
            if (
                "{bundle}" not in required
                and "{model}" not in required
                and value != required
            ):
                raise PreparationError(
                    "environment-conflict",
                    key
                    + " conflicts with the explicit profile. Unset it or select compatible settings.",
                )
        elif (
            key.startswith(("PAITON_", "_PAITON_", "RADIANCE_", "R4D_", "VLLM_"))
            and key not in special | TRANSPORT
        ):
            raise PreparationError(
                "environment-override",
                "Unqualified execution override "
                + key
                + ". Unset it before using the native launcher.",
            )
    if (
        inherited.get("PAITON_GPU_ARCH", package["runtime"]["gpu_arch"])
        != package["runtime"]["gpu_arch"]
    ):
        raise PreparationError(
            "environment-conflict",
            "PAITON_GPU_ARCH conflicts with the execution package.",
        )
    if inherited.get("VLLM_WORKER_MULTIPROC_METHOD", "spawn") != "spawn":
        raise PreparationError(
            "worker-method",
            "Native prepared workers require spawn; unset VLLM_WORKER_MULTIPROC_METHOD.",
        )
    if inherited.get("VLLM_TARGET_DEVICE", "rocm") != "rocm":
        raise PreparationError(
            "environment-conflict", "This package requires the ROCm runtime."
        )
    if offline and inherited.get("VLLM_HOST_IP", "127.0.0.1") not in (
        "127.0.0.1",
        "::1",
    ):
        raise PreparationError(
            "offline-ip",
            "Offline worker communication requires a loopback VLLM_HOST_IP.",
        )
