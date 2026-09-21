"""One explicit activation contract, shared by discovery and launchers."""

from __future__ import annotations
import os
from typing import Mapping, MutableMapping

MODES = {"off", "models", "dflash", "legacy", "native"}
LEGACY_FLAGS = (
    "VLLM_USE_PAITON_PLATFORM",
    "VLLM_PAITON_VANILLA_ROCM_PLATFORM",
    "PAITON_RUNTIME_COMPAT_FLOW",
    "PAITON_EXPERIMENTAL_MXFP4_W4A8",
)


def mode(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    selected = env.get("PAITON_PLUGIN_MODE")
    if selected is not None and selected not in MODES:
        raise ValueError(
            "PAITON_PLUGIN_MODE must be off, models, dflash, legacy or native"
        )
    legacy = any(env.get(k) == "1" for k in LEGACY_FLAGS)
    if selected == "off" and (legacy or env.get("PAITON_ACTIVE_PLAN")):
        raise ValueError(
            "Conflicting Paiton activation: mode=off with explicit enable settings"
        )
    if (
        env.get("VLLM_DISABLE_PAITON_PLATFORM") == "1"
        and env.get("VLLM_USE_PAITON_PLATFORM") == "1"
    ):
        raise ValueError(
            "Conflicting VLLM_USE_PAITON_PLATFORM and VLLM_DISABLE_PAITON_PLATFORM"
        )
    if selected is not None:
        return selected
    # Compatibility for documented release commands. Device architecture alone
    # never enables anything. An allowlist is permission, not activation.
    return "legacy" if legacy else "off"


def enabled(entry_point: str) -> bool:
    selected = mode()
    if selected == "off":
        return False
    return entry_point != "paiton_platform" or selected == "legacy"


def activate(
    selected: str, env: MutableMapping[str, str], *, requires_platform: bool = False
) -> None:
    if selected not in MODES - {"off"}:
        raise ValueError("invalid activation mode")
    if env.get("PAITON_PLUGIN_MODE") == "off":
        raise ValueError("PAITON_PLUGIN_MODE=off conflicts with explicit Paiton launch")
    required = {"register_paiton_models"}
    if (
        (selected == "legacy" or requires_platform)
        and env.get("PAITON_RUNTIME_COMPAT_FLOW") != "1"
        and env.get("VLLM_DISABLE_PAITON_PLATFORM") != "1"
    ):
        required.add("paiton_platform")
    if "VLLM_PLUGINS" in env:
        allowed = {x.strip() for x in env["VLLM_PLUGINS"].split(",") if x.strip()}
        missing = required - allowed
        if missing:
            raise ValueError(
                "VLLM_PLUGINS explicitly excludes required hooks: "
                + ", ".join(sorted(missing))
                + ". Add them to your allowlist or unset VLLM_PLUGINS."
            )
    env["PAITON_PLUGIN_MODE"] = selected
    mode(env)


def activate_model_launcher(stock: bool, env: MutableMapping[str, str]) -> None:
    """Compatibility for the published per-model Python server commands."""
    if stock:
        env["PAITON_PLUGIN_MODE"] = "off"
        mode(env)
    else:
        activate("models", env)
