"""Offline controls for Python preparation and the qualified native runtime."""

import ipaddress
import sys
from pathlib import Path
from .contracts import PreparationError

_installed = False


def environment(bundle: Path, inherited: dict) -> dict:
    guard = (bundle / "support/offline_guard.so").resolve(strict=True)
    if any(c.isspace() or c == ":" for c in str(guard)):
        raise PreparationError(
            "offline-path",
            "Offline mode needs a --cache-dir without spaces or colons for the native transport guard.",
        )
    result = dict(inherited)
    result.update(
        PAITON_OFFLINE_ACTIVE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1",
        VLLM_NO_USAGE_STATS="1",
        VLLM_HOST_IP="127.0.0.1",
    )
    result["LD_PRELOAD"] = str(guard) + (
        " " + result["LD_PRELOAD"] if result.get("LD_PRELOAD") else ""
    )
    return result


def validate_native_guard() -> None:
    import ctypes

    try:
        fn = ctypes.CDLL(None).paiton_offline_guard_version
        fn.restype = ctypes.c_int
        if fn() != 1:
            raise ValueError("version")
    except (AttributeError, OSError, ValueError) as exc:
        raise PreparationError(
            "offline-guard",
            "Offline native transport guard was not preloaded; start a new process with the prepared launcher.",
        ) from exc


def install():
    global _installed
    if _installed:
        return

    def audit(event, args):
        if event == "socket.getaddrinfo":
            host = args[0]
            if host in (None, "", "localhost", "127.0.0.1", "::1"):
                return
            try:
                if ipaddress.ip_address(host).is_loopback:
                    return
            except ValueError:
                pass
            raise PermissionError("Paiton offline mode blocks network name resolution")
        if event in ("socket.connect", "socket.sendto"):
            address = args[-1]
            if isinstance(address, (str, bytes)):
                return  # local Unix IPC
            try:
                if ipaddress.ip_address(address[0]).is_loopback:
                    return
            except (ValueError, TypeError, IndexError):
                pass
            raise PermissionError("Paiton offline mode blocks outbound network access")

    sys.addaudithook(audit)
    _installed = True
