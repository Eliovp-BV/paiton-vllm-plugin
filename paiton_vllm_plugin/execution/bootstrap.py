"""Explicit process-start preparation, inherited unchanged by spawned workers."""

from __future__ import annotations
import ctypes
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys
from .contracts import LaunchPlan, PreparationError, digest
from .api import validate_plan
from .catalogue import DATA
from ..activation import mode
from ..artifact_manifest import verify_file
from .adapters import MODELS, PLATFORM

_plan = None
_registered = False
_handles = []
VENDOR = Path(__file__).parents[1] / "_vendor/qwen38"
GROUPS = {
    "silu_prefill": "silu-prefill",
    "silu_decode": "silu-decode",
    "rms_decode": "rms-decode",
    "rms_prefill_fma": "rms-prefill",
    "prefill_tall": "prefill-tall",
    "gdn_fixed": "gdn-norm-fixed",
    "gdn_tiled": "gdn-tiled",
    "gdn_merged_prefill": "gdn-merged-prefill",
}


def report(stage: str, **details):
    print(
        json.dumps(
            {"paiton_stage": stage, "pid": os.getpid(), **details}, sort_keys=True
        ),
        file=sys.stderr,
        flush=True,
    )


class Aliases(importlib.abc.MetaPathFinder):
    def __init__(self, bundle: Path):
        self.bundle = bundle
        receipts = json.loads((DATA / "qwen38-adapters.json").read_text())
        self.names = {
            Path(n).stem
            for n in receipts["files"]
            if "/" not in n and n.endswith(".py")
        }

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "paiton_runtime_compat":
            return importlib.util.spec_from_file_location(
                fullname,
                VENDOR / "compat/__init__.py",
                submodule_search_locations=[str(VENDOR / "compat")],
            )
        if "." in fullname:
            return None
        if fullname in ("r4d", "radiance_mxfp4_fp8"):
            return importlib.util.spec_from_file_location(
                fullname, self.bundle / (fullname + ".so")
            )
        candidate = VENDOR / (fullname + ".py")
        if fullname in self.names and candidate.is_file():
            return importlib.util.spec_from_file_location(fullname, candidate)
        return None


def _overlays():
    # Reuse the release's source-hash and overlay-hash guarded loader. Relocate
    # only origin paths to the currently installed packages; never edit them.
    bootstrap = importlib.import_module("paiton_runtime_compat.bootstrap")
    original = json.loads((VENDOR / "compat/compat_manifest.json").read_text())
    entries = {}
    for path, entry in original["files"].items():
        relative = path.split("/dist-packages/", 1)[-1]
        top, _, tail = relative.partition("/")
        spec = importlib.machinery.PathFinder.find_spec(top)
        if spec is None or not spec.submodule_search_locations:
            raise PreparationError(
                "overlay-runtime", "Missing installed runtime component: " + top
            )
        root = Path(next(iter(spec.submodule_search_locations)))
        actual = root / tail
        if (
            not actual.is_file()
            or hashlib.sha256(actual.read_bytes()).hexdigest()
            != entry["original_sha256"]
        ):
            raise PreparationError(
                "runtime-source",
                "The qualified original source differs: "
                + relative
                + ". Use the matching existing vLLM build; no patch was applied.",
            )
        entries[str(actual)] = entry
    early = [
        name
        for name, module in sys.modules.items()
        if getattr(module, "__file__", None) in entries
    ]
    if early:
        raise PreparationError(
            "late-activation",
            "Paiton must activate before runtime imports. Start a fresh process: "
            + ", ".join(early),
        )
    sys.meta_path.insert(
        0, bootstrap.OverlayFinder({"files": entries}, VENDOR / "compat")
    )
    bootstrap._installed = True


def initialize():
    global _plan
    path = os.environ.get("PAITON_ACTIVE_PLAN")
    if not path:
        return
    if mode() != "native":
        raise PreparationError(
            "activation", "A native launch plan requires PAITON_PLUGIN_MODE=native."
        )
    value = json.loads(Path(path).read_text())
    if digest(value) != os.environ.get("PAITON_ACTIVE_PLAN_SHA256"):
        raise PreparationError("plan-digest", "Launch-plan digest mismatch.")
    plan = LaunchPlan.from_dict(value)
    if _plan is not None:
        if _plan.identity != plan.identity:
            raise PreparationError(
                "hot-switch",
                "Paiton process state cannot be changed after initialization.",
            )
        return
    validate_plan(plan)
    for key, expected in plan.environment:
        if os.environ.get(key) != expected:
            raise PreparationError(
                "worker-environment",
                "Worker environment differs from the validated plan: " + key,
            )
    if plan.offline:
        from .offline import install, validate_native_guard

        validate_native_guard()
        install()
    if plan.resolution.adapter in MODELS:
        if "vllm" in sys.modules:
            raise PreparationError(
                "late-activation",
                "Native activation must precede vLLM imports; start a fresh process.",
            )
        _plan = plan
        return
    if plan.resolution.adapter != "qwen38-runtime-v1":
        raise PreparationError(
            "adapter", "This wheel has no implementation of the selected adapter."
        )
    receipts = json.loads((DATA / "qwen38-adapters.json").read_text())
    for name, record in receipts["files"].items():
        verify_file(VENDOR / name, record)
    for name in ("radiance_kernels", "radiance_mxfp4", "r4d", "paiton_runtime_compat"):
        if name in sys.modules:
            raise PreparationError(
                "late-activation",
                "Start a fresh Python process; " + name + " was already imported.",
            )
    sys.meta_path.insert(0, Aliases(Path(plan.bundle)))
    _overlays()
    # These are the released external hooks. Their native handles remain lazy;
    # only startup/dispatch integration is installed here.
    importlib.import_module("radiance_amdsmi")
    for module, group in GROUPS.items():
        adapter = importlib.import_module("paiton_" + module + "_adapter")
        adapter.ROOT = Path(plan.bundle) / "native" / group
    importlib.import_module("paiton_prefill_attention_adapter").install()
    if os.environ.get("REFERENCE_SAFETENSORS_PREAD") == "1":
        import safetensors

        original = safetensors.safe_open

        def pread(*args, **kwargs):
            kwargs.setdefault("backend", "pread")
            return original(*args, **kwargs)

        safetensors.safe_open = pread
    _plan = plan


def register_worker():
    global _registered
    initialize()
    if _plan is None:
        raise PreparationError(
            "activation",
            "Native mode requires a prepared immutable launch plan. Use paiton vllm serve.",
        )
    if _registered:
        return
    # Existing plugin integration; it keeps qualified upstream operators.
    if _plan.resolution.adapter == "qwen38-runtime-v1":
        importlib.import_module("paiton_runtime_compat").register()
    else:
        from vllm import ModelRegistry

        ModelRegistry.register_model(
            MODELS[_plan.resolution.adapter][0],
            "paiton_vllm_plugin.models."
            + MODELS[_plan.resolution.adapter][1]
            + ":"
            + MODELS[_plan.resolution.adapter][0],
        )
        if _plan.resolution.adapter == "qwen38-neo-v1":
            from ..gguf_detokenizer import install_gguf_detokenizer_compat

            install_gguf_detokenizer_compat()
            importlib.import_module("paiton_vllm_plugin.gguf_model_loader")
    from vllm.v1.worker.gpu_worker import Worker

    original = Worker.init_device

    def init_device(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        import torch

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if (
            props.gcnArchName.split(":")[0] != "gfx1201"
            or torch.cuda.device_count() != 1
        ):
            raise PreparationError(
                "worker-device",
                "The worker did not select the qualified single gfx1201 device.",
            )
        validate_plan(_plan)
        inventory = json.loads((Path(_plan.bundle) / "execution.json").read_text())
        for name in inventory["files"]:
            if not name.startswith("support/") and name.endswith(".so"):
                _handles.append(
                    ctypes.CDLL(str(Path(_plan.bundle) / name), mode=ctypes.RTLD_LOCAL)
                )
        report(
            "Loaded",
            package=_plan.resolution.package_id,
            plan=_plan.identity,
            device=props.gcnArchName,
            evidence="verified native libraries loaded; request execution has not yet been observed",
        )
        return result

    Worker.init_device = init_device
    from .witness import install as install_witness

    install_witness(Worker, _plan, report)
    _registered = True


def platform_plugin():
    initialize()
    if _plan is None:
        raise PreparationError(
            "activation", "Native mode requires a prepared launch plan."
        )
    if _plan.resolution.adapter in PLATFORM:
        return "paiton_vllm_plugin.paiton_platform.PaitonPlatform"
    return None
