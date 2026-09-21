"""One bounded request-execution witness; removed after the first observation.

This confirms a successful native dispatch followed by a completed event on
its stream, outside capture and inside Worker.execute_model. It does not claim
that a Python call during graph capture proves later replay. Graph replay
requires a separate GPU trace.
"""

from __future__ import annotations
import ctypes as C
import importlib
import json
from pathlib import Path
import threading
import time

COMPONENTS = (
    ("rms_decode", "rms-decode", "paiton_rms_decode"),
    ("rms_prefill_fma", "rms-prefill", "paiton_rms_prefill"),
    ("silu_decode", "silu-decode", "paiton_silu_decode"),
    ("silu_prefill", "silu-prefill", "paiton_silu_prefill"),
    ("gdn_fixed", "gdn-norm-fixed", "paiton_gdn_fixed"),
    ("gdn_tiled", "gdn-tiled", "paiton_gdn_tiled"),
    ("prefill_tall", "prefill-tall", "paiton_fp8_direct"),
    ("gdn_merged_prefill", "gdn-merged-prefill", "paiton_fp8_direct"),
)


class Witness:
    def __init__(self, worker_class, plan, report):
        self.worker_class = worker_class
        self.plan = plan
        self.report = report
        self.in_request = False
        self.recorded = False
        self.remaining_steps = 32
        self.restores = []
        self.cleanups = []

    def replace(self, owner, name, wrapper):
        original = getattr(owner, name)
        self.restores.append((owner, name, original, wrapper))
        setattr(owner, name, wrapper)

    def disarm(self):
        for owner, name, original, wrapper in reversed(self.restores):
            if getattr(owner, name) is wrapper:
                setattr(owner, name, original)
        self.restores.clear()
        for cleanup in self.cleanups:
            cleanup()
        self.cleanups.clear()

    def observe(self, library, group, stream, artifact_sha256=None):
        if self.recorded or not self.in_request:
            return
        capture = C.c_int()
        library.hipStreamIsCapturing.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
        if library.hipStreamIsCapturing(stream, C.byref(capture)) or capture.value:
            return
        event = C.c_void_p()
        library.hipEventCreateWithFlags.argtypes = [C.POINTER(C.c_void_p), C.c_uint]
        library.hipEventRecord.argtypes = [C.c_void_p, C.c_void_p]
        library.hipEventQuery.argtypes = [C.c_void_p]
        library.hipEventDestroy.argtypes = [C.c_void_p]
        if library.hipEventCreateWithFlags(C.byref(event), 2):
            raise RuntimeError("Could not create the bounded native execution witness")
        if library.hipEventRecord(event, stream):
            library.hipEventDestroy(event)
            raise RuntimeError("Could not record the native execution witness")
        self.recorded = True
        self.disarm()

        def completed():
            try:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    status = library.hipEventQuery(event)
                    if status == 0:
                        checksum = artifact_sha256
                        if checksum is None:
                            checksum = json.loads(
                                (
                                    Path(self.plan.bundle)
                                    / "native"
                                    / group
                                    / "manifest.json"
                                ).read_text()
                            )["sha256"]
                        self.report(
                            "Executed",
                            plan=self.plan.identity,
                            component=group,
                            artifact_sha256=checksum,
                            phase="request",
                            evidence="native dispatch and completed HIP event outside graph capture",
                            graph_replay="requires separate GPU trace",
                        )
                        return
                    if status != 600:  # hipErrorNotReady
                        break
                    time.sleep(0.01)
                self.report(
                    "ExecutionEvidenceUnavailable",
                    component=group,
                    reason="HIP completion event failed or timed out",
                )
            finally:
                library.hipEventDestroy(event)

        threading.Thread(
            target=completed, name="paiton-first-request-witness", daemon=True
        ).start()

    def wrap_library(self, module, group, symbol):
        library = module.LIB
        original = getattr(library, symbol)

        def dispatch(*args):
            result = original(*args)
            if result == 0:
                self.observe(library, group, args[-1])
            return result

        self.replace(library, symbol, dispatch)

    def arm_worker(self):
        original = self.worker_class.execute_model

        def execute(worker, scheduler_output, *args, **kwargs):
            self.in_request = scheduler_output.total_num_scheduled_tokens > 0
            try:
                return original(worker, scheduler_output, *args, **kwargs)
            finally:
                self.in_request = False
                self.remaining_steps -= 1
                if not self.recorded and self.remaining_steps == 0:
                    self.disarm()
                    self.report(
                        "ExecutionNotYetObserved",
                        plan=self.plan.identity,
                        reason="No eligible uncaptured native dispatch in the first 32 request steps; use a GPU trace to verify graph replay",
                    )

        self.replace(self.worker_class, "execute_model", execute)

    def arm(self):
        self.arm_worker()
        for name, group, symbol in COMPONENTS:
            module = importlib.import_module("paiton_" + name + "_adapter")
            if module.LIB is not None:
                self.wrap_library(module, group, symbol)
            else:
                original_load = module.load

                def load(
                    module=module, group=group, symbol=symbol, original=original_load
                ):
                    result = original()
                    module.load = original
                    if not self.recorded:
                        self.wrap_library(module, group, symbol)
                    return result

                self.replace(module, "load", load)


class MiniCPMWitness(Witness):
    def arm(self):
        self.arm_worker()
        from ..models.paiton_minicpm5_awq import load_region

        library, functions = load_region()
        manifest = json.loads(Path(library._name).with_suffix(".json").read_text())
        for shape, original in list(functions.items()):

            def dispatch(*args, original=original):
                result = original[0](*args)
                if result == 0:
                    self.observe(library, "minicpm5-awq", args[-1], manifest["sha256"])
                return result

            replacement = (dispatch, original[1])
            functions[shape] = replacement

            def restore(shape=shape, original=original, replacement=replacement):
                if functions[shape] is replacement:
                    functions[shape] = original

            self.cleanups.append(restore)


class MoEWitness(Witness):
    def arm(self, worker):
        self.arm_worker()
        runner = worker.model_runner
        model = runner.get_model() if hasattr(runner, "get_model") else runner.model
        for layer in model.modules():
            method = getattr(layer, "quant_method", None)
            if method is None or type(method).__name__ not in (
                "PaitonCoderW4A16Method",
                "PaitonGptOssMxfp4Method",
            ):
                continue
            library, original, manifest = method.region

            def dispatch(*args, library=library, original=original, manifest=manifest):
                result = original(*args)
                if result == 0:
                    self.observe(
                        library,
                        self.plan.resolution.adapter,
                        args[-1],
                        manifest["sha256"],
                    )
                return result

            self.replace(method, "region", (library, dispatch, manifest))


class CompiledModelWitness(Witness):
    def arm(self):
        self.arm_worker()
        from ..runtime.core.model import Model

        original = Model._run_impl
        inventory = json.loads((Path(self.plan.bundle) / "execution.json").read_text())[
            "files"
        ]

        def dispatch(model, *args, **kwargs):
            result = original(model, *args, **kwargs)
            library = model.memloader.lib
            name = Path(library._name).name
            if name in inventory:
                stream = kwargs.get("stream_ptr", args[2] if len(args) > 2 else None)
                self.observe(
                    library,
                    self.plan.resolution.adapter,
                    stream,
                    inventory[name]["sha256"],
                )
            return result

        self.replace(Model, "_run_impl", dispatch)


def install(worker_class, plan, report):
    original = worker_class.compile_or_warm_up_model

    def warmup(worker, *args, **kwargs):
        result = original(worker, *args, **kwargs)
        worker_class.compile_or_warm_up_model = original
        from .adapters import PLATFORM

        adapter = plan.resolution.adapter
        if adapter in ("qwen3-coder-v1", "gptoss-mxfp4-v1"):
            MoEWitness(worker_class, plan, report).arm(worker)
        else:
            cls = (
                MiniCPMWitness
                if adapter == "minicpm5-awq-v1"
                else CompiledModelWitness
                if adapter in PLATFORM
                else Witness
            )
            cls(worker_class, plan, report).arm()
        return result

    worker_class.compile_or_warm_up_model = warmup
