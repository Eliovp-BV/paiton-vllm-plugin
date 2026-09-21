"""CPU contract tests. Native dispatch/inference qualification runs separately."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paiton_vllm_plugin.execution import api, cache, inspection, resolver
from paiton_vllm_plugin.execution.contracts import (
    Environment,
    LaunchPlan,
    PreparationError,
)
from paiton_vllm_plugin.execution.witness import Witness

catalogue_module = importlib.import_module("paiton_vllm_plugin.execution.catalogue")


def receipt(path):
    return {
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "renamed checkpoint"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"architectures":["Fixture"]}')
        (self.model / "model.safetensors").write_bytes(b"weights!")
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        (self.bundle / "kernel.so").write_bytes(b"native fixture")
        self.manifest = {
            "execution_schema": 1,
            "binary_abi_version": 1,
            "target": {"arch": "gfx1201", "wave_size": 32},
            "tp_size": 1,
            "files": {"kernel.so": receipt(self.bundle / "kernel.so")},
        }
        (self.bundle / "execution.json").write_text(json.dumps(self.manifest))
        self.package = {
            "id": "fixture",
            "manifest": "fixture.json",
            "adapter": "fixture-adapter",
            "archive": None,
            "runtime": {
                "python": "3.12",
                "system": "Linux",
                "machine": "x86_64",
                "versions": {},
                "gpu_arch": "gfx1201",
                "gpu_count": 1,
            },
            "model": {
                "repository": "example/fixture",
                "revision": "a" * 40,
                "storage_format": "fixture",
                "files": {p.name: receipt(p) for p in self.model.iterdir()},
            },
            "profiles": {
                "explicit": {
                    "description": "fixture",
                    "environment": {},
                    "settings": {"tensor-parallel-size": 1, "max-model-len": 64},
                    "requirements": {"tensor-parallel-size": 1},
                    "bounds": {"max-model-len": [1, 64]},
                }
            },
        }
        data = self.root / "data"
        data.mkdir()
        (data / "catalogue.json").write_text(
            json.dumps(
                {
                    "catalogue_schema": 1,
                    "catalogue_version": "fixture",
                    "packages": [self.package],
                }
            )
        )
        (data / "fixture.json").write_text(json.dumps(self.manifest))
        self.addCleanup(patch.stopall)
        patch.object(catalogue_module, "DATA", data).start()
        self.environment = Environment(
            sys.executable,
            "3.12.13",
            "Linux",
            "x86_64",
            (),
            gpu_arch="gfx1201",
            gpu_count=1,
        )
        patch.object(api, "inspect_environment", return_value=self.environment).start()
        self.cache = self.root / "cache"

    def resolve(self, args=(), environment=None):
        return resolver.resolve(
            str(self.model),
            list(args),
            environment or self.environment,
            profile="explicit",
            require_gpu=True,
        )

    def prepare(self, args=()):
        return api.prepare(self.resolve(args), cache_dir=self.cache, bundle=self.bundle)

    def test_shared_api_plan_roundtrip_and_exact_exec_vector(self):
        arguments = [
            "--port",
            "8000",
            "--port=8001",
            "--shutdown-timeout",
            "30",
            "--served-model-name",
            "name with spaces",
            "--override-generation-config",
            '{"stop":["a b"],"temperature":0.4}',
            "--api-key",
            "test-secret",
            "-tp",
            "1",
        ]
        plan = self.prepare(arguments)
        self.assertTrue(plan.resolution.model.verified)
        self.assertEqual(
            LaunchPlan.from_dict(json.loads(json.dumps(plan.as_dict()))), plan
        )
        self.assertNotIn("test-secret", json.dumps(plan.as_dict()))
        self.assertNotIn("test-secret", json.dumps(api.lock_record(plan)))
        with self.assertRaises(FrozenInstanceError):
            plan.offline = True
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(api.os, "execve") as execute,
        ):
            api.launch(plan, arguments, cache_dir=self.cache)
        executable, argv, env = execute.call_args.args
        self.assertEqual(executable, sys.executable)
        self.assertEqual(argv[-len(arguments) :], arguments)
        self.assertEqual(
            argv[:5],
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                str(self.model),
            ],
        )
        self.assertEqual(env["PAITON_ACTIVE_PLAN_SHA256"], plan.identity)
        self.assertEqual(env["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
        self.assertEqual(
            LaunchPlan.from_dict(
                json.loads(Path(env["PAITON_ACTIVE_PLAN"]).read_text())
            ),
            plan,
        )

    def test_short_and_original_cli_launch_the_identical_plan(self):
        from paiton_vllm_plugin.cli import main

        arguments = [
            "--port",
            "8123",
            "--served-model-name",
            "name with spaces",
            "--override-generation-config",
            '{"stop":["a b"]}',
        ]
        vectors = []
        for command in (["serve"], ["vllm", "serve"]):
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    inspection, "inspect_environment", return_value=self.environment
                ),
                patch.object(api.os, "execve") as execute,
                patch("sys.stderr", new_callable=io.StringIO),
            ):
                self.assertEqual(
                    main(
                        [
                            "--profile",
                            "explicit",
                            "--bundle",
                            str(self.bundle),
                            "--cache-dir",
                            str(self.cache),
                            *command,
                            str(self.model),
                            *arguments,
                        ]
                    ),
                    0,
                )
                vectors.append(execute.call_args.args)
        self.assertEqual(vectors[0], vectors[1])
        self.assertEqual(vectors[0][1][-len(arguments) :], arguments)

    def test_named_preset_reuses_local_checkpoint_and_rejects_conflicting_profile(self):
        from paiton_vllm_plugin.execution.presets import select, remember

        path = catalogue_module.DATA / "catalogue.json"
        data = json.loads(path.read_text())
        data["packages"][0]["presets"] = {"fixture-preset": "explicit"}
        path.write_text(json.dumps(data))
        self.assertEqual(
            select("fixture-preset", None), ("example/fixture", "explicit")
        )
        self.assertEqual(
            select("fixture-preset", None, self.model), (str(self.model), "explicit")
        )
        with patch(
            "paiton_vllm_plugin.execution.presets.Path.is_dir", return_value=True
        ):
            self.assertEqual(
                select("fixture-preset", "explicit"), ("fixture-preset", "explicit")
            )
        with self.assertRaisesRegex(PreparationError, "selects explicit"):
            select("fixture-preset", "something-else")
        with self.assertRaisesRegex(PreparationError, "requires a named preset"):
            select(str(self.model), None, self.model)
        self.assertEqual(select(str(self.model), None), (str(self.model), None))
        plan = self.prepare()
        remember("fixture-preset", plan, cache_dir=self.cache)
        self.assertEqual(
            select("fixture-preset", None, cache_dir=self.cache),
            (str(self.model), "explicit"),
        )
        binding = self.cache / "presets/fixture-preset.json"
        self.assertEqual(binding.stat().st_mode & 0o777, 0o600)
        binding.write_text("[]")
        with self.assertRaisesRegex(PreparationError, "Stored preset is invalid"):
            select("fixture-preset", None, cache_dir=self.cache)
        # Explicitly preparing again recovers a corrupt mutable binding; an
        # immutable launch lock remains subject to the normal lock checks.
        self.assertEqual(
            select("fixture-preset", None, self.model, cache_dir=self.cache),
            (str(self.model), "explicit"),
        )

    def test_compiled_overlay_preserves_weights_and_detects_changed_runtime(self):
        data = catalogue_module.DATA
        self.package["serving_layout"] = "overlay"
        (self.bundle / "config.json").write_text(
            '{"architectures":["CompiledFixture"]}'
        )
        self.manifest["files"]["config.json"] = receipt(self.bundle / "config.json")
        (self.bundle / "execution.json").write_text(json.dumps(self.manifest))
        (data / "fixture.json").write_text(json.dumps(self.manifest))
        cat = json.loads((data / "catalogue.json").read_text())
        cat["packages"] = [self.package]
        (data / "catalogue.json").write_text(json.dumps(cat))
        original_config = (self.model / "config.json").read_bytes()
        original_weight = inspection.fingerprint(self.model / "model.safetensors")
        plan = self.prepare()
        target = Path(plan.serving_model)
        self.assertEqual(
            (target / "config.json").read_bytes(),
            (self.bundle / "config.json").read_bytes(),
        )
        self.assertEqual((self.model / "config.json").read_bytes(), original_config)
        self.assertEqual(
            inspection.fingerprint(self.model / "model.safetensors"), original_weight
        )
        self.assertEqual(
            (target / "model.safetensors").resolve(), self.model / "model.safetensors"
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(api.os, "execve") as execute,
        ):
            api.launch(plan, [], cache_dir=self.cache)
        self.assertEqual(execute.call_args.args[1][4], str(target))
        from paiton_vllm_plugin.execution import serving

        with patch.object(serving, "sha256", wraps=serving.sha256) as checksum:
            reused = self.prepare()
        checksum.assert_not_called()
        self.assertEqual(reused.serving_files, plan.serving_files)
        runtime = target / "config.json"
        previous_stat = runtime.stat()
        runtime.write_bytes(runtime.read_bytes().replace(b"Compiled", b"Tampered"))
        os.utime(runtime, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
        with self.assertRaisesRegex(PreparationError, "checksum mismatch"):
            self.prepare()
        (target / "config.json").write_text("{}")
        with self.assertRaisesRegex(PreparationError, "Prepared model file changed"):
            api.validate_plan(plan)

    def test_original_gguf_identity_without_hf_config(self):
        root = self.root / "gguf"
        root.mkdir()
        (root / "model.gguf").write_bytes(b"GGUF-fixture")
        contract = {
            "repository": "example/gguf",
            "revision": "b" * 40,
            "storage_format": "GGUF",
            "identification": "gguf-files",
            "files": {"model.gguf": receipt(root / "model.gguf")},
        }
        result = inspection.inspect_model(root, contract, verify_weights=True)
        self.assertTrue(result.verified)
        (root / "model.gguf").write_bytes(b"BAD!-fixture")
        with self.assertRaisesRegex(PreparationError, "SHA256 mismatch"):
            inspection.inspect_model(root, contract, verify_weights=True)

    def test_direct_python_example_uses_same_resolver(self):
        path = Path(__file__).resolve().parents[1] / "examples/prepare_native.py"
        spec = importlib.util.spec_from_file_location("native_example", path)
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        with patch.object(
            example, "inspect_environment", return_value=self.environment
        ):
            plan = example.prepare_model(
                str(self.model), "explicit", self.bundle, self.cache, []
            )
        self.assertEqual(api.lock_record(plan), api.lock_record(self.prepare()))

    def test_explicit_environment_overrides_fail_before_preparation(self):
        with (
            patch.dict(os.environ, {"RADIANCE_UNREVIEWED_OVERRIDE": "1"}),
            patch.object(cache, "prepare_package") as prepare,
        ):
            with self.assertRaisesRegex(
                PreparationError, "Unqualified execution override"
            ):
                self.prepare()
        prepare.assert_not_called()

    def test_lock_conflicts_and_equivalent_numeric_flags(self):
        first = self.prepare()
        explicit = self.prepare(["-tp", "1", "--shutdown-timeout", "30"])
        self.assertEqual(api.lock_record(first), api.lock_record(explicit))
        lock = self.root / "lock.json"
        cache.write_json_immutable(lock, api.lock_record(first))
        api.check_lock(explicit, lock)
        api.check_lock(replace(first, offline=True), lock)
        with self.assertRaisesRegex(PreparationError, "Lockfile conflicts"):
            api.check_lock(self.prepare(["--max-model-len", "32"]), lock)

    def test_verified_weight_cache_invalidates_changed_bytes(self):
        first = self.prepare()
        with patch.object(
            api, "inspect_model", wraps=inspection.inspect_model
        ) as inspect:
            second = self.prepare()
        self.assertEqual(first.resolution.model, second.resolution.model)
        self.assertFalse(
            any(c.kwargs.get("verify_weights") for c in inspect.call_args_list)
        )
        weight = self.model / "model.safetensors"
        before = weight.stat()
        weight.write_bytes(b"changed!")
        os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(PreparationError, "Source file changed"):
            api.validate_plan(first)
        with self.assertRaisesRegex(PreparationError, "SHA256 mismatch"):
            self.prepare()

    def test_wrong_checkpoint_and_extra_loader_files(self):
        (self.model / "config.json").write_text('{"architectures":["Otherxx"]}')
        with self.assertRaisesRegex(PreparationError, "config is not"):
            self.resolve()
        (self.model / "config.json").write_text('{"architectures":["Fixture"]}')
        (self.model / "other.safetensors").write_bytes(b"weights!")
        with self.assertRaisesRegex(PreparationError, "Unqualified model"):
            self.resolve()

    def test_wrong_gpu_runtime_parallelism_and_unknown_settings(self):
        for env in (
            replace(self.environment, gpu_arch="gfx942"),
            replace(self.environment, gpu_count=2),
            replace(self.environment, python_version="3.11.9"),
            replace(self.environment, system="Windows"),
        ):
            with self.assertRaises(PreparationError):
                self.resolve(environment=env)
        for args in (
            ["-tp", "2"],
            ["--max-model-len", "65"],
            ["--enable-prefix-caching", "oops"],
            ["--speculative-config", "{}"],
            ["--dtype", "float16", "--enforce-eager"],
        ):
            with self.assertRaises(PreparationError):
                self.resolve(args)
        requirements = dict(
            self.package["runtime"], versions={"vllm": "qualified-build"}
        )
        with self.assertRaisesRegex(PreparationError, "not installed"):
            resolver.check_environment(self.environment, requirements, require_gpu=True)

    def test_yaml_cannot_change_between_validation_and_launch(self):
        config = self.root / "serving.yaml"
        config.write_text("max-model-len: 32\n")
        plan = self.prepare(["--config", str(config)])
        config.write_text("max-model-len: 64\n")
        with self.assertRaisesRegex(PreparationError, "configuration changed"):
            api.validate_plan(plan)

    def test_missing_hf_inputs_stay_offline_and_do_not_substitute(self):
        called = []

        def snapshot(**kwargs):
            called.append(kwargs)
            return str(self.model)

        with patch.dict(
            sys.modules,
            {"huggingface_hub": SimpleNamespace(snapshot_download=snapshot)},
        ):
            result = inspection.local_model(
                "example/fixture", self.package["model"], offline=True
            )
            resolver.resolve(
                "example/fixture",
                ["--download-dir", str(self.root / "existing hub")],
                self.environment,
                profile="explicit",
                offline=True,
            )
            with self.assertRaisesRegex(PreparationError, "revision differs"):
                resolver.resolve(
                    "example/fixture",
                    ["--revision", "wrong"],
                    self.environment,
                    profile="explicit",
                    offline=True,
                )
        self.assertEqual(result, self.model)
        self.assertTrue(called[0]["local_files_only"])
        self.assertEqual(called[0]["revision"], "a" * 40)
        self.assertNotIn("trust_remote_code", called[0])
        self.assertEqual(called[1]["cache_dir"], str(self.root / "existing hub"))
        self.assertEqual(len(called), 2)
        with self.assertRaises(PreparationError):
            inspection.local_model("example/other", self.package["model"], offline=True)

    def test_concurrent_preparation_only_installs_one_complete_package(self):
        original = shutil.copyfile
        copies = []

        def copy(*args, **kwargs):
            copies.append(args[0])
            return original(*args, **kwargs)

        with (
            patch.object(cache.shutil, "copyfile", side_effect=copy),
            ThreadPoolExecutor(max_workers=4) as pool,
        ):
            paths = list(
                pool.map(
                    lambda _: cache.prepare_package(
                        self.cache, self.package, source=self.bundle
                    ),
                    range(4),
                )
            )
        self.assertTrue(all(p == paths[0] for p in paths))
        self.assertEqual(len(copies), 2)
        cache.verify_package(paths[0], self.package)

    def test_interrupted_preparation_recovers_without_live_replacement(self):
        with patch.object(cache.shutil, "copyfile", side_effect=OSError("interrupted")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                cache.prepare_package(self.cache, self.package, source=self.bundle)
        self.assertFalse(cache.destination(self.cache, self.package).exists())
        self.assertFalse(list((self.cache / "executions").glob("*.partial-*")))
        recovered = cache.prepare_package(self.cache, self.package, source=self.bundle)
        self.assertEqual(recovered, cache.destination(self.cache, self.package))

    def test_unknown_catalogue_plan_and_execution_schemas(self):
        with self.assertRaises(PreparationError):
            LaunchPlan.from_dict({"schema": 99})
        p = catalogue_module.DATA / "catalogue.json"
        p.write_text('{"catalogue_schema":99}')
        with self.assertRaisesRegex(PreparationError, "catalogue schema"):
            catalogue_module.catalogue()

    def test_pinned_download_interruption_digest_and_offline_reuse(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for path in sorted(self.bundle.iterdir()):
                archive.add(path, arcname=path.name, recursive=False)
        content = buffer.getvalue()
        remote = {
            "url": "https://release.invalid/pinned.tar",
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        package = dict(self.package, archive=remote)

        class Response(io.BytesIO):
            def geturl(self):
                return remote["url"]

        class Interrupted(Response):
            def read(self, *args):
                raise ConnectionError("interrupted download")

        with patch.object(
            cache.urllib.request, "urlopen", return_value=Interrupted(content)
        ):
            with self.assertRaisesRegex(ConnectionError, "interrupted"):
                cache.prepare_package(self.cache, package)
        self.assertFalse(cache.destination(self.cache, package).exists())
        with patch.object(cache.urllib.request, "urlopen") as download:
            with self.assertRaisesRegex(PreparationError, "Offline mode"):
                cache.prepare_package(self.cache, package, offline=True)
        download.assert_not_called()
        wrong = dict(package, archive=dict(remote, sha256="0" * 64))
        with patch.object(
            cache.urllib.request, "urlopen", return_value=Response(content)
        ):
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                cache.prepare_package(self.cache, wrong)
        with patch.object(
            cache.urllib.request, "urlopen", return_value=Response(content)
        ):
            result = cache.prepare_package(self.cache, package)
        with patch.object(cache.urllib.request, "urlopen") as download:
            self.assertEqual(
                cache.prepare_package(self.cache, package, offline=True), result
            )
        download.assert_not_called()
        cache.verify_package(result, package)

    def test_worker_initialization_repeat_late_activation_and_hot_switch(self):
        from paiton_vllm_plugin.execution import bootstrap

        self.package["adapter"] = "minicpm5-awq-v1"
        path = catalogue_module.DATA / "catalogue.json"
        cat = json.loads(path.read_text())
        cat["packages"] = [self.package]
        path.write_text(json.dumps(cat))
        plan = self.prepare()
        path = self.root / "plan.json"
        path.write_text(json.dumps(plan.as_dict()))
        env = dict(
            plan.environment,
            PAITON_ACTIVE_PLAN=str(path),
            PAITON_ACTIVE_PLAN_SHA256=plan.identity,
        )
        with (
            patch.object(bootstrap, "_plan", None),
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules),
        ):
            sys.modules.pop("vllm", None)
            bootstrap.initialize()
            bootstrap.initialize()
            self.assertEqual(bootstrap._plan, plan)
            changed = replace(plan, offline=True)
            path.write_text(json.dumps(changed.as_dict()))
            os.environ["PAITON_ACTIVE_PLAN_SHA256"] = changed.identity
            with self.assertRaisesRegex(PreparationError, "cannot be changed"):
                bootstrap.initialize()
        path.write_text(json.dumps(plan.as_dict()))
        with (
            patch.object(bootstrap, "_plan", None),
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"vllm": SimpleNamespace()}),
        ):
            with self.assertRaisesRegex(PreparationError, "fresh process"):
                bootstrap.initialize()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
        with (
            patch.object(bootstrap, "_plan", None),
            patch.dict(os.environ, env, clear=True),
        ):
            with self.assertRaisesRegex(PreparationError, "Worker environment differs"):
                bootstrap.initialize()


class WitnessTests(unittest.TestCase):
    def test_warmup_and_capture_do_not_count_as_request_execution(self):
        class Function:
            def __init__(self, call):
                self.call = call

            def __call__(self, *args):
                return self.call(*args)

        capture = [1]
        completed = threading.Event()
        reports = []
        library = SimpleNamespace(
            hipStreamIsCapturing=Function(
                lambda stream, out: setattr(out._obj, "value", capture[0]) or 0
            ),
            hipEventCreateWithFlags=Function(
                lambda out, flags: setattr(out._obj, "value", 1) or 0
            ),
            hipEventRecord=Function(lambda *args: 0),
            hipEventQuery=Function(lambda *args: 0),
            hipEventDestroy=Function(lambda *args: 0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "native/rms-decode").mkdir(parents=True)
            (root / "native/rms-decode/manifest.json").write_text(
                '{"sha256":"expected"}'
            )

            def report(stage, **details):
                reports.append((stage, details))
                completed.set()

            witness = Witness(
                SimpleNamespace(), SimpleNamespace(bundle=tmp, identity="plan"), report
            )
            witness.observe(library, "rms-decode", 0)
            self.assertFalse(witness.recorded)
            witness.in_request = True
            witness.observe(library, "rms-decode", 0)
            self.assertFalse(witness.recorded)
            capture[0] = 0
            witness.observe(library, "rms-decode", 0)
            self.assertTrue(completed.wait(2))
            witness.observe(library, "rms-decode", 0)
            self.assertEqual(len(reports), 1)
            self.assertEqual(reports[0][0], "Executed")
            self.assertEqual(reports[0][1]["phase"], "request")


if __name__ == "__main__":
    unittest.main()
