import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from paiton_vllm_plugin.activation import activate, mode
from paiton_vllm_plugin.execution.arguments import inspect_arguments, inspect_tokens
from paiton_vllm_plugin.execution.cache import (
    extract_archive,
    prepare_package,
    write_json_immutable,
)
from paiton_vllm_plugin.execution.contracts import PreparationError, digest
from paiton_vllm_plugin.artifact_manifest import (
    ArtifactCompatibilityError,
    load_and_validate_artifact_manifest,
    verify_file,
)


class NativeActivationTests(unittest.TestCase):
    def test_inactive_discovery_ignores_arch_and_allowlist(self):
        code = """import sys
from paiton_vllm_plugin import paiton_platform_plugin,register_paiton_models
assert paiton_platform_plugin() is None
register_paiton_models()
assert not any(x.split('.')[0] in {'torch','triton','vllm','paiton_runtime_compat'} for x in sys.modules)
"""
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("PAITON_", "VLLM_", "RADIANCE_"))
        }
        env.update(
            PAITON_GPU_ARCH="gfx1201",
            VLLM_PLUGINS="paiton_platform,register_paiton_models",
        )
        subprocess.run([sys.executable, "-c", code], env=env, check=True)

    def test_allowlist_is_never_extended(self):
        env = {"VLLM_PLUGINS": "other"}
        with self.assertRaisesRegex(ValueError, "excludes"):
            activate("native", env)
        self.assertEqual(env, {"VLLM_PLUGINS": "other"})
        env = {"VLLM_PLUGINS": "other,register_paiton_models"}
        activate("native", env)
        self.assertEqual(env["VLLM_PLUGINS"], "other,register_paiton_models")

    def test_conflicting_switches_rejected(self):
        for env in (
            {"PAITON_PLUGIN_MODE": "off", "VLLM_USE_PAITON_PLATFORM": "1"},
            {"VLLM_DISABLE_PAITON_PLATFORM": "1", "VLLM_USE_PAITON_PLATFORM": "1"},
            {"PAITON_PLUGIN_MODE": "invalid"},
        ):
            with self.assertRaises(ValueError):
                mode(env)

    def test_help_and_doctor_without_frameworks(self):
        for args in [["--help"], ["doctor"], ["models"]]:
            result = subprocess.run(
                [sys.executable, "-m", "paiton_vllm_plugin.cli", *args],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_failed_process_start_cannot_continue_with_stock_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(
                os.environ,
                PAITON_ACTIVE_PLAN=str(Path(tmp) / "absent-plan.json"),
                PAITON_ACTIVE_PLAN_SHA256="0" * 64,
                PAITON_PLUGIN_MODE="native",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    'import site; import paiton_vllm_plugin.startup; site.execsitecustomize(); print("STOCK")',
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 78)
            self.assertNotIn("STOCK", result.stdout)
            report = json.loads(result.stderr)
            self.assertEqual(report["phase"], "process-start")
            self.assertFalse(report["stock_fallback"])


class ArgumentsTests(unittest.TestCase):
    def test_token_fidelity_and_last_option(self):
        args = [
            "--port",
            "8001",
            "--port=8002",
            "--served-model-name",
            "name with spaces",
            "alias",
            "--override-generation-config",
            '{"temperature":0.4,"stop":["a b"]}',
            "--compilation-config",
            '{"cudagraph_capture_sizes":[1,2]}',
        ]
        before = list(args)
        values = inspect_tokens(args)
        self.assertEqual(args, before)
        self.assertEqual(values["port"], "8002")
        self.assertEqual(values["override-generation-config"]["stop"], ["a b"])

    def test_yaml_cli_precedence_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "tensor-parallel-size: 2\nport: 9000\nenable-prefix-caching: false\ncompilation-config:\n  cudagraph_capture_sizes: [1, 2]\n"
            )
            args = ["--config", str(path), "-tp", "1", "--port", "8000"]
            effective, files = inspect_arguments(args)
            self.assertEqual(effective["tensor-parallel-size"], "1")
            self.assertFalse(effective["enable-prefix-caching"])
            self.assertEqual(files[0][1], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_unsafe_or_uninterpretable_configuration(self):
        for args in [
            ["--config=x.yaml"],
            ["--trust-remote-code"],
            ["--hf-overrides", "{}"],
            ["--speculative-config", "{bad}"],
            ["--new-unknown-flag", "1"],
        ]:
            with self.assertRaises(PreparationError):
                inspect_arguments(args)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.yaml"
            p.write_text("port: 1\nport: 2\n")
            with self.assertRaises(PreparationError):
                inspect_arguments(["--config", str(p)])


class ArtifactTests(unittest.TestCase):
    def test_first_read_may_update_atime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.so"
            path.write_bytes(b"payload")
            os.utime(path, ns=(1, path.stat().st_mtime_ns))
            verify_file(
                path,
                {"size_bytes": 7, "sha256": hashlib.sha256(b"payload").hexdigest()},
            )

    def legacy(self, root):
        lib = root / "native.so"
        lib.write_bytes(b"abcd")
        m = {
            "manifest_version": 1,
            "binary_abi_version": 1,
            "capability_version": 1,
            "target": {"arch": "gfx1201", "wave_size": 32},
            "parallelism": {"tp_size": 1},
            "artifact": {
                "filename": lib.name,
                "size_bytes": 4,
                "sha256": hashlib.sha256(b"abcd").hexdigest(),
            },
        }
        p = lib.with_suffix(".manifest.json")
        p.write_text(json.dumps(m))
        return lib, p, m

    def test_stale_digest_and_same_size_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib, p, m = self.legacy(Path(tmp))
            load_and_validate_artifact_manifest(
                lib, expected_arch="gfx1201", expected_tp_size=1
            )
            m["artifact"]["sha256"] = "0" * 64
            p.write_text(json.dumps(m))
            with self.assertRaises(ArtifactCompatibilityError):
                load_and_validate_artifact_manifest(
                    lib, expected_arch="gfx1201", expected_tp_size=1
                )
            m["artifact"]["sha256"] = hashlib.sha256(b"abcd").hexdigest()
            p.write_text(json.dumps(m))
            stat = lib.stat()
            lib.write_bytes(b"WXYZ")
            os.utime(lib, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            with self.assertRaises(ArtifactCompatibilityError):
                load_and_validate_artifact_manifest(
                    lib, expected_arch="gfx1201", expected_tp_size=1
                )

    def test_abi_wave_arch_tp_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib, p, m = self.legacy(Path(tmp))
            for key, value in [("binary_abi_version", 2), ("manifest_version", 99)]:
                changed = dict(m)
                changed[key] = value
                p.write_text(json.dumps(changed))
                with self.assertRaises(ArtifactCompatibilityError):
                    load_and_validate_artifact_manifest(
                        lib, expected_arch="gfx1201", expected_tp_size=1
                    )
            p.write_text(json.dumps(m))
            for arch, tp in [("gfx942", 1), ("gfx1201", 2)]:
                with self.assertRaises(ArtifactCompatibilityError):
                    load_and_validate_artifact_manifest(
                        lib, expected_arch=arch, expected_tp_size=tp
                    )
            m["target"]["wave_size"] = 64
            p.write_text(json.dumps(m))
            with self.assertRaises(ArtifactCompatibilityError):
                load_and_validate_artifact_manifest(
                    lib, expected_arch="gfx1201", expected_tp_size=1
                )

    def test_archive_traversal_and_links(self):
        import io

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, kind in [
                ("../escape", tarfile.REGTYPE),
                ("/absolute", tarfile.REGTYPE),
                ("a", tarfile.SYMTYPE),
                ("a", tarfile.LNKTYPE),
            ]:
                arc = root / "bad.tar"
                with tarfile.open(arc, "w") as out:
                    item = tarfile.TarInfo(name)
                    item.type = kind
                    item.linkname = "/etc/passwd"
                    item.size = 0
                    out.addfile(item, io.BytesIO())
                with self.assertRaises(PreparationError):
                    extract_archive(arc, root, {"files": {}})

    def test_immutable_lock_and_no_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            write_json_immutable(path, {"schema": 1, "immutable": ("a", "b")})
            write_json_immutable(path, {"schema": 1, "immutable": ("a", "b")})
            with self.assertRaises(PreparationError):
                write_json_immutable(path, {"schema": 2})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_atomic_preparation_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "kernel.so").write_bytes(b"native")
            m = {
                "execution_schema": 1,
                "binary_abi_version": 1,
                "target": {"arch": "gfx1201", "wave_size": 32},
                "tp_size": 1,
                "files": {
                    "kernel.so": {
                        "size_bytes": 6,
                        "sha256": hashlib.sha256(b"native").hexdigest(),
                    }
                },
            }
            (source / "execution.json").write_text(json.dumps(m))
            package = {"runtime": {"gpu_arch": "gfx1201"}, "archive": None}
            with (
                patch("paiton_vllm_plugin.execution.cache.manifest", return_value=m),
                patch(
                    "paiton_vllm_plugin.execution.cache.package_digest",
                    return_value=digest(m),
                ),
            ):
                with self.assertRaisesRegex(PreparationError, "unpublished"):
                    prepare_package(root / "cache", package, offline=True)
                target = prepare_package(
                    root / "cache", package, source=source, offline=True
                )
                self.assertEqual(
                    prepare_package(root / "cache", package, offline=True), target
                )
                (target / "kernel.so").write_bytes(b"broken")
                with self.assertRaisesRegex(PreparationError, "corrupt"):
                    prepare_package(root / "cache", package, source=source)
                self.assertEqual((target / "kernel.so").read_bytes(), b"broken")


if __name__ == "__main__":
    unittest.main()
