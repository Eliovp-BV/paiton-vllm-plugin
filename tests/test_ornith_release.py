import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paiton_vllm_plugin import ornith_release_server as release


class OrnithReleaseTests(unittest.TestCase):
    def test_server_command_uses_exact_dflash_contract(self):
        with patch.dict(os.environ, {"PAITON_ORNITH_DFLASH": "1"}, clear=False):
            with patch.object(release, "_select_model", return_value=Path("/model")):
                command = release.build_server_command(["--port", "9000"])
        self.assertEqual(command[:3], ["vllm", "serve", "/model"])
        self.assertIn("--enforce-eager", command)
        spec = json.loads(command[command.index("--speculative-config") + 1])
        self.assertEqual(spec["method"], "dflash")
        self.assertEqual(spec["num_speculative_tokens"], 16)
        self.assertEqual(spec["model"], "/model/dflash-draft")
        self.assertEqual(command[-2:], ["--port", "9000"])

    def test_server_command_can_disable_dflash_strictly(self):
        with patch.object(release, "_select_model", return_value=Path("/model")):
            with patch.dict(
                os.environ, {"PAITON_ORNITH_DFLASH": "0"}, clear=False
            ):
                command = release.build_server_command()
            self.assertNotIn("--speculative-config", command)

            with patch.dict(
                os.environ, {"PAITON_ORNITH_DFLASH": "false"}, clear=False
            ):
                with self.assertRaisesRegex(release.ReleaseModelError, "exactly 0 or 1"):
                    release.build_server_command()

    def test_explicit_model_skips_download_and_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"PAITON_MODEL": directory}, clear=False):
                with patch.object(release, "_resolve_base_model") as resolve:
                    self.assertEqual(release._select_model(), Path(directory).resolve())
                resolve.assert_not_called()

    def test_existing_invalid_stage_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory) / release.BASE_REVISION
            stage.mkdir()
            with patch.dict(
                os.environ, {"PAITON_ORNITH_CACHE": directory}, clear=False
            ):
                with self.assertRaisesRegex(release.ReleaseModelError, "incomplete"):
                    release._stage_model(Path("/base"), Path("/overlay"))

    def test_cached_stage_rejects_modified_runtime_file(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            marker = {
                "base_model": release.BASE_MODEL,
                "revision": release.BASE_REVISION,
                "checkpoint_sha256": release.CHECKPOINT_SHA256,
            }
            (stage / "paiton-ornith-release.json").write_text(json.dumps(marker))
            with patch.object(release, "sha256_file", return_value="wrong"):
                self.assertFalse(release._validate_existing_stage(stage))


if __name__ == "__main__":
    unittest.main()
