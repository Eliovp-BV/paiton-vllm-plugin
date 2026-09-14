import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paiton_vllm_plugin import qwen38_release_server as release


class Qwen38ReleaseTests(unittest.TestCase):
    def test_checked_metadata_matches_runtime_contract(self):
        metadata = Path(__file__).parents[1] / "models/Qwen3.8/paiton-release.json"
        self.assertEqual(metadata.stat().st_size, release.RELEASE_METADATA_SIZE)
        self.assertEqual(release._sha256_file(metadata), release.RELEASE_METADATA_SHA256)
        parsed = release._validate_release_metadata(metadata)
        self.assertEqual(parsed["schema_version"], 2)
        self.assertEqual(set(parsed["files"]), set(release.EXPECTED_OVERLAY_FILES))
        self.assertIn("decode-only INT3 MLP shadows", parsed["contract"]["quantization"])

    def test_default_command_enables_qualified_graph_and_w4_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / release.LM_HEAD_ARTIFACT_NAME).touch()
            with patch.object(release, "_select_model", return_value=(str(model), None)):
                with patch.dict(os.environ, {}, clear=True):
                    command = release.build_server_command()
                    self.assertIn("--no-enforce-eager", command)
                    self.assertIn("-O2", command)
                    self.assertNotIn("--enforce-eager", command)
                    config = json.loads(command[command.index("--compilation-config") + 1])
                    self.assertEqual(config["cudagraph_capture_sizes"], [1])
                    self.assertEqual(config["max_cudagraph_capture_size"], 1)
                    self.assertEqual(config["cudagraph_num_of_warmups"], 1)
                    self.assertEqual(
                        os.environ["PAITON_QWEN38_W4_LM_HEAD_SO"],
                        str(model / release.LM_HEAD_ARTIFACT_NAME),
                    )
                    self.assertEqual(
                        os.environ["PAITON_QWEN38_W4_LM_HEAD_SHA256"],
                        release.EXPECTED_OVERLAY_FILES["lm_head_artifact"]["sha256"],
                    )
                    for name in (
                        "PAITON_QWEN38_W4_LM_HEAD",
                        "PAITON_QWEN38_SERIALIZED_EXTERNAL_GRAPH_CAPTURE",
                        "PAITON_DECODE_ATTENTION_AOT",
                        "PAITON_PREFILL_ATTENTION_AOT",
                    ):
                        self.assertEqual(os.environ[name], "1")


if __name__ == "__main__":
    unittest.main()
