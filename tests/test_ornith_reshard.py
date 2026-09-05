import json
from pathlib import Path
import struct
import tempfile
import unittest

from paiton_vllm_plugin.ornith_reshard import reshard_checkpoint


class OrnithReshardTests(unittest.TestCase):
    def test_payloads_are_split_without_numeric_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.safetensors"
            specs = {
                "a": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
                "b": {"dtype": "U8", "shape": [5], "data_offsets": [3, 8]},
                "c": {"dtype": "U8", "shape": [4], "data_offsets": [8, 12]},
            }
            header = json.dumps(specs, separators=(",", ":")).encode()
            header += b" " * ((8 - len(header) % 8) % 8)
            payload = bytes(range(12))
            source.write_bytes(struct.pack("<Q", len(header)) + header + payload)

            output = root / "shards"
            report = reshard_checkpoint(source, output, max_shard_bytes=8)

            self.assertEqual(report["tensor_count"], 3)
            self.assertEqual(report["logical_payload_bytes"], len(payload))
            self.assertEqual(len(report["shards"]), 2)
            index = json.loads((output / "model.safetensors.index.json").read_text())
            self.assertEqual(
                index["weight_map"]["a"], "model-00001-of-00002.safetensors"
            )
            self.assertEqual(
                index["weight_map"]["b"], "model-00001-of-00002.safetensors"
            )
            self.assertEqual(
                index["weight_map"]["c"], "model-00002-of-00002.safetensors"
            )

            recovered = bytearray()
            for record in report["shards"]:
                raw = (output / record["filename"]).read_bytes()
                header_size = struct.unpack("<Q", raw[:8])[0]
                recovered.extend(raw[8 + header_size :])
            self.assertEqual(bytes(recovered), payload)

    def test_rejects_nonempty_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"not used")
            output = root / "output"
            output.mkdir()
            (output / "keep").write_text("x")
            with self.assertRaisesRegex(FileExistsError, "non-empty"):
                reshard_checkpoint(source, output)


if __name__ == "__main__":
    unittest.main()
