import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "inventory", Path(__file__).parents[1] / "paiton_vllm_plugin/gguf_inventory.py"
)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def fixture(*, kind=12, dim=256, offset=0, duplicate=False):
    metadata = string("general.architecture") + struct.pack("<I", 8) + string("qwen35")
    record = string("weight") + struct.pack("<IQIQ", 1, dim, kind, offset)
    data = b"GGUF" + struct.pack("<IQQ", 3, 2 if duplicate else 1, 1) + metadata + record
    if duplicate:
        data += record
    data += b"\0" * (-len(data) % 32)
    return data, data + b"\0" * 144


class InventoryTest(unittest.TestCase):
    def inspect(self, data, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.gguf"
            path.write_bytes(data)
            return inventory.inspect_gguf(path, **kwargs)

    def test_header_and_complete_agree(self):
        header, complete = fixture()
        a = self.inspect(header, file_size=len(complete))
        b = self.inspect(complete)
        self.assertTrue(a.pop("header_only"))
        self.assertFalse(b.pop("header_only"))
        self.assertEqual(a, b)
        self.assertEqual(a["tensors"][0]["size_bytes"], 144)
        self.assertEqual(a["tensors"][0]["offset"], len(header))
        json.dumps(a)

    def test_rejects_truncation_unknown_format_misalignment_and_duplicates(self):
        for kwargs in ({"kind": 999}, {"dim": 255}, {"offset": 1},
                       {"duplicate": True}, {"offset": 1024}):
            with self.subTest(kwargs=kwargs), self.assertRaises(inventory.InventoryError):
                self.inspect(fixture(**kwargs)[1])
        header, complete = fixture()
        for data in (header[:-1], complete[:-1], b"FUGG" + complete[4:]):
            with self.assertRaises(inventory.InventoryError):
                self.inspect(data)

    def test_metadata_budget(self):
        with self.assertRaises(inventory.InventoryError):
            self.inspect(fixture()[1], metadata_budget=32)


if __name__ == "__main__":
    unittest.main()
