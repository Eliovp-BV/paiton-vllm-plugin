import hashlib
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


root = Path(__file__).parents[1] / "paiton_vllm_plugin"
package = types.ModuleType("gguf_fixture")
package.__path__ = [str(root)]
sys.modules[package.__name__] = package
for module in ("gguf_inventory", "gguf_source"):
    spec = importlib.util.spec_from_file_location(f"gguf_fixture.{module}", root / (module + ".py"))
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
source = sys.modules["gguf_fixture.gguf_source"]


def fixture():
    name = b"weight"
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    header += struct.pack("<Q", len(name)) + name + struct.pack("<IQQIQ", 2, 256, 2, 12, 0)
    header += b"\0" * (-len(header) % 32)
    return header + bytes(range(144)) + bytes(reversed(range(144)))


class SourceTest(unittest.TestCase):
    def test_exact_bounded_rows_and_rejection(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.gguf"
            path.write_bytes(data)
            args = dict(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data), max_read_bytes=144)
            with source.GGUFTensorSource(path, **args) as reader:
                self.assertEqual(reader.read_rows("weight", 1, 1), bytes(reversed(range(144))))
                self.assertEqual(reader.read_rows("weight", 0, 1), bytes(range(144)))
                for start, count in ((0, 2), (-1, 1), (1, 2), (0, 0)):
                    with self.assertRaises(source.InventoryError):
                        reader.read_rows("weight", start, count)
            self.assertTrue(reader.source.closed)
            with self.assertRaises(source.InventoryError):
                source.GGUFTensorSource(path, **dict(args, sha256="0" * 64))

    def test_cache_symlink_replacement_keeps_verified_descriptor(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            original.write_bytes(data)
            link = root / "cached.gguf"
            link.symlink_to(original)
            inspect = source.inspect_gguf

            def replace_then_inspect(path):
                link.unlink()
                link.write_bytes(b"corrupt replacement")
                return inspect(path)

            with patch.object(source, "inspect_gguf", replace_then_inspect):
                with source.GGUFTensorSource(link, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)) as reader:
                    self.assertEqual(reader.read_rows("weight", 0, 1), bytes(range(144)))


if __name__ == "__main__":
    unittest.main()
