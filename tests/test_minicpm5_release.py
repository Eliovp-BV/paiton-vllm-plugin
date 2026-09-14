import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('minicpm_server', ROOT/'models/MiniCPM5-2B/server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class MiniCPMPreparation(unittest.TestCase):
    def test_receipt_reuse_and_changed_file_verification(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            model = root/'snapshot'; model.mkdir()
            source = model/'config.json'; source.write_bytes(b'abc')
            lock = dict(model='example/test', revision='a'*40, files=[dict(
                name=source.name, bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())])
            hub = types.SimpleNamespace(snapshot_download=lambda *args, **kwargs: str(model))
            with patch.object(server, 'LOCK', lock), patch.dict(sys.modules, huggingface_hub=hub), \
                    patch.dict('os.environ', XDG_CACHE_HOME=str(root/'cache')):
                self.assertEqual(server.prepare(offline=True), model)
                with patch.object(hashlib, 'file_digest', side_effect=AssertionError('Unchanged files should reuse receipts')):
                    server.prepare(offline=True)
                source.write_bytes(b'xyz')
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    server.prepare(offline=True)

    def test_stock_and_paiton_preserve_serving_configuration(self):
        stock = server.command('/checkpoint', stock=True)
        paiton = server.command('/checkpoint')
        self.assertEqual(paiton[:len(stock)], stock)
        self.assertEqual(paiton[len(stock)], '--hf-overrides')
        self.assertEqual(json.loads(paiton[-1])['architectures'], ['PaitonMiniCPM5AWQForCausalLM'])


if __name__ == '__main__':
    unittest.main()
