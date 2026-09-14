import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from paiton_vllm_plugin.gguf_release_server import (
    ReleaseModelError, serving_command, verify_payload,
)


class GGUFReleaseServerTest(unittest.TestCase):
    def make_bundle(self, directory):
        config = dict(architectures=['PaitonQwen38GGUFForCausalLM'],
                      paiton_qwen38_contract={'version':5})
        raw = json.dumps(config).encode()
        (directory/'config.json').write_bytes(raw)
        manifest = dict(bundle_version=1, files=[dict(path='config.json',
            size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())])
        (directory/'paiton-release.manifest.json').write_text(json.dumps(manifest))
        (directory/'paiton-release.spdx.json').write_text('{}')
        (directory/'SHA256SUMS').write_text('')
        return manifest

    def test_checks_hashes_and_rejects_extra_or_linked_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_bundle(directory)
            self.assertEqual(verify_payload(directory), {'version':5})
            (directory/'private.cpp').write_text('unexpected')
            with self.assertRaisesRegex(ReleaseModelError, 'inventory'):
                verify_payload(directory)
            (directory/'private.cpp').unlink()
            (directory/'config.json').write_text('tampered')
            with self.assertRaises(ReleaseModelError):
                verify_payload(directory)
            self.make_bundle(directory)
            (directory/'config.json').rename(directory/'other.json')
            (directory/'config.json').symlink_to('other.json')
            with self.assertRaisesRegex(ReleaseModelError, 'symlink'):
                verify_payload(directory)

    def test_rejects_parent_paths_before_reading(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.make_bundle(directory)
            manifest['files'][0]['path'] = '../config.json'
            (directory/'paiton-release.manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ReleaseModelError, 'path'):
                verify_payload(directory)

    def test_uses_vllm_native_loader_and_qualified_request_limits(self):
        command = serving_command(Path('/model'), '127.0.0.1', 8123)
        self.assertEqual(command[:3], ['vllm','serve','/model'])
        for flag,value in [('--load-format','paiton_gguf'),('--max-num-seqs','1'),
                           ('--max-model-len','8192'),('--max-num-batched-tokens','512')]:
            self.assertEqual(command[command.index(flag)+1], value)
        self.assertIn('--no-enable-prefix-caching', command)
        self.assertNotIn('--speculative-config', command)


if __name__ == '__main__':
    unittest.main()
