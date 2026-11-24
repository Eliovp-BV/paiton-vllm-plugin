import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'models/Inkling-2-Mini'
spec = importlib.util.spec_from_file_location('inkling2_server', FOLDER / 'server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class Inkling2Launcher(unittest.TestCase):
    def test_lock_pins_one_revision_and_the_weights(self):
        lock = json.loads((FOLDER / 'checkpoint.lock.json').read_text())
        self.assertEqual(len(lock['revision']), 40)
        names = {item['name'] for item in lock['files']}
        self.assertLessEqual({'config.json', 'model-00001-of-00001.safetensors', 'mtp.safetensors'}, names)
        self.assertTrue(all(len(item['sha256']) == 64 and item['bytes'] > 0 for item in lock['files']))

    def test_user_arguments_come_last_so_they_override(self):
        argv = server.command('/checkpoint', '127.0.0.1', ['--max-num-seqs', '1'])
        self.assertEqual(argv[-2:], ['--max-num-seqs', '1'])
        self.assertEqual(argv[argv.index('--host') + 1], '127.0.0.1')
        self.assertIn('--enforce-eager', argv)
        self.assertEqual(argv[argv.index('--quantization') + 1], 'inkling2')

    def test_drafter_is_on_unless_switched_off(self):
        argv = server.command('/checkpoint')
        config = json.loads(argv[argv.index('--speculative-config') + 1])
        self.assertEqual((config['method'], config['num_speculative_tokens']), ('dspark', 7))
        self.assertNotIn('--speculative-config', server.command('/checkpoint', drafter=False))

    def test_launcher_and_image_agree(self):
        script, docker = (FOLDER / 'serve-docker.sh').read_text(), (FOLDER / 'Dockerfile').read_text()
        port = server.command('/checkpoint')[server.command('/checkpoint').index('--port') + 1]
        self.assertIn(f':{port}"', script)
        self.assertIn(f'EXPOSE {port}', docker)
        self.assertIn('GPU_MAX_HW_QUEUES=1', script)
        self.assertIn(json.loads((FOLDER / 'checkpoint.lock.json').read_text())['revision'], docker)


if __name__ == '__main__':
    unittest.main()
