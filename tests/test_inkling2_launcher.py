import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest import mock

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
        self.assertEqual(config['num_speculative_tokens_per_batch_size'], [[1, 1, 5], [2, 4, 4]])
        self.assertNotIn('--speculative-config', server.command('/checkpoint', drafter=False))

    def test_launcher_and_image_agree(self):
        script, docker = (FOLDER / 'serve-docker.sh').read_text(), (FOLDER / 'Dockerfile').read_text()
        port = server.command('/checkpoint')[server.command('/checkpoint').index('--port') + 1]
        self.assertIn(f':{port}"', script)
        self.assertIn(f'EXPOSE {port}', docker)
        self.assertIn('GPU_MAX_HW_QUEUES=1', script)
        self.assertIn(json.loads((FOLDER / 'checkpoint.lock.json').read_text())['revision'], docker)

    def environment(self, *arguments, **preset):
        """The environment the launcher hands to vLLM for `arguments`, with `preset` already set by the user."""
        with mock.patch.dict(os.environ, preset), mock.patch.object(server, 'prepare', return_value='/checkpoint'), \
                mock.patch.object(server.os, 'execv') as execv, \
                mock.patch.object(server.sys, 'argv', ['server.py', *arguments]):
            if 'PAITON_INK2_KV' not in preset:
                os.environ.pop('PAITON_INK2_KV', None)
            server.main()
            execv.assert_called_once()
            return dict(os.environ)

    def test_one_constant_sets_the_cache_format(self):
        self.assertIn(server.KV_CACHE, ('bf16', 'fp8', 'kv4'))
        self.assertEqual(self.environment()['PAITON_INK2_KV'], server.KV_CACHE)
        for other in ('bf16', 'fp8', 'kv4'):
            with mock.patch.object(server, 'KV_CACHE', other):
                self.assertEqual(self.environment()['PAITON_INK2_KV'], other)

    def test_every_cache_format_can_be_asked_for(self):
        for name in ('bf16', 'fp8', 'kv4'):
            self.assertEqual(self.environment('--kv-cache', name)['PAITON_INK2_KV'], name)
            self.assertEqual(self.environment(PAITON_INK2_KV=name)['PAITON_INK2_KV'], name)
            # the option wins over the environment
            self.assertEqual(self.environment('--kv-cache', name, PAITON_INK2_KV='bf16')['PAITON_INK2_KV'], name)
        with self.assertRaises(SystemExit), mock.patch.object(server.sys, 'stderr'):
            self.environment('--kv-cache', 'int4')


if __name__ == '__main__':
    unittest.main()
