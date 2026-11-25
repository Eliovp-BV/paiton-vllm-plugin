from collections import namedtuple
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

Usage = namedtuple('Usage', 'total used free')

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


class Inkling2CacheTier(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patches = [mock.patch.dict(os.environ, {'PAITON_HOST_PIN_LIMIT_GIB': '64', 'PAITON_INK2_KV': 'bf16'}),
                   mock.patch.object(server, 'DISK_ROOT', Path(self.tmp.name) / 'kv-disk'),
                   mock.patch.object(server.shutil, 'disk_usage', lambda p: Usage(2 ** 40, 0, 2 ** 40))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)
        os.environ.pop('PAITON_HOST_KV_PRIVATE', None)

    def test_ram_tier_is_private_pinned_memory(self):
        args, _ = server.tier_settings(16.0, None)
        self.assertEqual(args, ['--kv-offloading-size', '16', '--kv-offloading-backend', 'native'])
        self.assertEqual(os.environ['PAITON_HOST_KV_PRIVATE'], '1')
        self.assertEqual(server.tier_settings(None, 'auto')[0][1], '64')   # auto: all the host can pin

    def test_refusals(self):
        for host, extend in ((65.0, None), (8.0, 'ram'), (2.0, None)):
            with self.subTest(host=host, extend=extend), self.assertRaises(SystemExit):
                server.tier_settings(host, extend)

    def test_disk_tier_and_its_folder_key(self):
        args, _ = server.tier_settings(None, 'disk')
        self.assertEqual(args[:4], ['--kv-offloading-size', f'{server.STAGING_GIB:g}', '--kv-offloading-backend', 'native'])
        self.assertEqual(args[args.index('--prefix-caching-hash-algo') + 1], 'sha256')
        config = json.loads(args[args.index('--kv-transfer-config') + 1])
        self.assertEqual(config['kv_connector_extra_config']['spec_name'], 'TieringOffloadingSpec')
        self.assertEqual(config['kv_connector_extra_config']['blocks_per_chunk'], 16)   # 9 MiB files: 4x faster restores
        folder = config['kv_connector_extra_config']['secondary_tiers'][0]['root_dir']
        self.assertEqual(folder, str(server.disk_folder()))
        with mock.patch.dict(os.environ, {'PAITON_INK2_PROJ': 'int8'}):   # other cached values, other folder
            self.assertNotEqual(server.disk_folder(), Path(folder))
        with mock.patch.dict(os.environ, {'PAITON_HOST_PIN_LIMIT_GIB': '12'}):  # auto: under 1.25 x the pool
            self.assertIn('--kv-transfer-config', server.tier_settings(None, 'auto')[0])
        self.assertNotIn('PAITON_HOST_KV_PRIVATE', os.environ)

    def test_over_full_disk_tier_drops_other_configurations_first(self):
        mine, other = server.disk_folder(), server.DISK_ROOT / 'other'
        for folder in (mine, other):
            folder.mkdir(parents=True)
            (folder / 'block.bin').write_bytes(b'x' * 1000)
        with mock.patch.object(server, 'DISK_MAX_GIB', 1500 / 2 ** 30), \
                mock.patch.object(server, 'CARD_POOL_GIB', 0.0):
            server.tier_settings(None, 'disk')
        self.assertTrue(mine.is_dir())
        self.assertFalse(other.exists())

    def test_image_and_script_carry_what_the_disk_tier_needs(self):
        docker, script = (FOLDER / 'Dockerfile').read_text(), (FOLDER / 'serve-docker.sh').read_text()
        source = 'paiton_vllm_plugin/_vendor/qwen38/compat/overlays/vllm/v1/kv_offload/cpu/gpu_worker.py'
        self.assertIn('COPY ' + source + ' ', docker)
        manifest = json.loads((ROOT / 'paiton_vllm_plugin/_vendor/qwen38/compat/compat_manifest.json').read_text())
        entry = next(v for k, v in manifest['files'].items() if k.endswith('vllm/v1/kv_offload/cpu/gpu_worker.py'))
        self.assertIn(entry['original_sha256'], docker)
        self.assertEqual(hashlib.sha256((ROOT / source).read_bytes()).hexdigest(), entry['overlay_sha256'])
        shm = int(script.split('--shm-size=')[1].split('g')[0])
        self.assertGreaterEqual(shm, server.STAGING_GIB + 0.5)


if __name__ == '__main__':
    unittest.main()
