from collections import namedtuple
import hashlib
import importlib.util
import json
import os
import sys
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

    def served(self, *arguments, library=True, prepare=None):
        """(speculative config or None, mocked prepare) of a launcher start with `arguments`."""
        prepare = prepare or mock.Mock(side_effect=lambda offline=False, verify=False, lock=server.LOCK:
                                       '/draft' if lock is not server.LOCK else '/checkpoint')
        with mock.patch.dict(os.environ, {server.DFLASH2_SO: __file__} if library else {}), \
                mock.patch.object(server, 'prepare', prepare), mock.patch.object(server, 'card_free_gib', return_value=None), \
                mock.patch.object(server.os, 'execv') as execv, mock.patch.object(server.sys, 'argv', ['server.py', *arguments]):
            if not library:
                os.environ.pop(server.DFLASH2_SO, None)
            server.main()
        if not execv.called:
            return None, prepare
        argv = execv.call_args.args[1]
        return (json.loads(argv[argv.index('--speculative-config') + 1]) if '--speculative-config' in argv else None), prepare

    def test_both_drafters_can_be_served_and_dflash2_is_the_default(self):
        entry = json.loads((FOLDER / 'checkpoint.lock.json').read_text())['drafters']['dflash2']
        self.assertEqual(len(entry['revision']), 40)
        self.assertTrue(all(len(item['sha256']) == 64 and item['bytes'] > 0 for item in entry['files']))
        self.assertEqual(server.DEFAULT_DRAFTER, 'dflash2')
        for arguments in ((), ('--drafter', 'dflash2')):
            config, prepare = self.served(*arguments)
            self.assertEqual((config['method'], config['model'], config['num_speculative_tokens']), ('dspark', '/draft', 7))
            self.assertEqual(config['num_speculative_tokens_per_batch_size'], [[1, 1, 5], [2, 4, 4]])
            self.assertEqual([call.args[2:] for call in prepare.call_args_list], [(), (entry,)])
        config, prepare = self.served('--drafter', 'dspark')
        self.assertEqual(config, server.DRAFTER)
        self.assertNotIn('model', config)
        self.assertEqual(prepare.call_count, 1)  # the model's own drafter is in the checkpoint
        config, prepare = self.served('--no-drafter')
        self.assertIsNone(config)
        self.assertEqual(prepare.call_count, 1)

    def test_offline_verify_and_download_only_reach_the_drafter_files(self):
        entry = server.LOCK['drafters']['dflash2']
        config, prepare = self.served('--offline', '--verify')
        self.assertEqual([call.args for call in prepare.call_args_list], [(True, True), (True, True, entry)])
        config, prepare = self.served('--download-only')
        self.assertIsNone(config)  # nothing served
        self.assertEqual([call.args for call in prepare.call_args_list], [(False, False), (False, False, entry)])
        config, prepare = self.served('--download-only', '--drafter', 'dspark')
        self.assertEqual(prepare.call_count, 1)

    def test_offline_start_without_the_drafter_files_stops_with_the_way_out(self):
        def missing(offline=False, verify=False, lock=server.LOCK):
            if lock is not server.LOCK:
                raise FileNotFoundError('no snapshot')  # what huggingface_hub raises for a repository not in the cache
            return '/checkpoint'
        with self.assertRaises(SystemExit) as stop:
            self.served('--offline', prepare=mock.Mock(side_effect=missing))
        for part in ('--offline', 'DFlash2', '--drafter dspark', server.LOCK['drafters']['dflash2']['revision']):
            self.assertIn(part, str(stop.exception))
        with self.assertRaises(FileNotFoundError):  # online, the download's own error stays
            self.served(prepare=mock.Mock(side_effect=missing))

    def test_image_without_the_drafter_library_stops_before_vllm(self):
        with self.assertRaises(SystemExit) as stop:
            self.served(library=False)
        self.assertIn(server.DFLASH2_SO, str(stop.exception))
        self.assertEqual(self.served('--drafter', 'dspark', library=False)[0], server.DRAFTER)

    def test_prepare_checks_the_files_of_the_entry_it_is_given(self):
        entry = dict(model='someone/draft', revision='r' * 40, files=[dict(name='a.bin', bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())])
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'a.bin').write_bytes(b'abc')
            hub = mock.Mock(snapshot_download=mock.Mock(return_value=folder))
            with mock.patch.dict(sys.modules, {'huggingface_hub': hub}), mock.patch.dict(os.environ, {'XDG_CACHE_HOME': folder}):
                self.assertEqual(server.prepare(True, True, entry), Path(folder))
                self.assertEqual(hub.snapshot_download.call_args.args, ('someone/draft',))
                self.assertEqual(hub.snapshot_download.call_args.kwargs['revision'], 'r' * 40)
                self.assertTrue(hub.snapshot_download.call_args.kwargs['local_files_only'])
                (Path(folder) / 'a.bin').write_bytes(b'abd')
                with self.assertRaises(ValueError):
                    server.prepare(True, True, entry)

    def environment(self, *arguments, **preset):
        """The environment the launcher hands to vLLM for `arguments`, with `preset` already set by the user."""
        # no card: the profile of the host that runs the tests (a busy card looks like a 16 GB one) stays out
        preset = {server.DFLASH2_SO: __file__, **preset}  # the default drafter's library, as the image sets it
        with mock.patch.dict(os.environ, preset), mock.patch.object(server, 'prepare', return_value='/checkpoint'), \
                mock.patch.object(server, 'card_free_gib', return_value=None), \
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

    def test_server_pins_come_off_the_limit(self):
        with mock.patch.dict(os.environ, {'PAITON_HOST_PIN_LIMIT_GIB': ''}), \
                mock.patch.object(server, 'pin_limit_gib', lambda: 40.0):
            self.assertEqual(server.tier_settings(None, 'auto', 15.45)[0][1], '24.5')   # floor to 0.5 GiB
            with self.assertRaises(SystemExit):
                server.tier_settings(30.0, None, 15.45)
        self.assertEqual(server.tier_settings(None, 'auto', 15.45)[0][1], '64')   # an explicit limit stays as given

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
        site = '/usr/local/lib/python3.14/dist-packages/'
        self.assertIn('COPY models/Inkling-2-Mini/vllm-overlays/ ' + site + '\n', docker)
        overlays = FOLDER / 'vllm-overlays'
        files = sorted(p.relative_to(overlays).as_posix() for p in overlays.rglob('*') if p.is_file())
        self.assertEqual(len(files), 5)
        for rel in files:  # every file the folder replaces is checked against vLLM 0.32's own
            self.assertIn('  ' + site + rel + '"', docker)
        shm = int(script.split('--shm-size=')[1].split('g')[0])
        self.assertGreaterEqual(shm, server.STAGING_GIB + 0.5)


if __name__ == '__main__':
    unittest.main()
