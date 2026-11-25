import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from paiton_vllm_plugin.inkling2.cache_plan import cache_bytes

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('inkling2_server_small', ROOT / 'models/Inkling-2-Mini/server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)
GIB = 2 ** 30


class Inkling2SmallCard(unittest.TestCase):
    def host(self, avail_gib, gtt_total_gib, gtt_used_gib=0.0):
        """A fake host: /proc/meminfo and one AMD card's GTT counters."""
        folder = Path(tempfile.mkdtemp())
        (folder / 'meminfo').write_text(f'MemTotal: 32000000 kB\nMemAvailable: {int(avail_gib * 2 ** 20)} kB\n')
        (folder / 'mem_info_gtt_total').write_text(str(int(gtt_total_gib * GIB)))
        (folder / 'mem_info_gtt_used').write_text(str(int(gtt_used_gib * GIB)))
        return [mock.patch.object(server, 'MEMINFO', folder / 'meminfo'),
                mock.patch.object(server, 'amd_cards', lambda: [folder]), mock.patch.object(server, 'EXPERT_POOL', True)]

    def run_small(self, patches, *args):
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return server.small_card(*args)

    def test_free_vram_skips_integrated_gpus(self):
        sysfs, dri = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        cards = []
        for node, total, used in (('renderD128', 0.5, 0.2), ('renderD129', 31.86, 0.06), ('renderD130', 15.9, 0.3)):
            dev = sysfs / node / 'device'
            dev.mkdir(parents=True)
            (dev / 'mem_info_vram_total').write_text(str(int(total * GIB)))
            (dev / 'mem_info_vram_used').write_text(str(int(used * GIB)))
            cards.append(dev)
        (dri / 'renderD128').touch()
        (dri / 'renderD129').touch()
        with mock.patch.object(server, 'amd_cards', lambda: cards):
            self.assertAlmostEqual(server.card_free_gib(dri), 31.8, places=2)  # the iGPU and the unmapped card ignored
            (dri / 'renderD130').touch()
            self.assertAlmostEqual(server.card_free_gib(dri), 15.6, places=2)

    def test_without_the_pool_it_stops_with_the_plan(self):
        with self.assertRaises(SystemExit) as stop:
            server.small_card(15.9, 32768, True, False, 'bf16')
        self.assertIn('16 GB profile', str(stop.exception))
        self.assertIn('not released yet', str(stop.exception))

    def test_dflash2_drafter_takes_its_size_and_its_cache_from_the_pool(self):
        own = float(self.run_small(self.host(28, 15.6), 15.9, 32768, True, False, 'bf16')[1]['PAITON_INK2_EXPERT_POOL_GIB'])
        other = float(self.run_small(self.host(28, 15.6), 15.9, 32768, 'dflash2', False, 'bf16')[1]['PAITON_INK2_EXPERT_POOL_GIB'])
        self.assertAlmostEqual(own - other, server.DFLASH2_GIB - server.DRAFTER_GIB
                               + (cache_bytes(32768, 1, 256, drafter='dflash2') - cache_bytes(32768, 1, 256, drafter=True)) / GIB,
                               delta=0.011)

    def test_pool_matches_the_plan_table(self):
        # PLAN.md section 1: a 15.9 GiB card keeps 9.39 GiB of experts at 32K and 8.26 at 64K with the drafter.
        for context, pool in ((32768, 9.39), (65536, 8.26)):
            args, env, note = self.run_small(self.host(28, 15.6), 15.9, context, True, False, 'bf16')
            self.assertAlmostEqual(float(env['PAITON_INK2_EXPERT_POOL_GIB']), pool, delta=0.02)
            self.assertEqual(args[args.index('--max-model-len') + 1], str(context))
        self.assertIn('hit rate 0.9', self.run_small(self.host(28, 15.6), 15.9, 32768, True, False, 'bf16')[2])

    def test_arguments_and_environment(self):
        args, env, _ = self.run_small(self.host(28, 15.6), 15.9, 32768, True, False, 'bf16')
        self.assertEqual(args[args.index('--max-num-seqs') + 1], '1')
        self.assertEqual(args[args.index('--max-num-batched-tokens') + 1], '256')
        self.assertEqual(json.loads(args[args.index('--limit-mm-per-prompt') + 1]), {'image': 0})
        self.assertGreater(int(args[args.index('--kv-cache-memory-bytes') + 1]), 1.8 * GIB)
        self.assertEqual((env['HSA_USERPTR_FOR_PAGED_MEM'], env['PAITON_INK2_HOST_EMBED']), ('0', '1'))
        vision, venv, _ = self.run_small(self.host(28, 15.6), 15.9, 32768, True, True, 'bf16')
        self.assertNotIn('--limit-mm-per-prompt', vision)
        self.assertAlmostEqual(float(env['PAITON_INK2_EXPERT_POOL_GIB']) - float(venv['PAITON_INK2_EXPERT_POOL_GIB']),
                               server.VISION_GIB, delta=0.011)

    def test_dry_run_prints_the_plan_on_any_host(self):
        import contextlib
        import io
        import os
        import sys
        patches = self.host(28, 15.6)[:2]  # the pool flag stays False: a dry run plans without it
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        out = io.StringIO()
        with mock.patch.object(sys, 'argv', ['server.py', '--dry-run', '15.9']), mock.patch.dict(os.environ), \
                contextlib.redirect_stdout(out):
            server.main()
        lines = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertIn('16 GB profile: 15.9 GiB free VRAM', lines[0]['card_profile'])
        argv = lines[1]['command']
        self.assertEqual(argv[len(argv) - argv[::-1].index('--max-model-len')], '32768')  # the profile's, last wins
        self.assertEqual(lines[2]['environment']['PAITON_INK2_HOST_EMBED'], '1')

    def test_refuses_a_host_that_cannot_pin_the_copies(self):
        for avail, gtt in ((20, 15.6), (28, 12.0)):  # 16 GB host; GTT already in use
            with self.assertRaises(SystemExit) as stop:
                self.run_small(self.host(avail, gtt), 15.9, 32768, True, False, 'bf16')
            self.assertIn('pinned host memory', str(stop.exception))

    def test_refuses_a_context_that_leaves_no_pool(self):
        with self.assertRaises(SystemExit) as stop:
            self.run_small(self.host(28, 15.6), 15.9, 327680, True, False, 'bf16')
        self.assertIn('shorter --max-model-len', str(stop.exception))


if __name__ == '__main__':
    unittest.main()
