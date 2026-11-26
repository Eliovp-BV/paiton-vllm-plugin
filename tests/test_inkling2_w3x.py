"""W3X loader (3-bit experts, Inkling-2-Mini fast profile): manifest, file read, expert module, no GPU."""
import ctypes as C
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from paiton_vllm_plugin.inkling2 import w3x
from paiton_vllm_plugin.inkling2.amd import experts_native

REAL = Path('/data/paiton/opus-98/inkling-w3/w3x/i3g64e_i3g64m')  # the weights session's v1 files (testbench only)
HEAD = dict(format='W3X', version=1, group=64, shapes=dict(w13=[2304, 2304], w2=[2304, 1152]))


def folder(entries, files=(), head=HEAD):
    root = Path(tempfile.mkdtemp())
    (root / 'manifest.json').write_text(json.dumps({**head, 'layers': entries}))
    for name in files:
        (root / name).write_bytes(b'')
    return str(root)


def fake_layer(root, layer, experts=2, shapes=None):
    """A small W3X layer file: codes = byte pattern by expert, scales = expert id, factor = expert id + 0.5."""
    tensors = {}
    for m, (n, k) in (shapes or w3x.SHAPES).items():
        codes = torch.arange(experts * w3x.codes_bytes(n, k), dtype=torch.int64) % 251
        tensors[f'{m}.codes'] = codes.to(torch.uint8).reshape(experts, n, w3x.codes_bytes(1, k))
        tensors[f'{m}.scales'] = torch.arange(experts, dtype=torch.uint8).repeat_interleave(n * k // 64).reshape(experts, n, k // 64)
        tensors[f'{m}.factor'] = torch.arange(experts, dtype=torch.float32) + 0.5
    name = f'layer_{layer:02d}.safetensors'
    save_file(tensors, str(Path(root) / name), metadata=dict(format='W3X', version='1', layer=str(layer), factor_stride='0'))
    return name


class Manifest(unittest.TestCase):
    def setUp(self):
        w3x._layers.cache_clear()

    def test_sizes(self):
        self.assertEqual(w3x.expert_bytes(), 3_110_408)  # notes/W3X-LAYOUT.md item 6
        all_w3 = folder([dict(layer=i, format='w3', file=f'l{i}', factor_stride=0) for i in range(24)], [f'l{i}' for i in range(24)])
        self.assertAlmostEqual(w3x.experts_gib(all_w3), 8.90, places=2)
        exact = folder([dict(layer=i, format='checkpoint') for i in range(24)])
        self.assertEqual(w3x.layers(exact), {})
        self.assertAlmostEqual(w3x.experts_gib(exact), 14.59, places=2)
        self.assertEqual(w3x.layers(None), {})  # no PAITON_INK2_W3X: the checkpoint's experts

    def test_refusals(self):
        good = dict(layer=0, format='w3', file='l0', factor_stride=0)
        for head, entries, files in (
                ({**HEAD, 'group': 128}, [good], ['l0']),
                ({**HEAD, 'shapes': dict(w13=[2304, 2304])}, [good], ['l0']),
                (HEAD, [good, good], ['l0']),  # one layer twice
                (HEAD, [{**good, 'layer': 24}], ['l0']),
                (HEAD, [{**good, 'format': 'w4'}], ['l0']),
                (HEAD, [{**good, 'factor_stride': 1}], ['l0']),  # per-row factors: formats 4 / 5, reserved
                (HEAD, [{**good, 'factor_stride': True}], ['l0']),
                (HEAD, [{**good, 'denominator': {'w13': 0}}], ['l0']),
                (HEAD, [{**good, 'denominator': {'w4': 2688}}], ['l0']),
                (HEAD, [{**good, 'denominator': 2688}], ['l0']),
                (HEAD, [{**good, 'file': '../l0'}], ['l0']),
                (HEAD, [good], [])):  # file missing
            w3x._layers.cache_clear()
            with self.assertRaises(ValueError):
                w3x.layers(folder(entries, files, head))
        with self.assertRaises(ValueError):
            w3x.layers('/nonexistent-w3x')

    def test_replaces_only_expert_tensors_of_3bit_layers(self):
        root = folder([dict(layer=3, format='w3', file='l3', factor_stride=0), dict(layer=4, format='checkpoint')], ['l3'])
        with mock.patch.dict(os.environ, PAITON_INK2_W3X=root):
            for part in ('w13_weight', 'w13_weight_scale', 'w2_weight_amax', 'w2_weight_outer_scale_denominator'):
                self.assertTrue(w3x.replaces(f'model.layers.3.mlp.experts.{part}'))
                self.assertFalse(w3x.replaces(f'model.layers.4.mlp.experts.{part}'))
            self.assertFalse(w3x.replaces('model.layers.3.mlp.gate.weight'))
            self.assertFalse(w3x.replaces('model.layers.13.mlp.experts.w13_weight'))


SMALL = {'w13': (32, 128), 'w2': (16, 128)}


class Read(unittest.TestCase):
    def setUp(self):
        w3x._layers.cache_clear()

    def test_read_hands_each_expert_row_block_to_the_repack(self):
        root = folder([], [])
        name = fake_layer(root, 5, shapes=SMALL)
        (Path(root) / 'manifest.json').write_text(json.dumps({**HEAD, 'shapes': {m: list(v) for m, v in SMALL.items()},
                                                       'layers': [dict(layer=5, format='w3', file=name, factor_stride=0)]}))
        calls = []

        def repack(src, n, k, out):  # stand-in for PaitonInkling2MoeW3Repack: copy, then invert
            size = w3x.codes_bytes(n, k)
            calls.append((n, k))
            C.memmove(out, src, size)
            buf = (C.c_uint8 * size).from_address(out)
            for i in range(size):
                buf[i] ^= 0xFF
            return 0
        with mock.patch.dict(os.environ, PAITON_INK2_W3X=root), mock.patch.object(w3x, 'SHAPES', SMALL), \
                mock.patch.object(w3x, 'EXPERTS', 2):
            out = w3x.read(5, repack, w3x.codes_bytes)
            for m, (n, k) in SMALL.items():
                card, scales, factor = out[m]
                want = (torch.arange(2 * w3x.codes_bytes(n, k)) % 251).to(torch.uint8).reshape(2, -1) ^ 0xFF
                self.assertTrue(torch.equal(card, want))
                self.assertEqual(scales.shape, (2, n, k // 64))
                self.assertEqual(scales[1].unique().tolist(), [1])
                self.assertEqual(factor.tolist(), [0.5, 1.5])
            self.assertEqual(calls, [SMALL['w13']] * 2 + [SMALL['w2']] * 2)
            with self.assertRaises(RuntimeError):  # a shape the library refuses
                w3x.read(5, lambda *a: 1, w3x.codes_bytes)
            with mock.patch.object(w3x, 'SHAPES', {'w13': (32, 256), 'w2': (16, 128)}), self.assertRaises(ValueError):
                w3x.read(5, repack, w3x.codes_bytes)  # file shape is not this model's

    @unittest.skipUnless((REAL / 'manifest.json').is_file(), 'needs the W3X v1 files on the testbench')
    def test_real_files(self):
        with mock.patch.dict(os.environ, PAITON_INK2_W3X=str(REAL)):
            self.assertEqual(sorted(w3x.layers()), list(range(24)))
            self.assertAlmostEqual(w3x.experts_gib(), 8.90, places=2)
            out = w3x.read(0, lambda src, n, k, dst: C.memmove(dst, src, w3x.codes_bytes(n, k)) and 0, w3x.codes_bytes)
        for m, (n, k) in w3x.SHAPES.items():
            card, scales, factor = out[m]
            self.assertEqual((card.shape, scales.shape, factor.shape), ((128, w3x.codes_bytes(n, k)), (128, n, k // 64), (128,)))
            self.assertTrue(torch.isfinite(factor).all() and (factor > 0).all())


CONFIG = SimpleNamespace(n_routed_experts=128, hidden_size=2304, intermediate_size=1152, num_experts_per_tok=8,
                         mlp_gate_softcap=4.0, mlp_up_softcap=25.0)


class Experts(unittest.TestCase):
    def setUp(self):
        w3x._layers.cache_clear()
        self.root = folder([dict(layer=2, format='w3', file='l2', factor_stride=0)], ['l2'])

    def test_3bit_layer_has_only_its_own_parameters(self):
        with mock.patch.dict(os.environ, PAITON_INK2_W3X=self.root), mock.patch.object(experts_native.expert_pool, 'ENABLED', False):
            w3 = experts_native.Inkling2NativeExperts(CONFIG, 'nvfp4', 2)
            exact = experts_native.Inkling2NativeExperts(CONFIG, 'nvfp4', 1)
            with mock.patch.object(experts_native.expert_pool, 'ENABLED', True), self.assertRaises(ValueError):
                experts_native.Inkling2NativeExperts(CONFIG, 'nvfp4', 2)  # no 3-bit experts in the pool
        shapes = {n: (tuple(p.shape), p.dtype) for n, p in w3.named_parameters()}
        self.assertEqual(shapes, {
            'w13_w3': ((128, 1_990_656), torch.uint8), 'w13_w3_scale': ((128, 2304, 36), torch.uint8),
            'w13_w3_factor': ((128,), torch.float32), 'w2_w3': ((128, 995_328), torch.uint8),
            'w2_w3_scale': ((128, 2304, 18), torch.uint8), 'w2_w3_factor': ((128,), torch.float32)})
        self.assertEqual(sum(p.numel() * p.element_size() for p in w3.parameters()), 128 * w3x.expert_bytes())
        self.assertIn('w13_weight', dict(exact.named_parameters()))

    def test_struct_and_load(self):
        lib = mock.Mock()
        lib.PaitonInkling2MoeW3Bytes.side_effect = w3x.codes_bytes
        host = {m: (torch.full((128, w3x.codes_bytes(n, k)), 7, dtype=torch.uint8), torch.ones(128, n, k // 64, dtype=torch.uint8),
                    torch.full((128,), 0.25)) for m, (n, k) in w3x.SHAPES.items()}
        with mock.patch.dict(os.environ, PAITON_INK2_W3X=self.root), mock.patch.object(experts_native.expert_pool, 'ENABLED', False), \
                mock.patch.object(experts_native, '_library', return_value=lib), mock.patch.object(w3x, 'read', return_value=host) as read:
            experts = {fmt: experts_native.Inkling2NativeExperts(CONFIG, fmt, 2) for fmt in ('nvfp4', 'mxfp8')}
            with mock.patch.object(experts_native, '_abi', 3), self.assertRaises(RuntimeError):
                experts['nvfp4'].load_w3()  # an ABI 3 library has no 3-bit kernels
            with mock.patch.object(experts_native, '_abi', 4), \
                    mock.patch.object(torch.Tensor, 'is_cuda', property(lambda self: True)):
                names = experts['nvfp4'].load_w3()
                self.assertEqual(read.call_args.args[:2], (2, lib.PaitonInkling2MoeW3Repack))
                self.assertEqual(sorted(names), sorted(n for n, _ in experts['nvfp4'].named_parameters()))
                self.assertTrue(bool((experts['nvfp4'].w2_w3 == 7).all()) and experts['nvfp4'].w13_w3_factor[0] == 0.25)
                for fmt, code in (('nvfp4', 2), ('mxfp8', 3)):  # W3X-LOADER question 1: 3 = MXFP8 activations
                    e = experts[fmt]
                    w = e._weights()
                    self.assertEqual((w.format, w.w13, w.w2_scale, w.w13_amax, w.w2_denominator, w.table),
                                     (code, e.w13_w3.data_ptr(), e.w2_w3_scale.data_ptr(), e.w13_w3_factor.data_ptr(), 1.0, None))
                    self.assertEqual(C.sizeof(w), 72)  # the struct the sequencer embeds (lead, 10:48)
                w3x._layers.cache_clear()  # a vehicle's amax and denominator for a bit gate
                (Path(self.root) / 'manifest.json').write_text(json.dumps(
                    {**HEAD, 'layers': [dict(layer=2, format='w3', file='l2', factor_stride=0, denominator={'w13': 2688})]}))
                w = experts['nvfp4']._weights()
                self.assertEqual((w.w13_denominator, w.w2_denominator), (2688.0, 1.0))


if __name__ == '__main__':
    unittest.main()
