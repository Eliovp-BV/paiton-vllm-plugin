"""CPU test for the compile-cache seed (A4): an empty namespace is filled from the seed, a warm one is left alone."""
import importlib.util
import json
import os
import pathlib
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cache_namespace_under_test', HERE / 'cache_namespace.py')
cn = importlib.util.module_from_spec(spec); spec.loader.exec_module(cn)


class Test(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        (self.tmp / 'compat').mkdir(); (self.tmp / 'compat' / 'bootstrap.py').write_text('# b\n')
        seed = self.tmp / 'seed'
        for kind, name in (('triton', 'abc123/k.hsaco'), ('inductor', 'ik/x/y.py'), ('comgr', 'llvmcache-1/obj')):
            f = seed / kind / name; f.parent.mkdir(parents=True); f.write_text(kind)
        self.env = dict(os.environ)
        for k in list(os.environ):
            if k.startswith(('PAITON_RUNTIME_COMPAT_CACHE', 'RADIANCE_', 'PAITON_COMPILE_CACHE', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'VLLM_CACHE_ROOT', 'AMD_COMGR')):
                del os.environ[k]
        os.environ['XDG_CACHE_HOME'] = str(self.tmp / 'cache'); os.environ['PAITON_CACHE_SEED'] = str(seed)

    def tearDown(self):
        os.environ.clear(); os.environ.update(self.env)

    def test_seed_fills_empty_namespace_once(self):
        ns = cn.configure(self.tmp / 'compat', {'files': {}})
        triton = pathlib.Path(os.environ['TRITON_CACHE_DIR']); self.assertTrue(triton.name == ns and (triton / 'abc123' / 'k.hsaco').read_text() == 'triton')
        self.assertEqual((pathlib.Path(os.environ['TORCHINDUCTOR_CACHE_DIR']) / 'ik' / 'x' / 'y.py').read_text(), 'inductor')
        self.assertEqual((self.tmp / 'cache' / 'comgr' / 'llvmcache-1' / 'obj').read_text(), 'comgr')
        self.assertEqual([p.name for p in triton.parent.iterdir()], [ns])   # no temp directory left behind
        # a warm cache is left alone: add a file, re-run, the seed does not overwrite or duplicate anything
        (triton / 'warm.txt').write_text('x'); (self.tmp / 'seed' / 'triton' / 'abc123' / 'k.hsaco').write_text('changed')
        del os.environ['PAITON_RUNTIME_COMPAT_CACHE_BASES']
        self.assertEqual(cn.configure(self.tmp / 'compat', {'files': {}}), ns)
        self.assertEqual((triton / 'abc123' / 'k.hsaco').read_text(), 'triton'); self.assertTrue((triton / 'warm.txt').exists())

    def test_seed_off_and_missing(self):
        os.environ['PAITON_CACHE_SEED'] = 'off'; cn.configure(self.tmp / 'compat', {'files': {}})
        self.assertFalse(pathlib.Path(os.environ['TRITON_CACHE_DIR']).exists())
        del os.environ['PAITON_RUNTIME_COMPAT_CACHE_BASES']; os.environ['PAITON_CACHE_SEED'] = str(self.tmp / 'nowhere')
        cn.configure(self.tmp / 'compat', {'files': {}}); self.assertFalse(pathlib.Path(os.environ['TRITON_CACHE_DIR']).exists())


if __name__ == '__main__':
    unittest.main()
