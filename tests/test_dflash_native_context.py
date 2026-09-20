"""The native source hook and context wrapper must compose without bypasses."""
import hashlib
import importlib.machinery
import importlib.util
import json
import linecache
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from paiton_vllm_plugin.dflash import context, native_hooks

class CompositionTests(unittest.TestCase):
    def test_native_transform_then_context_guard_and_wrap(self):
        source = b'class DFlashQwen3Model:\n    def forward(self): pass\n'
        transformed = source.decode() + 'native_fixture = True\n'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root/'original.py'; original.write_text('original\n')
            overlay = root/'overlay.py'; overlay.write_bytes(source)
            receipts = {'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
                        'overlay_sha256':hashlib.sha256(source).hexdigest()}
            (root/'external-source-receipts.json').write_text(json.dumps(receipts))
            prior = importlib.machinery.SourceFileLoader(native_hooks.MODEL, str(overlay))
            prior.original = str(original)
            loader = context.Loader(native_hooks.Loader(prior, native_hooks.MODEL))
            spec = importlib.util.spec_from_file_location(native_hooks.MODEL, original, loader=loader)
            module = importlib.util.module_from_spec(spec)
            events = []
            def transform(name, data):
                self.assertEqual(name, native_hooks.MODEL)
                self.assertEqual(data, source)
                return transformed
            try:
                with patch.object(context,'ROOT',root), patch.object(native_hooks,'transform',transform), \
                     patch.object(context,'wrap',lambda m:events.append(m.native_fixture)), \
                     patch.dict(os.environ,{'PAITON_DFLASH_NATIVE':'1'}):
                    loader.exec_module(module)
                    self.assertEqual(events,[True])
                    self.assertTrue(module.DFlashQwen3Model.forward.__qualname__.endswith(native_hooks.SUFFIX))
                    linecache.cache[str(original)] = (1,None,['changed\n'],str(original))
                    with self.assertRaisesRegex(RuntimeError,'native integration source changed'):
                        context.verify_source(module)
            finally:
                linecache.cache.pop(str(original),None)

    def test_install_registers_source_transform_before_context_import(self):
        # Use a fresh interpreter so importing context actually registers its finder.
        import subprocess,sys
        code = '''
import os,sys,types
os.environ.update(PAITON_DFLASH_NATIVE='1', PAITON_DFLASH_CONTEXT_NORM_ROPE='1')
runtime=types.ModuleType('paiton_vllm_plugin.dflash.native_runtime')
runtime.POLICY=types.SimpleNamespace(cache_key='composition_test')
sys.modules[runtime.__name__]=runtime
from paiton_vllm_plugin.dflash import install
install()
from paiton_vllm_plugin.dflash import context,native_hooks
positions={type(f):i for i,f in enumerate(sys.meta_path)}
assert positions[context.Finder] < positions[native_hooks.Finder]
assert not any(n in sys.modules for n in ('torch','triton','vllm'))
'''
        subprocess.run([sys.executable,'-S','-c',code],env={'PATH':os.environ['PATH'],
            'PYTHONPATH':str(Path(__file__).resolve().parents[1])},check=True)

if __name__=='__main__':unittest.main()
