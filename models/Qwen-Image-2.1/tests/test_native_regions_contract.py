"""CPU-only adapter safety/rollback checks; external harness, never compiler tests."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock
import torch

from paiton_image21 import native_regions as adapter

class Contract(unittest.TestCase):
    def test_unsupported_configuration_leaves_model_untouched(self):
        original=object()
        model=NS(config=NS(num_attention_heads=16,attention_head_dim=256,num_layers=32,mlp_ratio=3),transformer_blocks=[original])
        with mock.patch.object(adapter,'NativeRegions',side_effect=AssertionError('must not load library')):
            self.assertIsNone(adapter.install(model,'unused'))
        self.assertIs(model.transformer_blocks[0],original)

    def test_existing_custom_attention_processor_is_preserved(self):
        original=object()
        model=NS(config=NS(num_attention_heads=32,attention_head_dim=128,num_layers=32,mlp_ratio=3),transformer_blocks=[NS(attn=NS(processor=original))])
        with mock.patch.object(adapter,'NativeRegions',side_effect=AssertionError('must not load library')):
            self.assertIsNone(adapter.install(model,'unused'))
        self.assertIs(model.transformer_blocks[0].attn.processor,original)

    def test_corrupt_artifact_rejected_before_gpu_or_dlopen(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'fixture.so').write_bytes(b'corrupt')
            (p/'manifest.json').write_text(json.dumps(dict(file='fixture.so',sha256='0'*64,architecture='gfx1201',abi_version=1)))
            with mock.patch.object(adapter.C,'CDLL',side_effect=AssertionError('must not load corrupt library')):
                with self.assertRaisesRegex(RuntimeError,'artifact identity mismatch'):
                    adapter.NativeRegions(p)

    def test_unsupported_runtime_input_uses_original_attention(self):
        calls=[]
        class Fallback:
            _attention_backend=None
            _parallel_config=None
            def __call__(self,*a,**kw):calls.append((a,kw));return 'original result'
        native=NS(counts={'fallback':0})
        processor=adapter.NativeAttentionProcessor(native,Fallback())
        hidden=torch.zeros((1,2,4096),dtype=torch.bfloat16)
        cache=object()
        self.assertEqual(processor('attention',hidden,layer_cache=cache,kv_cache_mode='cached'),'original result')
        self.assertEqual(native.counts['fallback'],1)
        self.assertIs(calls[0][0][4],cache)
        self.assertEqual(calls[0][0][5],'cached')

    def test_upstream_source_change_keeps_original_model(self):
        model=object()
        with mock.patch.object(adapter,'qualified_upstream',return_value=False), mock.patch.object(adapter,'NativeRegions',side_effect=AssertionError('must not load')):
            self.assertIsNone(adapter.install(model,'unused'))

    def test_reference_backend_cannot_silently_enable_native_regions(self):
        from paiton_image21.runtime import ImageEngine
        with self.assertRaisesRegex(ValueError,'requires the native backend'):
            ImageEngine('unused',backend='reference',native_fusions=True)

    def test_unqualified_torch_version_keeps_original_model(self):
        with mock.patch.object(adapter.torch, '__version__', 'unqualified-build'), mock.patch.object(adapter, 'NativeRegions', side_effect=AssertionError('must not load')):
            self.assertIsNone(adapter.install(object(), 'unused'))

    def test_changed_normalization_source_keeps_original_model(self):
        hashes = ['0eb0555e21ca93195e1fe9389113cafbf0e8822f6454c3001cebb3d21521ebd7', '0'*64]
        with mock.patch.object(adapter, '_file_sha256', side_effect=hashes), mock.patch.object(adapter, 'NativeRegions', side_effect=AssertionError('must not load')):
            self.assertIsNone(adapter.install(object(), 'unused'))

if __name__=='__main__':unittest.main()
