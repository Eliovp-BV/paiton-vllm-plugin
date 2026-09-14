"""Verify atomic cache preparation and preservation of the downloaded source."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch, call
from huggingface_hub.errors import LocalEntryNotFoundError

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from sdnq_tool import __main__ as cli


class ConversionCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.output=self.root/'flux2-klein-runtime'
        self.source=self.root/'source';self.source.mkdir()
        (self.source/'weights').write_bytes(b'original checkpoint')
        self.pipeline=ModuleType('sdnq_tool.pipeline')
        self.pipeline.download_model=Mock(return_value=self.source)
        self.converter=ModuleType('sdnq_tool.convert')
        self.converter.main=Mock()

    def tearDown(self):self.temp.cleanup()

    def invoke(self):
        def redirect(value):
            return self.output if str(value)=='/models/flux2-klein-runtime' else Path(value)
        with patch.object(cli,'Path',side_effect=redirect), patch.object(sys,'argv',['tool','convert']), \
             patch.dict(sys.modules,{'sdnq_tool.pipeline':self.pipeline,'sdnq_tool.convert':self.converter}):
            cli.main()

    def test_failure_discards_only_staging(self):
        def fail():
            target=Path(sys.argv[sys.argv.index('--output')+1]);target.mkdir()
            (target/'partial').write_bytes(b'partial')
            raise RuntimeError('Simulated conversion failure')
        self.converter.main.side_effect=fail
        with self.assertRaisesRegex(RuntimeError,'Simulated conversion failure'):self.invoke()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('.flux2-conversion-*')),[])
        self.assertEqual((self.source/'weights').read_bytes(),b'original checkpoint')

    def test_success_promotes_complete_cache(self):
        def convert():
            target=Path(sys.argv[sys.argv.index('--output')+1]);target.mkdir()
            (target/'conversion.json').write_text('{}')
            (target/'tensor').write_bytes(b'complete')
        self.converter.main.side_effect=convert
        self.invoke()
        self.assertEqual((self.output/'tensor').read_bytes(),b'complete')
        self.assertEqual(list(self.root.glob('.flux2-conversion-*')),[])
        self.assertEqual((self.source/'weights').read_bytes(),b'original checkpoint')

    def prepare_existing_cache(self):
        self.output.mkdir()
        metadata={'format_version':1,'source_model':'Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic',
                  'source_revision':'45e9cc76cb70f84473ce5c6c2e2282d0ef3c6ecd'}
        (self.output/'conversion.json').write_text(json.dumps(metadata))

    def test_pinned_cache_uses_only_local_source_and_skips_conversion(self):
        self.prepare_existing_cache()
        self.invoke()
        self.pipeline.download_model.assert_called_once_with(local_files_only=True)
        self.converter.main.assert_not_called()

    def test_existing_tensors_restore_missing_source_without_reconversion(self):
        self.prepare_existing_cache()
        self.pipeline.download_model.side_effect=[LocalEntryNotFoundError("Missing cached source"),self.source]
        self.invoke()
        self.assertEqual(self.pipeline.download_model.call_args_list,[call(local_files_only=True),call()])
        self.converter.main.assert_not_called()
        self.assertEqual((self.source/'weights').read_bytes(),b'original checkpoint')


if __name__=='__main__':unittest.main()
