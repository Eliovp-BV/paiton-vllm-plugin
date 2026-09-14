import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import unittest

spec=importlib.util.spec_from_file_location('gguf_detokenizer',Path(__file__).parents[1]/'paiton_vllm_plugin/gguf_detokenizer.py')
compat=importlib.util.module_from_spec(spec);spec.loader.exec_module(compat)


class DetokenizerCompatibilityTest(unittest.TestCase):
    def test_scoped_fallback_and_idempotence(self):
        class Factory:
            @classmethod
            def from_new_request(cls,tokenizer,request):return ('fast',request)
        fake=ModuleType('vllm.v1.engine.detokenizer')
        fake.IncrementalDetokenizer=Factory
        fake.SlowIncrementalDetokenizer=lambda tokenizer,request:('slow',request)
        request=SimpleNamespace(prompt_token_ids=[100],sampling_params=SimpleNamespace(skip_special_tokens=True))
        incomplete=SimpleNamespace(decode=lambda *args,**kwargs:'prefix\ufffd')
        valid=SimpleNamespace(decode=lambda *args,**kwargs:'café 中文')
        with patch.dict('sys.modules',{'vllm.v1.engine.detokenizer':fake}),patch.dict('os.environ',{'PAITON_GGUF_SAFE_DETOKENIZER':'0'}):
            compat.install_gguf_detokenizer_compat()
            self.assertEqual(Factory.from_new_request(incomplete,request)[0],'fast')
            with patch.dict('os.environ',{'PAITON_GGUF_SAFE_DETOKENIZER':'1'}):
                compat.install_gguf_detokenizer_compat();first=Factory.from_new_request.__func__
                compat.install_gguf_detokenizer_compat()
                self.assertIs(Factory.from_new_request.__func__,first)
                self.assertEqual(Factory.from_new_request(incomplete,request)[0],'slow')
                self.assertEqual(Factory.from_new_request(valid,request)[0],'fast')
                self.assertEqual(Factory.from_new_request(None,request)[0],'fast')
