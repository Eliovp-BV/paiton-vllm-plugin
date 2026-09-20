import copy
from types import SimpleNamespace as S
import unittest
from paiton_vllm_plugin.dflash.uniform_graph_guards import runtime_scope
class Scope(unittest.TestCase):
    def baseline(self):
        return S(cache_config=S(enable_prefix_caching=False),scheduler_config=S(async_scheduling=False),
                 speculative_config=S(draft_sample_method='greedy',disable_padded_drafter_batch=True),
                 model_config=S(max_model_len=65536))
    def test_qualified_profile_and_unsupported_contexts(self):
        self.assertTrue(runtime_scope(self.baseline()))
        for owner,field,value in [('cache_config','enable_prefix_caching',True),('scheduler_config','async_scheduling',True),
                                  ('speculative_config','draft_sample_method','probabilistic'),('speculative_config','disable_padded_drafter_batch',False),
                                  ('model_config','max_model_len',220000)]:
            config=self.baseline();setattr(getattr(config,owner),field,value)
            self.assertFalse(runtime_scope(config),(owner,field,value))
            delattr(getattr(config,owner),field)
            self.assertFalse(runtime_scope(config),(owner,field,'missing'))
        self.assertFalse(runtime_scope(S()))

import hashlib,json,tempfile
from pathlib import Path
from unittest.mock import patch
from paiton_vllm_plugin.dflash import uniform_graph_guards as guard

class SourceGuard(unittest.TestCase):
    def test_changed_or_escaped_overlay_is_rejected(self):
        digest=lambda text:hashlib.sha256(text.encode()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);runtime=base/'paiton_runtime_compat';runtime.mkdir()
            path=base/'runner.py';path.write_text('original')
            overlay=runtime/'active.py';overlay.write_text('effective')
            entry={'original_sha256':digest('original'),'overlay_sha256':digest('effective'),'overlay':'active.py'}
            def receipt():
                (runtime/'compat_manifest.json').write_text(json.dumps({'files':{str(path):entry}}))
            with patch.object(guard,'PINS',{'runner.py':(digest('original'),digest('effective'))}):
                receipt();guard.validate_runtime_sources(base)
                overlay.write_text('modified')
                with self.assertRaises(RuntimeError):guard.validate_runtime_sources(base)
                overlay.write_text('effective');outside=base/'outside.py';outside.write_text('effective')
                entry['overlay']='../outside.py';receipt()
                with self.assertRaises(RuntimeError):guard.validate_runtime_sources(base)
                entry['overlay']='active.py';receipt();path.write_text('modified original')
                with self.assertRaises(RuntimeError):guard.validate_runtime_sources(base)

if __name__=='__main__':unittest.main()
