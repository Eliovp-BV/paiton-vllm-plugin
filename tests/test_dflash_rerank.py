import importlib.util,hashlib,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('draft_scoped',Path(__file__).parents[1]/'paiton_vllm_plugin/dflash/rerank.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
FIXTURE='''
import types
RERANK=80
FAST=True
KCAND=8
BITS=2
_HEAD_CACHE={}
def _apply_head_int2(self,lm_head,hidden_states,embedding_bias):
    return (RERANK, KCAND)
def _quantize_head_now(lp,lm_head):
    lp._radiance_wq=object();lp._radiance_n=248320
    lp._apply_head=types.MethodType(_apply_head_int2,lp)
    return 'bound'
def _quantize_draft_head(mtp,lp_attr='logits_processor'):
    lp=getattr(mtp,lp_attr);lp._radiance_topk_only=lp_attr=='candidate_logits_processor'
    if getattr(mtp,'lazy',False):return 'deferred'
    return _quantize_head_now(lp,mtp.lm_head)
'''
class Tests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);p=Path(self.t.name)/'head.py';p.write_text(FIXTURE)
  self.m=types.ModuleType('head');self.m.__file__=str(p);exec(compile(FIXTURE,str(p),'exec'),self.m.__dict__)
  self.original=self.m._apply_head_int2
  self.pool=getattr(self,'pool',128)
  self.block_candidates=getattr(self,'block_candidates',8)
  with patch.object(a,'SOURCE_SHA',hashlib.sha256(p.read_bytes()).hexdigest()):a.bind(self.m,self.pool,self.block_candidates)
  cls=type(a.DRAFT_CLASS[1],(),{});cls.__module__=a.DRAFT_CLASS[0];self.d=cls();self.d.lm_head=object();self.d.candidate_logits_processor=types.SimpleNamespace();self.d.logits_processor=types.SimpleNamespace()
  self.hidden=types.SimpleNamespace(shape=(8,5120))
 def test_draft_and_target_share_packing_but_not_policy(self):
  self.m._quantize_draft_head(self.d,'candidate_logits_processor');draft=self.d.candidate_logits_processor
  target=types.SimpleNamespace(_radiance_topk_only=True);self.m._quantize_head_now(target,self.d.lm_head)
  self.assertEqual(draft._apply_head(None,self.hidden,None),(self.pool,self.block_candidates))
  self.assertEqual(target._apply_head(None,self.hidden,None),(80,8))
  self.assertEqual(self.m.RERANK//4,20);self.assertIs(self.m._apply_head_int2,self.original)
  self.assertFalse(hasattr(target,'_paiton_draft_rerank'))
 def test_lazy_draft_binds_only_when_weights_exist(self):
  self.d.lazy=True;self.m._quantize_draft_head(self.d,'candidate_logits_processor');lp=self.d.candidate_logits_processor
  self.assertFalse(hasattr(lp,'_apply_head'));self.m._quantize_head_now(lp,self.d.lm_head)
  self.assertEqual(lp._apply_head(None,self.hidden,None),(self.pool,self.block_candidates));self.assertEqual(self.m.RERANK,80)
 def test_other_draft_logit_processor_is_unchanged(self):
  self.m._quantize_draft_head(self.d,'logits_processor')
  self.assertEqual(self.d.logits_processor._apply_head(None,self.hidden,None),(80,8))
 def test_shared_target_candidate_instance_rejected(self):
  self.d.logits_processor=self.d.candidate_logits_processor
  with self.assertRaises(RuntimeError):self.m._quantize_draft_head(self.d,'candidate_logits_processor')
 def test_target_policy_mutation_is_detected(self):
  self.m._quantize_draft_head(self.d,'candidate_logits_processor');self.m.RERANK=256
  with self.assertRaises(RuntimeError):self.d.candidate_logits_processor._apply_head(None,self.hidden,None)
 def test_wrong_runtime_source_rejected(self):
  with self.assertRaises(ImportError):a.bind(self.m)
 def test_target_block_policy_mutation_detected(self):
  self.m._quantize_draft_head(self.d,'candidate_logits_processor');self.m.KCAND=16
  with self.assertRaises(RuntimeError):self.d.candidate_logits_processor._apply_head(None,self.hidden,None)
 def test_draft_marker_mutation_detected(self):
  self.m._quantize_draft_head(self.d,'candidate_logits_processor');self.d.candidate_logits_processor._paiton_draft_block_candidates=32
  with self.assertRaises(RuntimeError):self.d.candidate_logits_processor._apply_head(None,self.hidden,None)
class WiderPoolTests(Tests):
 pool=256
class WiderBlockTests(Tests):
 block_candidates=16

class ConfigTests(unittest.TestCase):
 def test_exact_menu_and_required_target_defaults(self):
  base={'RADIANCE_DRAFT_RERANK':'80','RADIANCE_FAST_DRAFT':'1'}
  for pool in (128,256):self.assertEqual(a.configuration(base|{a.FLAG:str(pool)}),pool)
  for value in ('512','80','-1','bad'):
   with self.assertRaises(ValueError):a.configuration(base|{a.FLAG:value})
  with self.assertRaises(ValueError):a.configuration(base|{a.FLAG:'256','RADIANCE_DRAFT_RERANK':'128'})
 def test_block_policy_menu_is_bounded(self):
  env={a.BLOCK_FLAG:'16','PAITON_DFLASH_DRAFT_SAMPLE_METHOD':'probabilistic'}
  self.assertEqual(a.block_configuration(env,128),16)
  self.assertEqual(a.block_configuration({},128),8)
  for pool in (None,256):
   with self.assertRaises(ValueError):a.block_configuration(env,pool)
  with self.assertRaises(ValueError):a.block_configuration({a.BLOCK_FLAG:'16'},128)
  with self.assertRaises(ValueError):a.block_configuration(env|{a.BLOCK_FLAG:'32'},128)

if __name__=='__main__':unittest.main()
