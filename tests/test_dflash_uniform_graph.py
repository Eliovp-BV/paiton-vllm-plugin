"""Native-free lifecycle and exact uniform descriptor dispatch contracts."""
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch
from paiton_vllm_plugin.dflash.configuration import Settings
from paiton_vllm_plugin.dflash.serve import prepare_server_args
from paiton_vllm_plugin.dflash.uniform_graph_policy import eligible, extend_descriptors
from paiton_vllm_plugin.dflash import uniform_graph as g

class Mode(Enum): FULL=1; PIECEWISE=2
@dataclass(frozen=True)
class Desc:
 cg_mode: Mode
 num_tokens: int
 num_reqs: int | None
 uniform_token_count: int | None = None
 max_query_len: int | None = None
 num_active_loras: int = 0

class Policy(unittest.TestCase):
 def test_exact_cases_and_fallback_boundaries(self):
  for q in (4,5):self.assertTrue(eligible(8,8*q,q,0,q))
  for args in [(7,28,4,0,4),(8,31,4,0,4),(8,40,None,0,5),(8,32,4,1,4),(8,32,4,0,None),(8,48,6,0,6),(8,32,4,0,5)]:
   self.assertFalse(eligible(*args),args)
 def test_keep_originals_without_mutation(self):
  orig={Mode.FULL:[Desc(Mode.FULL,64,8,8),Desc(Mode.FULL,32,4,8)],Mode.PIECEWISE:[Desc(Mode.PIECEWISE,32,None)]}
  new,extra=extend_descriptors(orig,Desc,Mode.FULL)
  self.assertEqual(len(orig[Mode.FULL]),2);self.assertEqual(len(new[Mode.FULL]),4)
  self.assertEqual(new[Mode.PIECEWISE],orig[Mode.PIECEWISE]);self.assertIsNot(new[Mode.PIECEWISE],orig[Mode.PIECEWISE])
  self.assertEqual([(d.num_tokens,d.num_reqs,d.uniform_token_count) for d in extra],[(32,8,4),(40,8,5)])
  with self.assertRaises(RuntimeError):extend_descriptors(new,Desc,Mode.FULL)
 def test_default_aliases_and_configuration(self):
  self.assertFalse(Settings.from_environment({}).uniform_graphs)
  for value in ('1','on','true'):self.assertTrue(Settings.from_environment({'PAITON_DFLASH_UNIFORM_GRAPHS':value}).uniform_graphs)
  for name,value in [('PAITON_DFLASH_CONTEXT_GRAPH','1'),('PAITON_DFLASH_NATIVE','1'),('PAITON_DFLASH_RERANK','128'),('PAITON_DFLASH_DRAFT_SAMPLE_METHOD','greedy')]:
   with self.assertRaises(ValueError):Settings.from_environment({'PAITON_DFLASH_UNIFORM_GRAPHS':'1',name:value})
  args=['--speculative-config',json.dumps({'method':'dflash','num_speculative_tokens':7})]
  self.assertEqual(prepare_server_args(args,{'PAITON_DFLASH_UNIFORM_GRAPHS':'1'}),args)
  with self.assertRaises(ValueError):prepare_server_args([],{'PAITON_DFLASH_UNIFORM_GRAPHS':'1'})
 def test_no_framework_or_network_dependency_at_install(self):
  code="from paiton_vllm_plugin.dflash import install; import sys; s=install(); assert s.uniform_graphs; assert not any(n in sys.modules for n in ('torch','triton','vllm')); from paiton_vllm_plugin.dflash.uniform_graph import Finder; assert any(isinstance(x,Finder) for x in sys.meta_path)"
  subprocess.run([sys.executable,'-S','-c',code],env={'PATH':os.environ['PATH'],'PYTHONPATH':str(Path(__file__).parents[1]),'PAITON_DFLASH_UNIFORM_GRAPHS':'1'},check=True)
 def test_default_does_not_load_hook(self):
  code="from paiton_vllm_plugin.dflash import install; import sys; assert not install().uniform_graphs; assert 'paiton_vllm_plugin.dflash.uniform_graph' not in sys.modules"
  subprocess.run([sys.executable,'-S','-c',code],env={'PATH':os.environ['PATH'],'PYTHONPATH':str(Path(__file__).parents[1])},check=True)
 def module(self):
  class Manager:
   def _init_candidates(self):pass
   def capture(self,*args,**kwargs):
    self.calls.append(list(self._capture_descs))
    for mode,descs in self._capture_descs.items():
     for desc in descs:self.graphs[desc]=object()
    return 'original-result'
   def dispatch(self,*args):return ('fallback',args)
  cuda=types.SimpleNamespace(synchronize=lambda d:None,memory_reserved=lambda d:100,mem_get_info=lambda d:(1000,2000))
  return types.SimpleNamespace(ModelCudaGraphManager=Manager,CUDAGraphMode=Mode,BatchExecutionDescriptor=Desc,torch=types.SimpleNamespace(cuda=cuda))
 def manager(self,module):
  m=module.ModelCudaGraphManager();m._paiton_uniform_original_descs={Mode.FULL:[Desc(Mode.FULL,64,8,8)],Mode.PIECEWISE:[Desc(Mode.PIECEWISE,64,None)]}
  m._capture_descs,m._paiton_uniform_extra=extend_descriptors(m._paiton_uniform_original_descs,Desc,Mode.FULL)
  m._max_full_descs_to_capture=None;m.graphs={};m.calls=[];m.device=0;m._paiton_uniform_ready=False;m._paiton_uniform_hits={4:0,5:0};return m
 def test_original_then_extra_capture_and_exact_dispatch(self):
  module=self.module()
  with patch.object(g,'validate_source'):g.bind(module)
  m=self.manager(module);all_descs=m._capture_descs
  self.assertEqual(m.capture(),'original-result');self.assertTrue(m._paiton_uniform_ready);self.assertIs(m._capture_descs,all_descs)
  self.assertEqual(m.calls,[[Mode.FULL,Mode.PIECEWISE],[Mode.FULL]])
  self.assertEqual(m.dispatch(8,32,4,0,4),m._paiton_uniform_extra[0])
  self.assertEqual(m.dispatch(8,40,None,0,5),('fallback',(8,40,None,0,5)))
 def test_profiling_and_resource_failure_cannot_enable_dispatch(self):
  module=self.module()
  with patch.object(g,'validate_source'):g.bind(module)
  m=self.manager(module);m._max_full_descs_to_capture=2;m.capture();self.assertEqual(len(m.calls),1);self.assertFalse(m._paiton_uniform_ready)
  m=self.manager(module);all_descs=m._capture_descs
  with patch.object(module.torch.cuda,'memory_reserved',side_effect=[100,100+g.MAX_EXTRA_BYTES+1]):
   with self.assertRaises(RuntimeError):m.capture()
  self.assertFalse(m._paiton_uniform_ready);self.assertIs(m._capture_descs,all_descs)

if __name__=='__main__':unittest.main()
