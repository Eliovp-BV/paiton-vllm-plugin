"""Host configuration and lifecycle failures; no GPU or framework imports."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from paiton_vllm_plugin.dflash.configuration import Settings
from paiton_vllm_plugin.dflash.serve import prepare_server_args
from paiton_vllm_plugin.dflash import native_hooks

BASE={'method':'dflash','model':'/draft','num_speculative_tokens':7,
      'draft_tensor_parallel_size':1,'max_model_len':65536,
      'disable_padded_drafter_batch':True,'draft_sample_method':'greedy'}

class ConfigurationTests(unittest.TestCase):
 def test_disabled_does_not_import_framework_or_require_artifacts(self):
  code="import sys; from paiton_vllm_plugin.dflash import install; a=install(); b=install(); assert a==b and not a.native and not a.rerank; assert not any(x in sys.modules for x in ('torch','triton','vllm','paiton_vllm_plugin.dflash.native_runtime','paiton_vllm_plugin.dflash.native_hooks'))"
  env={'PATH':os.environ['PATH'],'PYTHONPATH':str(Path(__file__).parents[1])}
  subprocess.run([sys.executable,'-S','-c',code],env=env,check=True)
 def test_default_preserves_exact_arguments(self):
  args=['--model','/target','--speculative-config',json.dumps(BASE),'--max-num-seqs','8']
  self.assertEqual(prepare_server_args(args,{}),args)
 def test_probability_changes_only_draft_option(self):
  args=['--model','/target','--speculative-config',json.dumps(BASE),'--dtype','bfloat16']
  out=prepare_server_args(args,{'PAITON_DFLASH_DRAFT_SAMPLE_METHOD':'probabilistic'})
  self.assertEqual(out[:3],args[:3]);self.assertEqual(out[4:],args[4:])
  v=json.loads(out[3]);self.assertEqual(v.pop('draft_sample_method'),'probabilistic')
  self.assertEqual(v,{k:x for k,x in BASE.items() if k!='draft_sample_method'})
  self.assertEqual(BASE['draft_sample_method'],'greedy')
 def test_inline_and_rerank(self):
  env={'PAITON_DFLASH_DRAFT_SAMPLE_METHOD':'probabilistic','PAITON_DFLASH_RERANK':'128'}
  out=prepare_server_args(['--speculative-config='+json.dumps(BASE)],env)
  self.assertEqual(json.loads(out[0].split('=',1)[1])['draft_sample_method'],'probabilistic')
 def test_wrong_method_tp_or_width_rejected(self):
  for field,value in [('method','ngram'),('num_speculative_tokens',15),('draft_tensor_parallel_size',2)]:
   with self.subTest(field=field),self.assertRaises(ValueError):
    prepare_server_args(['--speculative-config',json.dumps(BASE|{field:value})],{'PAITON_DFLASH_NATIVE':'1'})
 def test_invalid_flags_and_unavailable_native_features(self):
  for env in ({'PAITON_DFLASH_NATIVE':'yes'},{'PAITON_DFLASH_RERANK':'512'},{'PAITON_DFLASH_DRAFT_SAMPLE_METHOD':'fast'}, {'PAITON_DFLASH_NATIVE':'1','PAITON_DFLASH_CONTEXT_KV':'1'}):
   with self.subTest(env=env),self.assertRaises(ValueError):Settings.from_environment(env)
 def test_rerank_requires_screened_sampling(self):
  with self.assertRaises(ValueError):prepare_server_args(['--speculative-config',json.dumps(BASE)],{'PAITON_DFLASH_RERANK':'128'})
 def test_block_policy_requires_exact_supported_combination(self):
  env={'PAITON_DFLASH_RERANK':'128','PAITON_DFLASH_DRAFT_SAMPLE_METHOD':'probabilistic','PAITON_DFLASH_BLOCK_CANDIDATES':'16'}
  self.assertEqual(Settings.from_environment(env).block_candidates,16)
  self.assertEqual(Settings.from_environment({}).block_candidates,0)
  for bad in ({'PAITON_DFLASH_RERANK':'256'},{'PAITON_DFLASH_RERANK':'0'},
              {'PAITON_DFLASH_DRAFT_SAMPLE_METHOD':'inherit'},{'PAITON_DFLASH_BLOCK_CANDIDATES':'32'}):
   with self.subTest(bad=bad),self.assertRaises(ValueError):Settings.from_environment(env|bad)
 def test_worker_receipt_cannot_hide_changed_block_policy(self):
  import paiton_vllm_plugin.dflash as d
  env={'PAITON_DFLASH_RERANK':'128','PAITON_DFLASH_DRAFT_SAMPLE_METHOD':'probabilistic',
       'PAITON_DFLASH_BLOCK_CANDIDATES':'16','_PAITON_DFLASH_SERVER_OPTIONS_V3':json.dumps(['probabilistic',128,0,0])}
  with patch.dict(os.environ,env,clear=True),patch.object(d,'_installed',None):
   with self.assertRaisesRegex(RuntimeError,'paiton-dflash-serve'):d.install()
 def test_proposal_option_cannot_be_silently_ignored(self):
  import paiton_vllm_plugin.dflash as d
  with patch.dict(os.environ,{'PAITON_DFLASH_DRAFT_SAMPLE_METHOD':'probabilistic'},clear=True),patch.object(d,'_installed',None):
   with self.assertRaisesRegex(RuntimeError,'paiton-dflash-serve'):d.install()
 def test_missing_duplicate_or_invalid_spec_rejected(self):
  for args in ([],['--speculative-config'],['--speculative-config','[]'],['--speculative-config','{}','--speculative-config={}']):
   with self.subTest(args=args),self.assertRaises(ValueError):prepare_server_args(args,{'PAITON_DFLASH_NATIVE':'1'})
 def test_settings_frozen_per_worker(self):
  import paiton_vllm_plugin.dflash as d
  with patch.dict(os.environ,{},clear=True),patch.object(d,'_installed',None):
   d.install()
   with patch.dict(os.environ,{'PAITON_DFLASH_SILU_QUANT':'0'}),self.assertRaises(RuntimeError):d.install()

class HookTests(unittest.TestCase):
 def test_unreviewed_source_rejected(self):
  for name in (native_hooks.MODEL,native_hooks.UTILS):
   with self.subTest(name=name),self.assertRaises(ImportError):native_hooks.transform(name,b'changed source')
 def test_forward_identity_is_draft_only(self):
  def forward():pass
  forward.__qualname__='DFlashQwen3Model.forward'
  module=types.SimpleNamespace(__name__=native_hooks.MODEL,DFlashQwen3Model=types.SimpleNamespace(forward=forward))
  native_hooks.after_load(module)
  self.assertEqual(forward.__qualname__,'DFlashQwen3Model.forward_paiton_native_plugin_v1')
 def test_loader_delegates_existing_import_hook(self):
  events=[]
  class Prior:
   path='unused'
   def create_module(self,spec):events.append('create');return None
   def exec_module(self,module):events.append('prior')
  with patch.object(native_hooks,'after_load',lambda m:events.append('after')):
   loader=native_hooks.Loader(Prior(),native_hooks.DISPATCH)
   loader.create_module(None);loader.exec_module(types.SimpleNamespace())
  self.assertEqual(events,['create','prior','after'])
 def test_late_model_import_fails_before_installing(self):
  runtime=types.ModuleType('paiton_vllm_plugin.dflash.native_runtime')
  runtime.POLICY=types.SimpleNamespace(cache_key='test')
  with patch.dict(os.environ,{'PAITON_DFLASH_NATIVE':'1'}),patch.dict(sys.modules,{native_hooks.MODEL:types.ModuleType(native_hooks.MODEL),runtime.__name__:runtime}),patch.object(native_hooks,'SUFFIX',native_hooks.SUFFIX):
   with self.assertRaises(RuntimeError):native_hooks.install()
 def test_component_flags_and_artifacts_have_distinct_cache_identities(self):
  from paiton_vllm_plugin.dflash.native_policy import Policy
  keys=set()
  for down in ('0','1'):
   for silu in ('0','1'):
    for artifact in ('a'*64,'b'*64):
     env={'PAITON_DFLASH_NATIVE':'1','PAITON_DFLASH_DOWN_PROJECTION':down,'PAITON_DFLASH_SILU_QUANT':silu}
     p=Policy.from_environment(env,available_features=['down_projection','silu_quant'],artifact_sha256=artifact)
     keys.add(p.cache_key)
  self.assertEqual(len(keys),8)


class ProjectionFallbackTests(unittest.TestCase):
 def test_unsupported_call_preserves_arguments_and_reference(self):
  calls=[]
  def original(*args,**kwargs):calls.append((args,kwargs));return 'reference'
  with tempfile.TemporaryDirectory() as tmp:
   source=Path(tmp)/'dispatch.py';source.write_text('fixture')
   import hashlib
   module=types.SimpleNamespace(__file__=str(source),_PS=types.SimpleNamespace(gemm_a8w8_blockscale_preshuffle=original),torch=types.SimpleNamespace(bfloat16='bf16'))
   runtime=types.ModuleType('paiton_vllm_plugin.dflash.native_runtime')
   def run(A,B,As,Bs,N,K,fallback):return 'native' if (A.shape,N,K)==((8,17408),5120,17408) else fallback()
   runtime.run=run
   with patch.dict(native_hooks.HASHES,{native_hooks.DISPATCH:hashlib.sha256(source.read_bytes()).hexdigest()}),patch.dict(sys.modules,{runtime.__name__:runtime}):
    native_hooks.bind_projection(module)
    wrapper=module._PS.gemm_a8w8_blockscale_preshuffle
    args=(types.SimpleNamespace(shape=(8,17408)),types.SimpleNamespace(shape=(320,278528)),'As','Bs','bf16')
    self.assertEqual(wrapper(*args,is_x_scale_tranposed=False),'native')
    for extra,kw in [((),{}), ((),{'is_x_scale_tranposed':True}),((42,),{'is_x_scale_tranposed':False}), ((),{'is_x_scale_tranposed':False,'unknown':7})]:
     self.assertEqual(wrapper(*args,*extra,**kw),'reference')
     self.assertEqual(calls[-1],(args+extra,kw))
    with self.assertRaises(RuntimeError):native_hooks.bind_projection(module)

if __name__=='__main__':unittest.main()
