"""Behavior checks at the external host boundary, without framework imports."""
import importlib.util
import json
import hashlib
import linecache
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from paiton_vllm_plugin.dflash import context as a
from paiton_vllm_plugin.dflash.configuration import Settings
from paiton_vllm_plugin.dflash.serve import prepare_server_args

class Tensor:
    def __init__(self,shape,pointer=16,dtype='torch.bfloat16'):
        self.shape=shape; self.pointer=pointer; self.dtype=dtype; self.device='cuda:0'; self.is_cuda=True
    def is_contiguous(self): return True
    def data_ptr(self): return self.pointer
    def numel(self): return 40960
    def __getitem__(self,index): return ('view',self.pointer,index)

class Tests(unittest.TestCase):
    def setUp(self):
        self.events=[]; events=self.events
        class Model:
            def _build_fused_kv_buffers(self): pass
            def _project_context_kv(self,*args): events.append('project'); return Tensor((5,8,8,128),16),Tensor((5,8,8,128),32)
            def _normalize_context_k(self,k): events.append('norm'); return Tensor(k.shape,48)
            def precompute_and_store_context_kv(self,context_states,context_positions,context_slot_mapping=None):
                k,_=self._project_context_kv(context_states,8,5,8,128)
                self._normalize_context_k(k); events.append('original_rope_and_cache'); return 'original'
        module=types.SimpleNamespace(DFlashQwen3Model=Model)
        a.wrap(module); self.model=Model()
        for key,value in dict(_num_attn_layers=5,_num_kv_heads=8,_head_dim=128,_kv_size=1024,
                _rms_norm_eps=1.e-6,_rope_is_neox=True,_rope_head_size=128,
                _k_norm_weights=Tensor((5,128)),_rope_cos_sin_cache=Tensor((65536,128))).items():
            setattr(self.model,key,value)
        self.model._attn_layers=[types.SimpleNamespace(kv_cache=i,impl=types.SimpleNamespace(
            do_kv_cache_update=lambda *args:events.append(('cache',args[1:]))) ) for i in range(5)]
        a.LIB=object(); a.POLICY=a.Policy(True,'a'*64); a.SHADOW=False; a.LOCAL.active=None
        a.AUDIT=types.SimpleNamespace(paiton_draft_audit_compare=lambda *args:events.append(('compare',args)) or 0)
        self.states=Tensor((8,5120)); self.positions=Tensor((8,),dtype='torch.int64')

    def test_opt_in_requires_dflash_contract_before_server_import(self):
        env={'PAITON_DFLASH_CONTEXT_NORM_ROPE':'1'}
        self.assertFalse(Settings.from_environment({}).context_norm_rope)
        self.assertTrue(Settings.from_environment(env).context_norm_rope)
        for args in ([],['--speculative-config','{"method":"ngram","num_speculative_tokens":7}']):
            with self.assertRaises(ValueError): prepare_server_args(args,env)
        args=['--speculative-config','{"method":"dflash","num_speculative_tokens":7}']
        self.assertEqual(prepare_server_args(args,env),args)

    def test_shadow_returns_original_and_compares_after_original_rope(self):
        a.SHADOW=True
        with patch.object(a,'launch',return_value=(Tensor((5,8,8,128),64),112)):
            result=self.model.precompute_and_store_context_kv(self.states,self.positions,'slots')
        self.assertEqual(result,'original'); self.assertEqual(self.events[:3],['project','norm','original_rope_and_cache'])
        self.assertEqual(self.events[3],('compare',(48,64,40960,112))); self.assertIsNone(a.LOCAL.active)

    def test_native_path_preserves_layer_slots_and_values(self):
        slots=['a',None,'c',None,'e']
        with patch.object(a,'launch',return_value=(Tensor((5,8,8,128),64),112)):
            self.assertIsNone(self.model.precompute_and_store_context_kv(self.states,self.positions,slots))
        self.assertEqual(self.events[0],'project'); self.assertEqual(len(self.events),4)
        for event,i in zip(self.events[1:],(0,2,4)):
            self.assertEqual(event,('cache',(('view',64,i),('view',32,i),i,slots[i])))

    def test_dummy_cache_none_does_not_write(self):
        with patch.object(a,'launch',return_value=(Tensor((5,8,8,128),64),112)):
            self.model.precompute_and_store_context_kv(self.states,self.positions,None)
        self.assertEqual(self.events,['project'])

    def test_unqualified_rows_or_dtype_fall_back(self):
        for shape,dtype in [((7,5120),'torch.bfloat16'),((8,5120),'torch.float16')]:
            self.events.clear()
            with patch.object(a,'launch') as native:
                result=self.model.precompute_and_store_context_kv(Tensor(shape,dtype=dtype),self.positions)
            self.assertEqual(result,'original'); native.assert_not_called()

    def test_source_guard_checks_active_overlay_not_only_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'original.py'; source.write_text('original\n')
            hashes={'original_sha256':hashlib.sha256(b'original\n').hexdigest(),
                    'overlay_sha256':hashlib.sha256(b'overlay\n').hexdigest()}
            (root/'external-source-receipts.json').write_text(json.dumps(hashes))
            module=types.SimpleNamespace(__file__=str(source))
            with patch.object(a,'ROOT',root):
                linecache.cache[str(source)]=(8,None,['overlay\n'],str(source)); a.verify_source(module)
                linecache.cache.pop(str(source))
                with self.assertRaisesRegex(RuntimeError,'active overlay'): a.verify_source(module)

if __name__=='__main__': unittest.main()
