"""Opt-in scheduling configuration and framework-free disabled lifecycle."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from paiton_vllm_plugin.dflash.configuration import Settings
from paiton_vllm_plugin.dflash.serve import prepare_server_args

class Tests(unittest.TestCase):
    def test_default_and_aliases(self):
        self.assertFalse(Settings.from_environment({}).context_graph)
        for value in ('1','on','true'):
            self.assertTrue(Settings.from_environment({'PAITON_DFLASH_CONTEXT_GRAPH':value}).context_graph)
            code="import sys; from paiton_vllm_plugin.dflash import install; s=install(); assert s.context_graph; from paiton_vllm_plugin.dflash.context_graph import ENABLED, Finder; assert ENABLED and any(isinstance(f,Finder) for f in sys.meta_path); assert 'torch' not in sys.modules and 'triton' not in sys.modules and 'vllm' not in sys.modules"
            subprocess.run([sys.executable,'-S','-c',code],env={'PATH':os.environ['PATH'],'PYTHONPATH':str(Path(__file__).parents[1]),'PAITON_DFLASH_CONTEXT_GRAPH':value},check=True)
    def test_contract_required_without_mutating_vllm_arguments(self):
        env={'PAITON_DFLASH_CONTEXT_GRAPH':'1'}
        spec={'method':'dflash','num_speculative_tokens':7,'draft_tensor_parallel_size':1}
        args=['--speculative-config',json.dumps(spec),'--dtype','bfloat16']
        self.assertEqual(prepare_server_args(args,env),args)
        for changed in ({'method':'eagle'},{'num_speculative_tokens':15},{'draft_tensor_parallel_size':2}):
            with self.assertRaises(ValueError):prepare_server_args(['--speculative-config',json.dumps(spec|changed)],env)
    def test_unqualified_combination_rejected(self):
        with self.assertRaises(ValueError):Settings.from_environment({'PAITON_DFLASH_CONTEXT_GRAPH':'1','PAITON_DFLASH_CONTEXT_NORM_ROPE':'1'})
    def test_default_does_not_import_graph_module(self):
        code="import sys; from paiton_vllm_plugin.dflash import install; assert not install().context_graph; assert 'paiton_vllm_plugin.dflash.context_graph' not in sys.modules and 'torch' not in sys.modules"
        subprocess.run([sys.executable,'-S','-c',code],env={'PATH':os.environ['PATH'],'PYTHONPATH':str(Path(__file__).parents[1])},check=True)
if __name__=='__main__':unittest.main()
