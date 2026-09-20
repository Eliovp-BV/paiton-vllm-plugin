import types
import unittest
from paiton_vllm_plugin.dflash.context_graph_guards import capture_allowed, configuration_allowed
N=types.SimpleNamespace
class Tests(unittest.TestCase):
    def config(self):
        return N(parallel_config=N(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,decode_context_parallel_size=1,prefill_context_parallel_size=1),
            model_config=N(enable_sleep_mode=False),lora_config=None,kv_transfer_config=None,cache_config=N(kv_offloading_size=None),
            offload_config=N(uva=N(cpu_offload_gb=0),prefetch=N(offload_group_size=0)),speculative_config=N(method='dflash',num_speculative_tokens=7))
    def test_supported(self):self.assertTrue(configuration_allowed(self.config()))
    def test_unsupported_falls_back(self):
        paths=[('parallel_config','tensor_parallel_size',2),('model_config','enable_sleep_mode',True),('cache_config','kv_offloading_size',1),('speculative_config','method','eagle')]
        for group,field,value in paths:
            c=self.config();setattr(getattr(c,group),field,value);self.assertFalse(configuration_allowed(c))
        c=self.config();c.offload_config.prefetch.offload_group_size=1;self.assertFalse(configuration_allowed(c))
        c=self.config();del c.offload_config;self.assertFalse(configuration_allowed(c))
    def test_live_or_disabled_never_captures(self):
        calls=[];m=N(cudagraph_capturing_enabled=True,validate_cudagraph_capturing_enabled=lambda:calls.append(1))
        self.assertFalse(capture_allowed([1],m));self.assertEqual(calls,[])
        m.cudagraph_capturing_enabled=False;self.assertFalse(capture_allowed(None,m));self.assertEqual(calls,[])
        m.cudagraph_capturing_enabled=True;self.assertTrue(capture_allowed(None,m));self.assertEqual(calls,[1])
if __name__=='__main__':unittest.main()
