import types
import unittest
from paiton_vllm_plugin.dflash.context_graph_runtime import CapturedTail as ContextGraph, model_key, slot_list

class Tensor:
    def __init__(self,shape,pointer,dtype='torch.bfloat16'):
        self.shape=shape; self.pointer=pointer; self.dtype=dtype; self.device='cuda:0'; self.is_cuda=True
    def stride(self):
        s=1; out=[]
        for n in reversed(self.shape): out.append(s); s*=n
        return tuple(reversed(out))
    def data_ptr(self): return self.pointer
    def is_contiguous(self): return True

class Tests(unittest.TestCase):
    def setUp(self):
        self.states=Tensor((8,5120),16); self.positions=Tensor((8,),32,'torch.int64')
        self.slots=[Tensor((8,),48+16*i,'torch.int64') for i in range(5)]
        impl=types.SimpleNamespace(kv_cache_dtype='fp8',head_size=128,_is_per_token_head_quant=False)
        self.model=types.SimpleNamespace(_num_attn_layers=5,_num_kv_heads=8,_head_dim=128,_kv_size=1024,
            _rms_norm_eps=1.e-6,_rope_head_size=128,_rope_is_neox=True,_fused_kv_bias=None,
            _hidden_norm_weight=Tensor((5120,),160),_fused_kv_weight=Tensor((10240,5120),176),
            _k_norm_weights=Tensor((5,128),192),_rope_cos_sin_cache=Tensor((65536,128),208),
            _attn_layers=[types.SimpleNamespace(kv_cache=Tensor((4,8,880,256),256+16*i,'torch.uint8'),
                _k_scale=Tensor((1,),400+16*i,'torch.float32'),_v_scale=Tensor((1,),500+16*i,'torch.float32'),impl=impl) for i in range(5)])
        self.events=[]
        self.stream=types.SimpleNamespace(cuda_stream=123)
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False,current_stream=lambda device:self.stream))
        hip=types.SimpleNamespace(hipMemcpyAsync=lambda *args:self.events.append(('copy',args)) or 0)
        self.runtime=ContextGraph(lambda *args:self.events.append(('original',)) or 'original',torch,hip)
        self.runtime.key=model_key(self.model,self.states,self.positions)[0]
        self.runtime.graph=types.SimpleNamespace(replay=lambda:self.events.append(('replay',)))
        self.runtime.slots=[Tensor((8,),800+16*i,'torch.int64') for i in range(5)]
        self.runtime.replay_stream=123

    def test_every_current_slot_copied_before_replay(self):
        self.runtime(self.model,self.states,self.positions,self.slots)
        self.assertEqual(len(self.events),6); self.assertEqual(self.events[-1],('replay',))
        for i,event in enumerate(self.events[:5]):
            self.assertEqual(event,('copy',(800+16*i,48+16*i,64,3,123)))

    def test_shared_mapping_copied_for_all_layers(self):
        self.runtime(self.model,self.states,self.positions,self.slots[0])
        self.assertEqual([x[1][1] for x in self.events[:5]],[48]*5)

    def test_pointer_change_or_stream_change_never_replays(self):
        for mode in ('input','weight','cache','scale','stream'):
            self.setUp()
            if mode=='input':self.states.pointer+=8
            if mode=='weight':self.model._fused_kv_weight.pointer+=8
            if mode=='cache':self.model._attn_layers[2].kv_cache.pointer+=8
            if mode=='scale':self.model._attn_layers[2]._v_scale.pointer+=8
            if mode=='stream':self.stream.cuda_stream+=1
            self.assertEqual(self.runtime(self.model,self.states,self.positions,self.slots),'original')
            self.assertEqual(self.events,[('original',)])

    def test_partial_layer_mapping_and_dummy_fall_back(self):
        for values in (None,[None]*5,self.slots[:4]):
            self.events.clear()
            self.assertEqual(self.runtime(self.model,self.states,self.positions,values),'original')
            self.assertEqual(self.events,[('original',)])

    def test_outside_shape_and_per_head_quant_rejected(self):
        self.assertIsNone(model_key(self.model,Tensor((9,5120),16),self.positions))
        self.model._attn_layers[0].impl._is_per_token_head_quant=True
        self.assertIsNone(model_key(self.model,self.states,self.positions))

    def test_gpu_copy_failure_is_not_hidden(self):
        self.runtime.hip.hipMemcpyAsync=lambda *args:7
        with self.assertRaisesRegex(RuntimeError,'HIP status'):self.runtime(self.model,self.states,self.positions,self.slots)
        self.assertEqual(self.runtime.replays,0)

if __name__=='__main__':unittest.main()
