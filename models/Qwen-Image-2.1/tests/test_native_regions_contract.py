"""CPU-only adapter safety/rollback checks; external harness, never compiler tests."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock
import torch

from paiton_image21 import native_regions as adapter

class Contract(unittest.TestCase):
    def test_unsupported_configuration_leaves_model_untouched(self):
        original=object()
        model=NS(config=NS(num_attention_heads=16,attention_head_dim=256,num_layers=32,mlp_ratio=3),transformer_blocks=[original])
        with mock.patch.object(adapter,'NativeRegions',side_effect=AssertionError('must not load library')):
            self.assertIsNone(adapter.install(model,'unused'))
        self.assertIs(model.transformer_blocks[0],original)

    def test_existing_custom_attention_processor_is_preserved(self):
        original=object()
        model=NS(config=NS(num_attention_heads=32,attention_head_dim=128,num_layers=32,mlp_ratio=3),transformer_blocks=[NS(attn=NS(processor=original))])
        with mock.patch.object(adapter,'NativeRegions',side_effect=AssertionError('must not load library')):
            self.assertIsNone(adapter.install(model,'unused'))
        self.assertIs(model.transformer_blocks[0].attn.processor,original)

    def test_corrupt_artifact_rejected_before_gpu_or_dlopen(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'fixture.so').write_bytes(b'corrupt')
            (p/'manifest.json').write_text(json.dumps(dict(file='fixture.so',sha256='0'*64,architecture='gfx1201',abi_version=1)))
            with mock.patch.object(adapter.C,'CDLL',side_effect=AssertionError('must not load corrupt library')):
                with self.assertRaisesRegex(RuntimeError,'artifact identity mismatch'):
                    adapter.NativeRegions(p)

    def test_unsupported_runtime_input_uses_original_attention(self):
        calls=[]
        class Fallback:
            _attention_backend=None
            _parallel_config=None
            def __call__(self,*a,**kw):calls.append((a,kw));return 'original result'
        native=NS(counts={'fallback':0})
        processor=adapter.NativeAttentionProcessor(native,Fallback())
        hidden=torch.zeros((1,2,4096),dtype=torch.bfloat16)
        cache=object()
        self.assertEqual(processor('attention',hidden,layer_cache=cache,kv_cache_mode='cached'),'original result')
        self.assertEqual(native.counts['fallback'],1)
        self.assertIs(calls[0][0][4],cache)
        self.assertEqual(calls[0][0][5],'cached')

    def test_upstream_source_change_keeps_original_model(self):
        model=object()
        with mock.patch.object(adapter,'qualified_upstream',return_value=False), mock.patch.object(adapter,'NativeRegions',side_effect=AssertionError('must not load')):
            self.assertIsNone(adapter.install(model,'unused'))

    def test_reference_backend_cannot_silently_enable_native_regions(self):
        from paiton_image21.runtime import ImageEngine
        with self.assertRaisesRegex(ValueError,'requires the native backend'):
            ImageEngine('unused',backend='reference',native_fusions=True)

    def test_unqualified_torch_version_keeps_original_model(self):
        with mock.patch.object(adapter.torch, '__version__', 'unqualified-build'), mock.patch.object(adapter, 'NativeRegions', side_effect=AssertionError('must not load')):
            self.assertIsNone(adapter.install(object(), 'unused'))

    def test_changed_normalization_source_keeps_original_model(self):
        hashes = ['0eb0555e21ca93195e1fe9389113cafbf0e8822f6454c3001cebb3d21521ebd7', '0'*64]
        with mock.patch.object(adapter, '_file_sha256', side_effect=hashes), mock.patch.object(adapter, 'NativeRegions', side_effect=AssertionError('must not load')):
            self.assertIsNone(adapter.install(object(), 'unused'))

if __name__=='__main__':unittest.main()


class Companions(unittest.TestCase):
    """Exact attention and fused normalization companions: fallback rules without any GPU or library."""
    def _regions(self,**kw):
        fields=dict(attention_library=None,normfuse_library=None,_mask_checks={},counts={'attention_mask_fallback':0});fields.update(kw)
        ns=NS(**fields)
        ns._dense_input=adapter.NativeRegions._dense_input
        ns._mask_is_trivial=lambda mask,_self=ns:adapter.NativeRegions._mask_is_trivial(_self,mask)
        return ns

    def test_missing_attention_companion_keeps_packed_framework_attention(self):
        regions=self._regions()
        q=torch.zeros((1,4,32,128),dtype=torch.bfloat16)
        self.assertFalse(adapter.NativeRegions.attention_native_usable(regions,q,q,q,None,None,None))

    def test_dense_input_rules(self):
        good=torch.zeros((1,4,32,128),dtype=torch.bfloat16)
        self.assertFalse(adapter.NativeRegions._dense_input(good))   # CPU tensor never qualifies
        with mock.patch.object(torch.Tensor,'is_cuda',new_callable=mock.PropertyMock,return_value=True):
            self.assertTrue(adapter.NativeRegions._dense_input(good))
            self.assertFalse(adapter.NativeRegions._dense_input(good.float()))
            self.assertFalse(adapter.NativeRegions._dense_input(good.transpose(1,2)))
            self.assertFalse(adapter.NativeRegions._dense_input(torch.zeros((2,4,32,128),dtype=torch.bfloat16)))
            self.assertFalse(adapter.NativeRegions._dense_input(torch.zeros((1,4,16,128),dtype=torch.bfloat16)))

    def test_partial_key_validity_mask_falls_back_and_is_counted(self):
        regions=self._regions(attention_library=object())
        q=torch.zeros((1,4,32,128),dtype=torch.bfloat16)
        mask=torch.ones((1,1,1,4),dtype=torch.bool);mask[0,0,0,2]=False
        with mock.patch.object(torch.Tensor,'is_cuda',new_callable=mock.PropertyMock,return_value=True):
            self.assertFalse(adapter.NativeRegions.attention_native_usable(regions,q,q,q,None,None,mask))
            self.assertEqual(regions.counts['attention_mask_fallback'],1)
            self.assertTrue(adapter.NativeRegions.attention_native_usable(regions,q,q,q,None,None,torch.ones((1,1,1,4),dtype=torch.bool)))
            self.assertTrue(adapter.NativeRegions.attention_native_usable(regions,q,q,q,None,None,None))
            self.assertFalse(adapter.NativeRegions.attention_native_usable(regions,q,q,q,None,None,torch.zeros((1,1,4,4),dtype=torch.float)))

    def test_all_true_mask_check_is_cached_per_storage(self):
        regions=self._regions(attention_library=object())
        mask=torch.ones((1,1,1,8),dtype=torch.bool)
        with mock.patch.object(torch.Tensor,'all',wraps=mask.all) as spy:
            self.assertTrue(regions._mask_is_trivial(mask));self.assertTrue(regions._mask_is_trivial(mask))
        self.assertEqual(spy.call_count,1)
        mask[0,0,0,3]=False   # in-place change bumps the version counter: re-evaluated
        self.assertFalse(regions._mask_is_trivial(mask))

    def test_prefix_buffers_must_pair_and_fit(self):
        regions=self._regions(attention_library=object())
        q=torch.zeros((1,4,32,128),dtype=torch.bfloat16);pre=torch.zeros((1,2,32,128),dtype=torch.bfloat16)
        with mock.patch.object(torch.Tensor,'is_cuda',new_callable=mock.PropertyMock,return_value=True):
            self.assertTrue(adapter.NativeRegions.attention_native_usable(regions,q,q,q,pre,pre,None))
            self.assertFalse(adapter.NativeRegions.attention_native_usable(regions,q,q,q,pre,None,None))
            self.assertFalse(adapter.NativeRegions.attention_native_usable(regions,q,q,q,pre,torch.zeros((1,3,32,128),dtype=torch.bfloat16),None))

    def test_cached_decode_passes_prefix_without_concatenation(self):
        calls=[]
        native=NS(counts={'fallback':0},norm_rope=lambda x,norm,freq:x,
                  attention_native_usable=lambda *a:True,
                  attention_native=lambda q,k,v,kp,vp:(calls.append((k.shape,kp.shape,vp.shape)) or q))
        class Fallback:
            _attention_backend=None
            _parallel_config=None
        linear=lambda x:x.reshape(x.shape[0],x.shape[1],4096)
        attn=NS(to_q=linear,to_k=linear,to_v=linear,norm_q=None,norm_k=None,to_out=[lambda x:x,lambda x:x])
        hidden=torch.zeros((1,3,4096),dtype=torch.bfloat16)
        cached=torch.zeros((1,2,32,128),dtype=torch.bfloat16)
        cache=NS(get=lambda:(cached,cached))
        rope=torch.zeros((3,64),dtype=torch.complex64)
        with torch.no_grad(), mock.patch.object(torch.Tensor,'is_cuda',new_callable=mock.PropertyMock,return_value=True), mock.patch.object(torch,'cat',side_effect=AssertionError('must not concatenate')):
            out=adapter.NativeAttentionProcessor(native,Fallback())(attn,hidden,rotary_emb=rope,layer_cache=cache,kv_cache_mode='cached')
        self.assertEqual(tuple(out.shape),(1,3,4096))
        self.assertEqual(calls,[((1,3,32,128),(1,2,32,128),(1,2,32,128))])

    def test_segmented_prefill_keeps_masked_framework_path(self):
        seen=[]
        native=NS(counts={'fallback':0},norm_rope=lambda x,norm,freq:x,
                  attention_native_usable=lambda *a:(seen.append('asked') or True),
                  attention_native=lambda *a:AssertionError('native attention must not run with segments'),
                  attention=lambda q,k,v,**kw:q)
        class Fallback:
            _attention_backend=None
            _parallel_config=None
        linear=lambda x:x.reshape(x.shape[0],x.shape[1],4096)
        attn=NS(to_q=linear,to_k=linear,to_v=linear,norm_q=None,norm_k=None,to_out=[lambda x:x,lambda x:x])
        hidden=torch.zeros((1,3,4096),dtype=torch.bfloat16)
        with torch.no_grad(), mock.patch.object(torch.Tensor,'is_cuda',new_callable=mock.PropertyMock,return_value=True):
            out=adapter.NativeAttentionProcessor(native,Fallback())(attn,hidden,segments=[(0,1,True)])
        self.assertEqual(tuple(out.shape),(1,3,4096))
        self.assertEqual(seen,[])

    def test_affine_layernorm_disables_fused_normalization_only(self):
        captured=[]
        class Regions:
            def __init__(self,directory,attention_directory=None,normfuse_directory=None):
                captured.append((directory,attention_directory,normfuse_directory));self.normfuse_library=None;self.counts={}
        weight=torch.ones(128,dtype=torch.bfloat16)
        norm=NS(weight=weight,bias=None)
        block=NS(attn=NS(processor=adapter.QwenImage21AttnProcessor(),norm_q=norm,norm_k=norm,set_processor=lambda p:None),
                 img_norm1=torch.nn.LayerNorm(4096,elementwise_affine=True),img_norm2=torch.nn.LayerNorm(4096,elementwise_affine=False),forward=lambda *a,**k:None)
        model=NS(config=NS(num_attention_heads=32,attention_head_dim=128,num_layers=32,mlp_ratio=3),transformer_blocks=[block])
        with mock.patch.object(adapter,'qualified_upstream',return_value=True), mock.patch.object(adapter,'NativeRegions',Regions):
            self.assertIsNotNone(adapter.install(model,'regions','attention','normfuse'))
        self.assertEqual(captured,[('regions','attention',None)])


class AttentionCallShape(unittest.TestCase):
    def test_attention_entry_point_receives_stream_before_variant(self):
        seen=[]
        regions=NS(attention_workspace_elements=lambda k,p,h:64,_workspace=None,counts={'attention_native':0},
                   attention_pack_v=lambda *a:(seen.append(('pack',a)) or 0),
                   attention_kernel=lambda *a:(seen.append(('attn',a)) or 0))
        regions.call=lambda fn,*a:adapter.NativeRegions.call(regions,fn,*a)
        q=torch.zeros((1,3,32,128),dtype=torch.bfloat16);pre=torch.zeros((1,2,32,128),dtype=torch.bfloat16)
        with mock.patch.object(adapter.torch.cuda,'current_stream',return_value=NS(cuda_stream=777)):
            out=adapter.NativeRegions.attention_native(regions,q,q,q,pre,pre)
        self.assertEqual(tuple(out.shape),(1,3,32,128))
        pack=[a for kind,a in seen if kind=='pack'][0];attn=[a for kind,a in seen if kind=='attn'][0]
        self.assertEqual(len(pack),11);self.assertEqual(pack[-1],777)
        self.assertEqual(len(attn),20);self.assertEqual(attn[-2],777);self.assertEqual(attn[-1],0)   # (..., sm_scale, stream, variant)
        self.assertEqual((attn[5],attn[6],attn[7],attn[8]),(3,3,2,32))
        self.assertAlmostEqual(attn[17],1/128**0.5)


class Fp8Contract(unittest.TestCase):
    """Candidate fp8 GEMM path: off unless explicitly requested; call sequence and artifact checks without a GPU."""
    def _model(self):
        weight=torch.ones(128,dtype=torch.bfloat16); norm=NS(weight=weight,bias=None)
        block=NS(attn=NS(processor=adapter.QwenImage21AttnProcessor(),norm_q=norm,norm_k=norm,set_processor=lambda p:None),
                 img_norm1=torch.nn.LayerNorm(4096,elementwise_affine=False),img_norm2=torch.nn.LayerNorm(4096,elementwise_affine=False),forward=lambda *a,**k:None)
        return NS(config=NS(num_attention_heads=32,attention_head_dim=128,num_layers=32,mlp_ratio=3),transformer_blocks=[block])

    def test_fp8_is_off_unless_directories_are_given(self):
        class Regions:
            def __init__(self,*a):self.normfuse_library=None;self.counts={}
        with mock.patch.object(adapter,'qualified_upstream',return_value=True), mock.patch.object(adapter,'NativeRegions',Regions):
            native=adapter.install(self._model(),'regions','attention','normfuse')
        self.assertIsNone(native.fp8)

    def test_fp8_directories_construct_the_candidate_regions(self):
        class Regions:
            def __init__(self,*a):self.normfuse_library=None;self.counts={}
        built=[]
        from paiton_image21 import fp8_regions
        class Fake:
            active=False
            def __init__(self,*dirs,**config):built.append(dirs or tuple(sorted(config)))
        with mock.patch.object(adapter,'qualified_upstream',return_value=True), mock.patch.object(adapter,'NativeRegions',Regions), mock.patch.object(fp8_regions,'Fp8Regions',Fake):
            native=adapter.install(self._model(),'regions','attention','normfuse',('gemm','quant','unpack'))
        self.assertEqual(built,[('gemm','quant','unpack')]);self.assertIsInstance(native.fp8,Fake)

    def test_fp8_feed_forward_call_sequence(self):
        calls=[]
        class Fp8:
            active=True
            def quantize(self,x,y=None,activation=0,int8=False):calls.append(('quantize',x.shape,None if y is None else y.shape,activation));return ('codes',x.shape[-1]),'scale'
            def linear(self,module,codes,scale):calls.append(('linear',module.name,codes,scale));return torch.zeros((x_rows,module.out),dtype=torch.bfloat16)
        x_rows=5
        mlp=NS(gate_layer=NS(name='gate',out=12288),proj=NS(name='proj',out=12288),out=NS(name='out',out=4096))
        regions=NS(fp8=Fp8(),counts={'silu':0})
        x=torch.zeros((1,x_rows,4096),dtype=torch.bfloat16)
        out=adapter.NativeRegions.feed_forward(regions,mlp,x)
        self.assertEqual(tuple(out.shape),(1,x_rows,4096))
        self.assertEqual([c[0]+':'+str(c[1]) for c in calls],['quantize:torch.Size([1, 5, 4096])','linear:gate','linear:proj','quantize:torch.Size([5, 12288])','linear:out'])
        self.assertEqual(calls[3][3],2)   # fused silu(gate)*up quantization
        self.assertEqual(regions.counts['silu'],1)

    def test_fp8_corrupt_artifact_rejected_before_dlopen(self):
        from paiton_image21 import fp8_regions
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'fixture.so').write_bytes(b'corrupt')
            (p/'manifest.json').write_text(json.dumps(dict(file='fixture.so',sha256='0'*64,architecture='gfx1201',abi_version=1)))
            with mock.patch.object(adapter.C,'CDLL',side_effect=AssertionError('must not load corrupt library')):
                with self.assertRaisesRegex(RuntimeError,'artifact identity mismatch'):
                    fp8_regions.Fp8Regions(p,p,p)


class LowPrecisionSchedule(unittest.TestCase):
    """The per-forward decision of the candidate low-precision path, without loading any library."""
    def _regions(self, from_step=0, exact_prefix=True):
        from paiton_image21.fp8_regions import Fp8Regions
        r=Fp8Regions.__new__(Fp8Regions)
        r.from_step=from_step; r.exact_prefix=exact_prefix; r.forwards=0; r.active=False; r.schedule_active=False
        r.counts=dict(exact_forwards=0, low_precision_forwards=0)
        return r

    def _forward(self, r, mode, blocks=3):
        r.begin_transformer_forward()
        return [r.begin_forward(mode) for _ in range(blocks)]

    def test_prefix_pass_stays_exact_and_denoiser_forwards_are_low_precision(self):
        r=self._regions()
        self.assertEqual(self._forward(r,'extract'), [False]*3)
        self.assertEqual(self._forward(r,'cached'), [True]*3); self.assertEqual(self._forward(r,'cached'), [True]*3)
        self.assertEqual(r.counts, dict(exact_forwards=3, low_precision_forwards=6))

    def test_schedule_counts_transformer_forwards_not_blocks(self):
        r=self._regions(from_step=3)
        decisions=[self._forward(r,'extract')[0]]+[self._forward(r,'cached',blocks=32)[-1] for _ in range(4)]
        self.assertEqual(decisions, [False, False, False, True, True])   # transformer forwards 1-3 exact, 4 and 5 low precision

    def test_prefix_quantization_can_be_requested_explicitly(self):
        r=self._regions(exact_prefix=False)
        self.assertEqual(self._forward(r,'extract'), [True]*3)

    def test_formats_map_to_kernel_flags(self):
        from paiton_image21.fp8_regions import FORMATS
        self.assertEqual(FORMATS['fp8-row'], (False, 0)); self.assertEqual(FORMATS['int8-32'], (True, 32)); self.assertEqual(FORMATS['int8-256'], (True, 256))


class Qk8Routing(unittest.TestCase):
    """The int8-QK attention companion is used only on low-precision forwards; exact forwards keep the exact kernel."""
    def _run(self, active, has_qk8):
        calls=[]
        class Fp8:
            def __init__(self): self.active=active
            def quantize(self,x,y=None,activation=0,int8=None,block=None): return 'codes','scale'
            def linear(self,module,codes,scale): return torch.zeros((1,3,4096),dtype=torch.bfloat16).view(3,4096)
        regions=NS(fp8=Fp8(), counts={'fallback':0,'attention_native':0}, qk8_library=object() if has_qk8 else None,
                   norm_rope=lambda x,norm,freq: x,
                   attention_native_usable=lambda *a: True,
                   attention_native=lambda *a: (calls.append('exact') or torch.zeros((1,3,32,128),dtype=torch.bfloat16)),
                   attention_qk8=lambda *a: (calls.append('qk8') or torch.zeros((1,3,32,128),dtype=torch.bfloat16)))
        lin=lambda x: torch.zeros((1,3,4096),dtype=torch.bfloat16)
        attn=NS(to_q=lin,to_k=lin,to_v=lin,norm_q=None,norm_k=None,to_out=[lambda x: x, lambda x: x])
        processor=adapter.NativeAttentionProcessor(regions, NS(_attention_backend=None,_parallel_config=None))
        hidden=torch.zeros((1,3,4096),dtype=torch.bfloat16)
        with mock.patch.object(torch.Tensor,'is_cuda',new_callable=mock.PropertyMock,return_value=True), torch.no_grad():
            processor(attn, hidden, None, None, None, 'cached')
        return calls

    def test_low_precision_forward_uses_qk8_when_present(self):
        self.assertEqual(self._run(active=True, has_qk8=True), ['qk8'])

    def test_exact_forward_keeps_the_exact_kernel(self):
        self.assertEqual(self._run(active=False, has_qk8=True), ['exact'])

    def test_without_the_companion_low_precision_forwards_keep_the_exact_kernel(self):
        self.assertEqual(self._run(active=True, has_qk8=False), ['exact'])


class PrecisionProfiles(unittest.TestCase):
    def test_profile_table_is_explicit_and_exact_is_empty(self):
        from paiton_image21.runtime import ImageEngine
        p=ImageEngine.PRECISION_PROFILES
        self.assertEqual(p['exact'], {})
        self.assertEqual(p['schedule-int8']['PAITON_IMAGE21_LP'], 'int8-256'); self.assertEqual(p['schedule-int8']['PAITON_IMAGE21_LP_FROM_STEP'], '11')
        self.assertEqual(p['schedule-fp8']['PAITON_IMAGE21_LP'], 'fp8-row')
        self.assertEqual(set(p['exact-w64']), {'PAITON_IMAGE21_ATTENTION_W64'})
