import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from paiton_vllm_plugin.dflash import verification as v
from paiton_vllm_plugin.dflash.configuration import Settings
from paiton_vllm_plugin.dflash.serve import prepare_server_args

@dataclass
class Result:
    req_ids: list
    draft_token_ids: list

class Tests(unittest.TestCase):
    def setUp(self):
        self.ids=[str(i) for i in range(8)]
        self.tokens=[[-1]*7 for _ in self.ids]
    def bind(self, cap=3, trace=False):
        result=Result(self.ids,self.tokens)
        class Runner:
            def take_draft_token_ids(self):return result
            def prepare_inputs(self,scheduler_output,batch_req_state,batch_desc):return scheduler_output
        m=types.SimpleNamespace(GPUModelRunner=Runner)
        with patch.object(v,'validate_source'):v.bind(m,cap,trace)
        instance=Runner();instance.speculative_config=types.SimpleNamespace(
            method='dflash',num_speculative_tokens=7,draft_sample_method='probabilistic')
        instance.num_speculative_steps=7
        instance.scheduler_config=types.SimpleNamespace(async_scheduling=False)
        instance.adaptive_verification=None
        instance.draft_tokens_handler=types.SimpleNamespace(draft_tokens_np=None)
        instance.speculator=types.SimpleNamespace(num_speculative_steps=7,num_query_per_req=8,
            draft_logits=types.SimpleNamespace(shape=(8,7,1000)))
        instance.req_states=types.SimpleNamespace(req_id_to_index={k:7-i for i,k in enumerate(self.ids)})
        return instance,result,m
    def test_both_caps_keep_request_identity_and_original_storage(self):
        for cap in (3,5):
            runner,original,_=self.bind(cap);before=copy.deepcopy(original)
            result=runner.take_draft_token_ids()
            self.assertEqual(result.req_ids,self.ids)
            self.assertEqual(result.draft_token_ids,[row[:cap] for row in self.tokens])
            self.assertEqual(original,before);self.assertIsNot(result,original)
            self.assertEqual(runner.num_speculative_steps,7)
            self.assertEqual(runner.speculator.draft_logits.shape,(8,7,1000))
    def test_other_batch_sizes_and_partial_or_token_bearing_lists_keep_identity(self):
        for count in (0,1,4,7,9):
            ids=[str(i) for i in range(count)];rows=[[-1]*7 for _ in ids]
            self.assertIs(v.prefix_for_eight(ids,rows,3),rows)
        for row in ([],[-1]*3,list(range(7)),[-2]*7):
            rows=copy.deepcopy(self.tokens);rows[1]=row
            self.assertIs(v.prefix_for_eight(self.ids,rows,3),rows)
    def test_malformed_request_mapping_is_rejected(self):
        for ids in (self.ids[:-1],['same']*8):
            with self.assertRaises(ValueError):v.prefix_for_eight(ids,self.tokens,3)
    def test_async_or_changed_draft_contract_is_rejected(self):
        runner,_,_=self.bind();runner.scheduler_config.async_scheduling=True
        with self.assertRaisesRegex(RuntimeError,'synchronous'):runner.take_draft_token_ids()
        runner.scheduler_config.async_scheduling=False;runner.num_speculative_steps=3
        with self.assertRaisesRegex(RuntimeError,'seven-token'):runner.take_draft_token_ids()
    def test_adaptive_verification_and_changed_trained_block_are_rejected(self):
        runner,_,_=self.bind();runner.adaptive_verification=object()
        with self.assertRaisesRegex(RuntimeError,'adaptive'):runner.take_draft_token_ids()
        runner.adaptive_verification=None;runner.speculator.num_query_per_req=4
        with self.assertRaisesRegex(RuntimeError,'seven-token'):runner.take_draft_token_ids()
    def test_persistent_probability_identity_and_shape_are_required(self):
        runner,_,_=self.bind()
        for mapping,shape in [({},(8,7,1000)),({k:0 for k in self.ids},(8,7,1000)),({k:i for i,k in enumerate(self.ids)},(8,3,1000))]:
            runner.req_states.req_id_to_index=mapping;runner.speculator.draft_logits=types.SimpleNamespace(shape=shape)
            with self.assertRaisesRegex(RuntimeError,'probability'):runner.take_draft_token_ids()
    def test_structured_outputs_keep_original_return(self):
        runner,original,_=self.bind();runner.draft_tokens_handler.draft_tokens_np=object()
        self.assertIs(runner.take_draft_token_ids(),original)
    def test_empty_original_request_batch_is_preserved(self):
        runner,original,_=self.bind();original.req_ids=[];original.draft_token_ids=[]
        self.assertIs(runner.take_draft_token_ids(),original)
    def test_scheduled_observation_preserves_original_batch(self):
        runner,_,_=self.bind()
        batch=types.SimpleNamespace(num_draft_tokens_per_req=[3]*8,num_tokens=32,num_tokens_after_padding=64)
        self.assertIs(runner.prepare_inputs(batch,None,None),batch)
        self.assertEqual(runner.take_draft_token_ids._paiton_counts['scheduled_capped'],1)
        batch.num_draft_tokens_per_req=[7]*8
        self.assertIs(runner.prepare_inputs(batch,None,None),batch)
        self.assertEqual(runner.take_draft_token_ids._paiton_counts['scheduled_capped'],1)
    def test_duplicate_bind_is_rejected(self):
        _,_,m=self.bind()
        with patch.object(v,'validate_source'),self.assertRaisesRegex(RuntimeError,'already'):v.bind(m,3)
    def test_source_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'worker/gpu').mkdir(parents=True);p=root/'worker/gpu/model_runner.py';p.write_bytes(b'one')
            m=types.SimpleNamespace(__file__=str(p))
            with patch.object(v,'validate_active_source'), patch.object(v,'SOURCES',{'worker/gpu/model_runner.py':hashlib.sha256(b'one').hexdigest()}):
                v.validate_source(m);p.write_bytes(b'two')
                with self.assertRaisesRegex(RuntimeError,'pinned'):v.validate_source(m)
    def test_trace_only_preserves_original_and_emits_no_request_ids(self):
        import io
        from contextlib import redirect_stdout
        runner,original,_=self.bind(0,True)
        original.req_ids=['private-request-'+str(i) for i in range(8)]
        output=io.StringIO()
        with redirect_stdout(output):
            self.assertIs(runner.take_draft_token_ids(),original)
            self.assertIs(runner.take_draft_token_ids(),original)
        self.assertEqual(output.getvalue().count('PAITON_DFLASH_VERIFY_TRACE'),1)
        self.assertNotIn('private-request-',output.getvalue())
        self.assertEqual(runner.take_draft_token_ids._paiton_counts['capped'],0)

    def test_effective_overlay_and_unexpected_overlay_are_checked(self):
        import json, linecache
        with tempfile.TemporaryDirectory() as tmp:
            site=Path(tmp); root=site/'vllm/v1'; runner=root/'worker/gpu/model_runner.py'
            runner.parent.mkdir(parents=True); runner.write_bytes(b'original')
            runtime=site/'paiton_runtime_compat'; overlay=runtime/'overlays/runner.py'
            overlay.parent.mkdir(parents=True); overlay.write_bytes(b'overlay')
            source_hash=hashlib.sha256(b'original').hexdigest()
            overlay_hash=hashlib.sha256(b'overlay').hexdigest()
            entries={str(runner):dict(overlay='overlays/runner.py',original_sha256=source_hash,overlay_sha256=overlay_hash)}
            manifest=runtime/'compat_manifest.json'; manifest.write_text(json.dumps({'files':entries}))
            module=types.SimpleNamespace(__file__=str(runner))
            linecache.cache[str(runner)]=(7,None,['overlay'],str(runner))
            with patch.object(v,'SOURCES',{'worker/gpu/model_runner.py':source_hash}), patch.object(v,'OVERLAYS',{'worker/gpu/model_runner.py':overlay_hash}):
                v.validate_active_source(module,root)
                overlay.write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError,'content changed'):v.validate_active_source(module,root)
                overlay.write_bytes(b'overlay');linecache.cache[str(runner)]=(7,None,['different'],str(runner))
                with self.assertRaisesRegex(RuntimeError,'runner source changed'):v.validate_active_source(module,root)
            with patch.object(v,'SOURCES',{'worker/gpu/model_runner.py':source_hash}), patch.object(v,'OVERLAYS',{}):
                with self.assertRaisesRegex(RuntimeError,'Unexpected'):v.validate_active_source(module,root)
            linecache.cache.pop(str(runner),None)

    def test_off_install_does_not_import_runner(self):
        with patch.object(v,'bind') as bind:
            v.install(0);bind.assert_not_called()
    def test_configuration_is_bounded_and_scoped(self):
        env={'PAITON_DFLASH_RERANK':'128','PAITON_DFLASH_DRAFT_SAMPLE_METHOD':'probabilistic'}
        self.assertEqual(Settings.from_environment({}).verify_cap,0)
        for cap in ('3','5'):self.assertEqual(Settings.from_environment(dict(env,PAITON_DFLASH_VERIFY_CAP=cap)).verify_cap,int(cap))
        for bad in ({'PAITON_DFLASH_VERIFY_CAP':'3'},dict(env,PAITON_DFLASH_VERIFY_CAP='4'),dict(env,PAITON_DFLASH_VERIFY_CAP='3',PAITON_DFLASH_BLOCK_CANDIDATES='16')):
            with self.assertRaises(ValueError):Settings.from_environment(bad)
    def test_server_rejects_async_before_importing_vllm(self):
        env={'PAITON_DFLASH_RERANK':'128','PAITON_DFLASH_DRAFT_SAMPLE_METHOD':'probabilistic','PAITON_DFLASH_VERIFY_CAP':'3'}
        args=['--speculative-config','{"method":"dflash","num_speculative_tokens":7}']
        for suffix in ([],['--async-scheduling'],['--no-async-scheduling','--async-scheduling=false']):
            with self.assertRaisesRegex(ValueError,'explicit'):prepare_server_args(args+suffix,env)
        self.assertIn('--no-async-scheduling',prepare_server_args(args+['--no-async-scheduling'],env))

if __name__=='__main__':unittest.main()
