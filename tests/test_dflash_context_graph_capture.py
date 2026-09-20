import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from paiton_vllm_plugin.dflash import context_graph as a

class Tests(unittest.TestCase):
    def setup_capture(self,needed):
        events=[];a.QUALIFIED=True;a.RUNTIME=types.SimpleNamespace(captures=0)
        class Speculator:
            def capture(self):events.append('query-capture');return 'original'
        a.wrap(types.SimpleNamespace(__name__=a.SPECULATOR,DFlashSpeculator=Speculator))
        obj=Speculator();obj.hidden_states=list(range(16));obj.context_positions=list(range(16))
        def context(states,positions,slots):
            self.assertEqual(states,list(range(8)));self.assertEqual(positions,list(range(8)));self.assertIsNone(slots)
            events.append('context')
            if events.count('context')==needed:a.RUNTIME.captures=1
        obj.model=types.SimpleNamespace(precompute_and_store_context_kv=context)
        def validate():events.append('validated')
        monitor=types.SimpleNamespace(validate_cudagraph_capturing_enabled=validate)
        modules={'torch':types.SimpleNamespace(cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False)),
                 'vllm':types.ModuleType('vllm'),'vllm.compilation':types.SimpleNamespace(monitor=monitor)}
        return obj,events,modules,monitor
    def test_capture_follows_original_startup_only(self):
        for needed in (1,2):
            obj,events,modules,_=self.setup_capture(needed)
            with patch.dict(sys.modules,modules):self.assertEqual(obj.capture(),'original')
            self.assertEqual(events,['query-capture','validated']+['context']*needed)
    def test_disabled_capture_guard_is_not_bypassed(self):
        obj,events,modules,monitor=self.setup_capture(1)
        def disabled():raise RuntimeError('disabled')
        monitor.validate_cudagraph_capturing_enabled=disabled
        with patch.dict(sys.modules,modules),self.assertRaisesRegex(RuntimeError,'disabled'):obj.capture()
        self.assertEqual(events,['query-capture'])
    def test_missing_eligibility_stops_after_two_calls(self):
        obj,events,modules,_=self.setup_capture(3)
        with patch.dict(sys.modules,modules),self.assertRaisesRegex(RuntimeError,'No eligible'):obj.capture()
        self.assertEqual(events.count('context'),2)
if __name__=='__main__':unittest.main()
