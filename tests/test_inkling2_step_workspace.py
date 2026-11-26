"""Inkling-2-Mini native sequencer: the workspace of the largest step exists before vLLM sizes the cache (no GPU).
c17r --fast died at a 2,030-row step: the workspace grew lazily after the cache had taken the room (lead, 26 Nov)."""
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch

from paiton_vllm_plugin.inkling2.amd import native_step

ROW_BYTES = 640_000  # the fake library's workspace per row


class Cache:
    def __init__(self, bound):
        self.kv_cache = torch.empty(4 if bound else 0)


def model(bound):
    attn = SimpleNamespace(_pass_cache=lambda p: ("layer", torch.empty(4 if bound else 0)))
    return SimpleNamespace(loop_core=range(6, 18), aux_hidden_state_layers=(),
                           layers=[SimpleNamespace(attn=attn, conv_states=[Cache(bound)])])


class StepWorkspace(unittest.TestCase):
    def setUp(self):
        lib = SimpleNamespace(PaitonInkling2StepWorkspaceBytes=lambda handle, tokens, flags: tokens * ROW_BYTES)

        def init(step, m):
            step.handle, step.moe_flags, step.visits = None, 0, None
            step.workspace = torch.empty(0, dtype=torch.uint8)
        self.context = SimpleNamespace(attn_metadata=None)
        vllm = ModuleType('vllm.forward_context')
        vllm.get_forward_context = lambda: self.context
        self.patches = [mock.patch.dict(sys.modules, {'vllm.forward_context': vllm}),
                        mock.patch.object(native_step, '_library', return_value=lib),
                        mock.patch.object(native_step.NativeStep, '__init__', init)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def run_forward(self, m, rows):
        return native_step.forward(m, torch.zeros(rows, dtype=torch.int64), torch.arange(rows), None)

    def test_profile_run_reserves_the_largest_step(self):
        m = model(bound=False)
        self.assertIsNone(self.run_forward(m, 2048))  # vLLM's profile run: no metadata, the Python loop runs it
        step = m._native_step
        self.assertEqual(step.workspace.numel(), 2048 * ROW_BYTES)  # allocated now, so the peak counts it
        before = step.workspace
        self.assertIsNone(self.run_forward(m, 512))  # a smaller dummy run neither shrinks nor reallocates
        self.assertIs(step.workspace, before)
        self.context.attn_metadata = {'layer': object()}  # metadata but no caches yet: still the Python loop
        self.assertIsNone(self.run_forward(m, 2030))
        self.assertIs(step.workspace, before)

    def test_bound_caches_bind_once_and_serve_without_growing(self):
        m = model(bound=True)
        self.assertIsNone(self.run_forward(m, 2048))  # profile run with caches already there (no metadata)
        step = m._native_step
        before = step.workspace
        self.context.attn_metadata = {'layer': object()}
        with mock.patch.object(native_step.NativeStep, 'bind', autospec=True,
                               side_effect=lambda s, _: setattr(s, 'visits', ())) as bind, \
                mock.patch.object(native_step.NativeStep, 'forward', autospec=True, return_value='out') as fwd:
            self.assertEqual(self.run_forward(m, 2030), 'out')
            self.assertEqual(self.run_forward(m, 7), 'out')
        self.assertEqual(bind.call_count, 1)
        self.assertEqual(fwd.call_count, 2)
        self.assertIs(step.workspace, before)  # the 2,030-row step found its workspace in place

    def test_forward_grows_through_reserve(self):
        step = native_step.NativeStep(None)
        step.reserve(100, 'cpu')
        self.assertEqual(step.workspace.numel(), 100 * ROW_BYTES)
        step.reserve(50, 'cpu')
        self.assertEqual(step.workspace.numel(), 100 * ROW_BYTES)


if __name__ == '__main__':
    unittest.main()
