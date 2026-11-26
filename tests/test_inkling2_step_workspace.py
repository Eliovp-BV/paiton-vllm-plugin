"""Inkling-2-Mini native sequencer: the workspace of the largest step exists before vLLM sizes the cache (no GPU).
c17r --fast died at a 2,030-row step: the workspace grew lazily after the cache had taken the room (lead, 26 Nov)."""
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch

from paiton_vllm_plugin.inkling2.amd import native_step

ROW_BYTES = 640_000  # the fake library's workspace per row


def size(tokens):  # not monotone: a bump at 1,039 rows above the 2,048-row size (the worst case is not the largest step)
    return tokens * ROW_BYTES + (2_000 * ROW_BYTES if tokens == 1039 else 0)


WORST = 3_039 * ROW_BYTES + native_step.WORKSPACE_MARGIN


class Cache:
    def __init__(self, bound):
        self.kv_cache = torch.empty(4 if bound else 0)


def model(bound, max_step_rows=2048):
    attn = SimpleNamespace(_pass_cache=lambda p: ("layer", torch.empty(4 if bound else 0)))
    return SimpleNamespace(loop_core=range(6, 18), aux_hidden_state_layers=(), max_step_rows=max_step_rows,
                           layers=[SimpleNamespace(attn=attn, conv_states=[Cache(bound)])])


class StepWorkspace(unittest.TestCase):
    def setUp(self):
        lib = SimpleNamespace(PaitonInkling2StepWorkspaceBytes=lambda handle, tokens, flags: size(tokens))

        def init(step, m):
            step.handle, step.moe_flags, step.visits, step.worst_rows = None, 0, None, 0
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
        out = native_step.forward(m, torch.zeros(rows, dtype=torch.int64), torch.arange(rows), None)
        if out is None:  # model.py: the Python loop runs the step, then the reservation
            self.assertEqual(m._native_step.workspace.numel(), self.loop_sees)  # nothing new held during the loop
            native_step.reserve_after_loop(m, torch.arange(rows))
        return out

    loop_sees = 0

    def test_profile_run_reserves_the_largest_step(self):
        m = model(bound=False)
        self.assertIsNone(self.run_forward(m, 2048))  # vLLM's profile run: no metadata, the Python loop runs it
        step = m._native_step
        self.assertEqual(step.workspace.numel(), WORST)  # the worst size of 1 to 2,048 rows, allocated now: the peak counts it
        before, self.loop_sees = step.workspace, WORST
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

    def test_scheduler_maximum_wins_over_a_smaller_dummy_run(self):
        m = model(bound=False)
        self.assertIsNone(self.run_forward(m, 16))  # a dummy run of 16 rows still reserves the scheduler's largest step
        self.assertEqual(m._native_step.workspace.numel(), WORST)

    def test_default_chunk_grows_lazily(self):
        m = model(bound=False, max_step_rows=native_step.LAZY_STEP_ROWS)  # the launcher's default: as c17r
        self.assertIsNone(self.run_forward(m, 512))
        self.assertEqual(m._native_step.workspace.numel(), 0)  # nothing counted in the profile run
        m = model(bound=False, max_step_rows=native_step.LAZY_STEP_ROWS + 1)
        self.assertIsNone(self.run_forward(m, 513))
        self.assertEqual(m._native_step.workspace.numel(), 513 * ROW_BYTES + native_step.WORKSPACE_MARGIN)

    def test_forward_grows_through_reserve(self):
        step = native_step.NativeStep(None)
        step.reserve(100, 'cpu')  # a step's own size (forward), no margin
        self.assertEqual(step.workspace.numel(), 100 * ROW_BYTES)
        step.reserve(50, 'cpu')
        self.assertEqual(step.workspace.numel(), 100 * ROW_BYTES)
        step.reserve(1039, 'cpu', worst=True)
        self.assertEqual(step.workspace.numel(), 3_039 * ROW_BYTES + native_step.WORKSPACE_MARGIN)


if __name__ == '__main__':
    unittest.main()
