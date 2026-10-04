"""CPU unit test for the start-time fp8 warm-up planner: M list from the scheduler limits, layer discovery, disable switch."""
import sys
import types
import unittest

import torch

if "vllm" not in sys.modules:  # the planner only needs vllm's logger; run without vLLM installed
    vllm = types.ModuleType("vllm"); logger = types.ModuleType("vllm.logger")
    import logging
    logger.init_logger = lambda name: logging.getLogger(name)
    vllm.logger = logger; sys.modules["vllm"] = vllm; sys.modules["vllm.logger"] = logger

from paiton_vllm_plugin import fp8_warmup as fw


class _Cfg:
    def __init__(self, seqs, batched, k):
        self.scheduler_config = types.SimpleNamespace(max_num_seqs=seqs, max_num_batched_tokens=batched)
        self.speculative_config = types.SimpleNamespace(num_speculative_tokens=k) if k is not None else None


class _Lin(torch.nn.Module):
    def __init__(self, n, k, block):
        super().__init__()
        self.output_size_per_partition, self.input_size_per_partition = n, k
        self.quant_method = types.SimpleNamespace(block_quant=block)
        self.calls = []

    def forward(self, x):
        self.calls.append(tuple(x.shape)); return torch.zeros(x.shape[0], self.output_size_per_partition), None


class Test(unittest.TestCase):
    def test_m_values_from_scheduler(self):
        w = types.SimpleNamespace(vllm_config=_Cfg(8, 4096, 7))
        ms, m_decode, m_max = fw._m_values(w)
        self.assertEqual((m_decode, m_max), (64, 4096))
        self.assertEqual(ms[:64], list(range(1, 65)))
        self.assertEqual(ms[64:], [65, 80, 128, 129, 144, 256, 257, 272, 512, 513, 528, 1024, 2048, 4096])

    def test_m_values_without_speculation_and_small_batch(self):
        ms, m_decode, m_max = fw._m_values(types.SimpleNamespace(vllm_config=_Cfg(4, 100, None)))
        self.assertEqual((m_decode, m_max), (4, 100)); self.assertEqual(ms, [1, 2, 3, 4, 65, 80, 100])

    def test_layers_and_forwards(self):
        tgt = torch.nn.Module(); tgt.a = _Lin(24576, 4096, True); tgt.b = _Lin(4096, 4096, False)
        drafter = types.SimpleNamespace(model=torch.nn.Module(), other=3); drafter.model.c = _Lin(6144, 4096, True)
        w = types.SimpleNamespace(vllm_config=_Cfg(2, 32, 3), model_runner=types.SimpleNamespace(model=tgt, drafter=drafter),
                                  device=torch.device("cpu"), model_config=types.SimpleNamespace(dtype=torch.float32))
        layers = fw._fp8_block_layers(fw._models(w))
        self.assertEqual([(n, N, K) for n, _, N, K in layers], [("target.a", 24576, 4096), ("drafter.model.c", 6144, 4096)])
        orig = torch.cuda.synchronize; torch.cuda.synchronize = lambda *_: None
        try:
            r = fw.fp8_blockscale_warmup(w)
        finally:
            torch.cuda.synchronize = orig
        self.assertEqual(r["calls"], 2 * (8 + 1)); self.assertEqual([s[0] for s in tgt.a.calls], [1, 2, 3, 4, 5, 6, 7, 8, 32])
        self.assertEqual(tgt.b.calls, [])

    def test_disabled(self):
        import os
        os.environ["PAITON_FP8_WARMUP"] = "0"
        try:
            self.assertIsNone(fw.fp8_blockscale_warmup(types.SimpleNamespace()))
        finally:
            del os.environ["PAITON_FP8_WARMUP"]


if __name__ == "__main__":
    unittest.main()
