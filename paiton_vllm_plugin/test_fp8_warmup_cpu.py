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
        self.assertEqual(ms[:66], list(range(1, 65)) + [4096, 4095])        # decode rows, then the full chunk, first
        expected = set(range(1, 65)) | {b for b in range(16, 4097, 16)} | {min(4096, b + 1) for b in range(16, 4097, 16)} | {4096, 4095}
        self.assertEqual(set(ms), expected); self.assertEqual(len(ms), len(expected)); self.assertEqual(ms[66:], sorted(expected - set(ms[:66])))

    def test_m_values_without_speculation_and_small_batch(self):
        ms, m_decode, m_max = fw._m_values(types.SimpleNamespace(vllm_config=_Cfg(4, 100, None)))
        self.assertEqual((m_decode, m_max), (4, 100)); self.assertEqual(ms, [1, 2, 3, 4, 100, 99, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97])

    def test_layers_and_forwards(self):
        tgt = torch.nn.Module(); tgt.a = _Lin(24576, 4096, True); tgt.b = _Lin(4096, 4096, False)
        drafter = types.SimpleNamespace(model=torch.nn.Module(), other=3); drafter.model.c = _Lin(6144, 4096, True)
        w = types.SimpleNamespace(vllm_config=_Cfg(2, 32, 3), model_runner=types.SimpleNamespace(model=tgt, speculator=drafter),
                                  device=torch.device("cpu"), model_config=types.SimpleNamespace(dtype=torch.float32))
        layers = fw._fp8_block_layers(fw._models(w))
        self.assertEqual([(n, N, K) for n, _, N, K in layers], [("target.a", 24576, 4096), ("speculator.model.c", 6144, 4096)])
        orig = torch.cuda.synchronize; torch.cuda.synchronize = lambda *_: None
        try:
            r = fw.fp8_blockscale_warmup(w)
        finally:
            torch.cuda.synchronize = orig
        self.assertEqual(sorted(set(s[0] for s in tgt.a.calls)), [1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 31, 32]); self.assertEqual(r["calls"], 2 * 12)
        self.assertEqual(tgt.b.calls, [])

    def test_budget_keeps_decode_rows_and_chunk(self):
        import os, time
        tgt = torch.nn.Module(); tgt.a = _Lin(1024, 256, True)
        slow = tgt.a.forward
        def forward(x):
            time.sleep(0.002); return slow(x)
        tgt.a.forward = forward
        w = types.SimpleNamespace(vllm_config=_Cfg(2, 4096, 3), model_runner=types.SimpleNamespace(model=tgt),
                                  device=torch.device("cpu"), model_config=types.SimpleNamespace(dtype=torch.float32))
        os.environ["PAITON_FP8_WARMUP_BUDGET_S"] = "0.05"; os.environ.pop("TRITON_CACHE_DIR", None)
        try:
            r = fw.fp8_blockscale_warmup(w)
        finally:
            del os.environ["PAITON_FP8_WARMUP_BUDGET_S"]
        ms = [s[0] for s in tgt.a.calls]
        self.assertEqual(ms[:10], [1, 2, 3, 4, 5, 6, 7, 8, 4096, 4095]); self.assertIsNotNone(r["truncated_at"]); self.assertLess(r["calls"], r["planned"])
        # a seeded cache is reported but gets the same budget (a seed without the variants would otherwise cost minutes)
        import tempfile, pathlib
        d = tempfile.mkdtemp(); pathlib.Path(d, ".paiton-seed").write_text("x"); os.environ["TRITON_CACHE_DIR"] = d
        tgt.a.calls.clear(); os.environ["PAITON_FP8_WARMUP_BUDGET_S"] = "0.05"
        try:
            r = fw.fp8_blockscale_warmup(w)
        finally:
            del os.environ["PAITON_FP8_WARMUP_BUDGET_S"]; del os.environ["TRITON_CACHE_DIR"]
        self.assertIsNotNone(r["truncated_at"]); self.assertTrue(r["seeded"])
        # with a generous budget the sweep completes
        tgt.a.calls.clear(); os.environ["PAITON_FP8_WARMUP_BUDGET_S"] = "600"
        try:
            r = fw.fp8_blockscale_warmup(w)
        finally:
            del os.environ["PAITON_FP8_WARMUP_BUDGET_S"]
        self.assertIsNone(r["truncated_at"]); self.assertEqual(r["calls"], r["planned"])

    def test_disabled(self):
        import os
        os.environ["PAITON_FP8_WARMUP"] = "0"
        try:
            self.assertIsNone(fw.fp8_blockscale_warmup(types.SimpleNamespace()))
        finally:
            del os.environ["PAITON_FP8_WARMUP"]


if __name__ == "__main__":
    unittest.main()
