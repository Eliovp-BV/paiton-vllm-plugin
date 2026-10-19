"""GPU exactness for the Ref2VA packed attention lengths and ref churn."""
import os
from pathlib import Path
import sys
import unittest

import torch

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE))

from paiton_video import refpack


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("PAITON_H3_ARTIFACT"),
                     "ROCm and PAITON_H3_ARTIFACT required")
class H3RefAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from paiton_video.projection import install_projections
        install_projections(os.environ["PAITON_H3_ARTIFACT"])

    def _assert_exact(self, tokens, seed):
        import comfy_kitchen as ck
        from paiton_video import fusions
        if fusions._attention is None:
            self.skipTest("Artifact predates H3 attention")
        torch.manual_seed(seed)
        q, k, v = [torch.randn(1, 56, tokens, 128, device="cuda", dtype=torch.bfloat16)
                   for _ in range(3)]
        quantized = ck.prequantize_int8_attention(q, k, v)
        expected = ck.int8_attention_from_prequantized(quantized)
        actual = fusions.attend(quantized.q, quantized.k, quantized.v,
                                quantized.q_scale, quantized.k_scale, quantized.v_scale)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        del q, k, v, quantized, expected, actual
        torch.cuda.empty_cache()

    def test_exact_at_reference_layout_lengths(self):
        one = refpack.build_packed_layout(
            56, 37, 30, 54, 207,
            [{"kind": "image", "latent_h": 48, "latent_w": 86}]).seq_len
        nine = refpack.build_packed_layout(
            56, 37, 30, 54, 207,
            [{"kind": "image", "latent_h": 48, "latent_w": 86}]*9).seq_len
        self.assertEqual((one, nine), (16487, 24743))
        for tokens in (one, 20000, nine, 36900, 42900, 45200, 49152):
            with self.subTest(tokens=tokens):
                self._assert_exact(tokens, 800+tokens)

    def test_fifty_ref_reinjection_steps_do_not_grow_the_allocator(self):
        import comfy_kitchen as ck
        from paiton_video import fusions
        tokens = 16487
        torch.manual_seed(900)
        q, k, v = [torch.randn(1, 56, tokens, 128, device="cuda", dtype=torch.bfloat16)
                   for _ in range(3)]
        quantized = ck.prequantize_int8_attention(q, k, v)
        expected = ck.int8_attention_from_prequantized(quantized)
        fusions.attend(quantized.q, quantized.k, quantized.v,
                       quantized.q_scale, quantized.k_scale, quantized.v_scale)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        for _ in range(50):
            actual = fusions.attend(quantized.q, quantized.k, quantized.v,
                                    quantized.q_scale, quantized.k_scale, quantized.v_scale)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            del actual
        torch.cuda.synchronize()
        self.assertLessEqual(torch.cuda.memory_allocated(), baseline)
        del q, k, v, quantized, expected
        torch.cuda.empty_cache()

    def test_graph_replay_at_one_reference_length(self):
        import comfy_kitchen as ck
        from paiton_video import fusions
        tokens = 16487
        torch.manual_seed(901)
        q, k, v = [torch.randn(1, 56, tokens, 128, device="cuda", dtype=torch.bfloat16)
                   for _ in range(3)]
        quantized = ck.prequantize_int8_attention(q, k, v)
        expected = ck.int8_attention_from_prequantized(quantized)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay = fusions.attend(quantized.q, quantized.k, quantized.v,
                                    quantized.q_scale, quantized.k_scale, quantized.v_scale)
        graph.replay()
        torch.testing.assert_close(replay, expected, atol=0, rtol=0)
        del q, k, v, quantized, expected
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
