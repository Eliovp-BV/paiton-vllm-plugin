"""Runs on CPU with a tiny random Kimi-K3.1 checkpoint: prompt-at-once logits equal token-by-token logits,
with the layers split over two devices.

    python3 tests/test_tiny.py /path/to/tiny-random/kimi-k3.1-pro
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kimi31_community.model import KimiK31  # noqa: E402

torch.manual_seed(0)
m = KimiK31(sys.argv[1], ["cpu", "cpu"])
m.layer_dev = ["cpu"] * m.n_layers
ids = torch.randint(0, m.c.vocab_size, (24,))
full, _ = m.forward(ids)
lg, st = m.forward(ids[:10])
for t in range(10, 24):
    lg, st = m.forward(ids[t:t + 1], st)
err = (lg - full).abs().max().item() / full.abs().max().item()
print("prompt-at-once vs token-by-token, last-position rel max err", err)
assert err < 1e-4, "FAIL"
print("PASS")
