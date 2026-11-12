"""Runs on CPU with a tiny random Kimi-K3.1 checkpoint: prompt-at-once logits equal token-by-token logits,
with the layers split over two devices; and the KDA decay scale A_log is read per head (padding entries ignored).

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

# KDA decay scale: one A_log value per head, in the first num_heads entries of the stored tensor; the rest is padding.
la = m.c.linear_attn_config
H, dk = la["num_heads"], la["head_dim"]
i = min(la["kda_layers"]) - 1
pre, key = f"layers.{i}.self_attn", f"{m.P}layers.{i}.self_attn.A_log"
a_log = m.w.get(key).float().clone()
assert a_log.numel() > H, "the check needs a padded A_log (more entries than heads)"
m.dev = "cpu"
x = torch.randn(12, m.c.hidden_size)
base = m.kda_attn(x, pre)[0]
pad = a_log.clone()
pad[H:] += 1.0                      # change only the padding
m.res[key] = pad
d_pad = (m.kda_attn(x, pre)[0] - base).abs().max().item()
head = a_log.clone()
head[0] += 1.0                      # change head 0's value
m.res[key] = head
d_head = (m.kda_attn(x, pre)[0] - base).abs().max().item()
del m.res[key]
print(f"A_log: output change from the padding entries {d_pad:.3e} (must be 0), from head 0's entry {d_head:.3e} (must be > 0)")
assert d_pad == 0.0, "FAIL: A_log entries past num_heads change the output (A_log is being read per channel)"
assert d_head > 0.0, "FAIL: A_log entry 0 does not reach the output"
print("PASS")
