"""Check the plugin's binding of the native attention library on the attention fixture.

Usage: python3 attention_binding_check.py <fixtures/attn dir> [hash,hash,hash,hash]
Feeds the fixture's projection outputs through `NativeAttention.write` and `.forward` with tensors shaped as vLLM
shapes them (cache `(blocks, 8, 16, 256)`, scattered pages, inputs as column views of one merged projection row).
Gates: relative L2 of the gated output against the fixture at most 2e-2 (the kernel piece's stated tolerance for
upstream's serving arithmetic against the PyTorch reference), a chunked second call (rows 1000 to 1279 after 1000
cached keys) bit equal to the rows of the full call, and, when given, the first 16 hex digits of sha256 of the
full-prefill output for layer 7 pass 1, 2 and layer 11 pass 1, 2 (the kernel piece's own harness prints them).
"""
import hashlib
import sys

import numpy as np
import torch

from paiton_vllm_plugin.inkling2.amd.native_attention import NativeAttention

directory = sys.argv[1]
hashes = sys.argv[2].split(",") if len(sys.argv) > 2 else []
fx, w = np.load(f"{directory}/attn_fixture.npz"), np.load(f"{directory}/attn_weights.npz")
dev, T, BLOCKS = "cuda:0", 1280, 96
bf16 = lambda a: torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16).to(dev)
ok, case = True, 0
for layer, local in ((7, True), (11, False)):
    for ps in (1, 2):
        g = lambda s: bf16(fx[f"L{layer}_pass{ps}_{s}"][0])
        # one merged row per token, as the projection gives it: q | r | gate | sink | pad
        merged = torch.cat((g("q_proj_linear_out0"), g("r_linear_out0"), g("o_gate_proj_out0"), g("sink_proj_out0"),
                            torch.zeros((T, 8), dtype=torch.bfloat16, device=dev)), 1).contiguous()
        q, r, gate, sinks, _ = merged.split([3072, 384, 3072, 24, 8], dim=-1)
        pre, gate_logits = g("o_proj_in_arg0").float(), g("o_gate_proj_out0").float()
        want = (pre * torch.sigmoid(gate_logits)).to(torch.bfloat16).float()
        attn = NativeAttention(local, 1024, 0.1, 8192.0)
        attn.prepare(bf16(w[f"L{layer}.relative_position_bias.proj"]), bf16(w[f"L{layer}.q_proj.norm.scale"]),
                     bf16(w[f"L{layer}.relative_position_bias.norm.scale"]))
        cache = torch.zeros((BLOCKS, 8, 16, 256), dtype=torch.bfloat16, device=dev)
        table = torch.randperm(BLOCKS, generator=torch.Generator().manual_seed(7))[: T // 16].int().to(dev)[None]
        slots = (table[0].long()[torch.arange(T, device=dev) // 16] * 16 + torch.arange(T, device=dev) % 16).contiguous()
        attn.write(cache, g("k_proj_sconv_out0"), g("v_proj_sconv_out0"), slots, T)
        out = torch.empty((T, 3072), dtype=torch.bfloat16, device=dev)
        i32 = lambda *v: torch.tensor(v, dtype=torch.int32, device=dev)
        attn.forward(cache, table, i32(T), i32(0, T), T, T, q, r, gate, sinks, out)
        torch.cuda.synchronize()
        rel = ((out.float() - want).norm() / want.norm()).item()
        digest = hashlib.sha256(out.view(torch.int16).cpu().numpy().view(np.uint16).tobytes()).hexdigest()[:16]
        part = torch.empty((280, 3072), dtype=torch.bfloat16, device=dev)
        attn.forward(cache, table, i32(T), i32(0, 280), 280, 280, q[1000:], r[1000:], gate[1000:], sinks[1000:], part)
        torch.cuda.synchronize()
        same = torch.equal(part.view(torch.int16), out[1000:].view(torch.int16))
        good = rel <= 2e-2 and same and (not hashes or digest == hashes[case])
        ok &= good
        print(f"L{layer} pass {ps}: {'PASS' if good else 'FAIL'} rel to fixture {rel:.2e}, chunk bit equal {same}, "
              f"sha256 {digest}" + (f" (want {hashes[case]})" if hashes else ""), flush=True)
        case += 1
print("ATTENTION BINDING CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
