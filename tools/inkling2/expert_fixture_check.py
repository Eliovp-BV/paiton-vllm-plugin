"""Check the bring-up experts against an expert-layer fixture (keys L<layer>_c<call>_{x,idx,w,yA,yB}).

Usage: python3 expert_fixture_check.py <model dir> <fixture.npz> [noise | native | route]
yA: bf16 expert inputs; yB: the trained input quantization. Weights are dequantized from the checkpoint and rounded
to bf16, as the PyTorch reference that made the fixture does. `noise` (for example 0.005) multiplies each input by
1 + noise * N(0, 1) first: how far the outputs move for input differences of that size.
"""
import json
import sys
import types

import numpy as np
import torch
from safetensors import safe_open

from paiton_vllm_plugin.inkling2.amd.experts_bringup import Inkling2BringupExperts
from paiton_vllm_plugin.inkling2.amd.experts_native import Inkling2NativeExperts

model_dir, fixture_path = sys.argv[1:3]
route = sys.argv[3:4] == ["route"]  # as native, and the library routes the rows itself (gate from the checkpoint)
native = route or sys.argv[3:4] == ["native"]  # the native library against the fixture's trained-arithmetic key (yB2, else yB)
noise = float(sys.argv[3]) if len(sys.argv) > 3 and not native else 0.0
torch.manual_seed(0)
cfg = types.SimpleNamespace(**json.load(open(f"{model_dir}/config.json"))["text_config"])
fixture = np.load(fixture_path)
worst = 0.0
STAGE = torch.empty(32 << 20, dtype=torch.uint8).pin_memory()


def to_card(tensor: torch.Tensor) -> torch.Tensor:
    """Upload through one pinned 32 MB buffer: large pageable copies cause KFD queue evictions on shared hosts."""
    out = torch.empty(tensor.shape, dtype=tensor.dtype, device="cuda:0")
    src, dst = tensor.contiguous().view(-1).view(torch.uint8), out.view(-1).view(torch.uint8)
    for i in range(0, src.numel(), STAGE.numel()):
        n = min(STAGE.numel(), src.numel() - i)
        STAGE[:n].copy_(src[i:i + n])
        dst[i:i + n].copy_(STAGE[:n])
    return out


with safe_open(f"{model_dir}/model-00001-of-00001.safetensors", "pt", device="cpu") as ckpt:
    for layer in sorted({int(k.split("_")[0][1:]) for k in fixture.files}):
        prefix = f"model.llm.layers.{layer}.mlp.experts."
        fmt = "nvfp4" if f"{prefix}w13_weight_amax" in ckpt.keys() else "mxfp8"
        keyB = next((k for k in ("yB3", "yB2") if any(f.endswith("_" + k) for f in fixture.files)), "yB")
        gate = types.SimpleNamespace(weight=to_card(ckpt.get_tensor(prefix[:-8] + "gate.weight")),
                                     bias=to_card(ckpt.get_tensor(prefix[:-8] + "gate.bias"))) if route else None
        for act_quant, key in ((True, keyB),) if native else ((False, "yA"), (True, "yB")):
            experts = (Inkling2NativeExperts(cfg, fmt) if native
                       else Inkling2BringupExperts(cfg, fmt, act_quant, True)).to("cuda:0")
            for name, param in experts.named_parameters():
                param.data = to_card(ckpt.get_tensor(prefix + name))
            if not native:
                experts.process_weights_after_loading()
            for call in sorted({k.split("_")[1] for k in fixture.files if k.startswith(f"L{layer}_")}):
                x, idx, w, ref = (torch.from_numpy(fixture[f"L{layer}_{call}_{n}"]).to("cuda:0")
                                  for n in ("x", "idx", "w", key))
                if noise:
                    x = x.float() * (1 + noise * torch.randn_like(x.float()))
                y = (experts(x.to(torch.bfloat16), gate=gate, route_scale=cfg.route_scale) if route
                     else experts(x.to(torch.bfloat16), idx, w)).float()
                rel = ((y - ref).norm() / ref.norm()).item()
                worst = max(worst, rel)
                print(f"layer {layer} {fmt} {call} {key}: rel diff {rel:.5f}  max abs {(y - ref).abs().max().item():.4f}"
                      f"  ref rms {ref.square().mean().sqrt().item():.4f}", flush=True)
            del experts
print("EXPERT FIXTURE CHECK DONE worst rel diff", round(worst, 5))
