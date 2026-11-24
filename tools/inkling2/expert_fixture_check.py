"""Check the bring-up experts against an expert-layer fixture (keys L<layer>_c<call>_{x,idx,w,yA,yB}).

Usage: python3 expert_fixture_check.py <model dir> <fixture.npz> [noise]
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

model_dir, fixture_path = sys.argv[1:3]
noise = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
torch.manual_seed(0)
cfg = types.SimpleNamespace(**json.load(open(f"{model_dir}/config.json"))["text_config"])
fixture = np.load(fixture_path)
worst = 0.0
with safe_open(f"{model_dir}/model-00001-of-00001.safetensors", "pt", device="cuda:0") as ckpt:
    for layer in sorted({int(k.split("_")[0][1:]) for k in fixture.files}):
        prefix = f"model.llm.layers.{layer}.mlp.experts."
        fmt = "nvfp4" if f"{prefix}w13_weight_amax" in ckpt.keys() else "mxfp8"
        for act_quant, key in ((False, "yA"), (True, "yB")):
            experts = Inkling2BringupExperts(cfg, fmt, act_quant, True).to("cuda:0")
            for name, param in experts.named_parameters():
                param.data = ckpt.get_tensor(prefix + name)
            experts.process_weights_after_loading()
            for call in sorted({k.split("_")[1] for k in fixture.files if k.startswith(f"L{layer}_")}):
                x, idx, w, ref = (torch.from_numpy(fixture[f"L{layer}_{call}_{n}"]).to("cuda:0")
                                  for n in ("x", "idx", "w", key))
                if noise:
                    x = x.float() * (1 + noise * torch.randn_like(x.float()))
                y = experts(x.to(torch.bfloat16), idx, w).float()
                rel = ((y - ref).norm() / ref.norm()).item()
                worst = max(worst, rel)
                print(f"layer {layer} {fmt} {call} {key}: rel diff {rel:.5f}  max abs {(y - ref).abs().max().item():.4f}"
                      f"  ref rms {ref.square().mean().sqrt().item():.4f}", flush=True)
            del experts
print("EXPERT FIXTURE CHECK DONE worst rel diff", round(worst, 5))
