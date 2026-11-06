"""timm/vit_huge_plus_patch16_dinov3.lvd1689m -> transformers DINOv3ViTModel folder (key rename + qkv split only;
tensor values untouched). Then check: HF(mapped) last_hidden_state == timm forward_features on a random image.
usage: python -m paiton_idol.convert_dinov3 SRC_DIR DST_DIR"""
import json, os, shutil, sys

import torch
from safetensors.torch import load_file, save_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

src, dst = sys.argv[1], sys.argv[2]
torch.set_num_threads(8)
t = load_file(f'{src}/model.safetensors')
cfg = DINOv3ViTConfig(hidden_size=1280, num_hidden_layers=32, num_attention_heads=20, intermediate_size=5120,
                      use_gated_mlp=True, num_register_tokens=4, patch_size=16, layerscale_value=1.0, hidden_act='silu',
                      rope_theta=100.0, query_bias=False, key_bias=False, value_bias=False)  # timm qkv has no bias
cfg.architectures = ['DINOv3ViTModel']
m = DINOv3ViTModel(cfg).eval()
keys = set(m.state_dict())
out = {'embeddings.cls_token': t.pop('cls_token'), 'embeddings.register_tokens': t.pop('reg_token'),
       'embeddings.patch_embeddings.weight': t.pop('patch_embed.proj.weight'),
       'embeddings.patch_embeddings.bias': t.pop('patch_embed.proj.bias'),
       'embeddings.mask_token': torch.zeros(1, 1, 1280)}  # training-only token, not in the timm export
pre = 'model.layer' if any(k.startswith('model.layer.') for k in keys) else 'layer'
for i in range(32):
    b, h = f'blocks.{i}.', f'{pre}.{i}.'
    q, k, v = t.pop(b + 'attn.qkv.weight').chunk(3, 0)
    out.update({h + 'attention.q_proj.weight': q, h + 'attention.k_proj.weight': k, h + 'attention.v_proj.weight': v})
    for a, c in [('attn.proj', 'attention.o_proj'), ('mlp.fc1_g', 'mlp.gate_proj'), ('mlp.fc1_x', 'mlp.up_proj'),
                 ('mlp.fc2', 'mlp.down_proj'), ('norm1', 'norm1'), ('norm2', 'norm2')]:
        for s in ('weight', 'bias'):
            out[h + c + '.' + s] = t.pop(b + a + '.' + s)
    out[h + 'layer_scale1.lambda1'] = t.pop(b + 'gamma_1')
    out[h + 'layer_scale2.lambda1'] = t.pop(b + 'gamma_2')
out['norm.weight'], out['norm.bias'] = t.pop('norm.weight'), t.pop('norm.bias')
assert not t, f'unmapped timm keys: {list(t)[:5]}'
assert set(out) == keys, (sorted(set(out) ^ keys)[:10])
m.load_state_dict(out, strict=True)

import timm
ref = timm.create_model('vit_huge_plus_patch16_dinov3', pretrained=False, num_classes=0).eval()
ref.load_state_dict(load_file(f'{src}/model.safetensors'), strict=True)
x = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(0))
with torch.no_grad():
    a, r = m(pixel_values=x).last_hidden_state, ref.forward_features(x)
err = (a - r).abs().max().item()
print('shape', tuple(a.shape), tuple(r.shape), 'max_abs_diff', err, 'ref_absmax', r.abs().max().item())
assert a.shape == r.shape and err < 1e-3, 'mapped HF model does not match timm'

os.makedirs(dst, exist_ok=True)
save_file({k: v.contiguous() for k, v in out.items()}, f'{dst}/model.safetensors')
cfg.save_pretrained(dst)
shutil.copy(f'{src}/LICENSE.md', dst)
json.dump({'source': 'timm/vit_huge_plus_patch16_dinov3.lvd1689m', 'max_abs_diff_vs_timm_fp32_256px': err},
          open(f'{dst}/PROVENANCE.json', 'w'))
print('OK', dst)
