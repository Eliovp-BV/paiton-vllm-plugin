"""One model-loading policy for the Comfy workflows and terminal runner."""
from pathlib import Path


def load(preset,engine='stock',compile_blocks=True):
    import torch,nodes
    from comfy_extras.nodes_model_advanced import ModelSamplingSD3
    if preset not in ('base','fast') or engine not in ('stock','paiton'):
        raise ValueError('Unknown Wan preset or engine')
    filename='fastwan22_5b_fullattn_comfy_bf16.safetensors' if preset=='fast' else 'wan2.2_ti2v_5B_fp16.safetensors'
    clip=nodes.CLIPLoader().load_clip('umt5_xxl_fp8_e4m3fn_scaled.safetensors','wan','default')[0]
    vae=nodes.VAELoader().load_vae('wan2.2_vae.safetensors')[0]
    model=nodes.UNETLoader().load_unet(filename,'default')[0]
    model=ModelSamplingSD3().patch(model,8)[0]
    if compile_blocks:
        for block in model.model.diffusion_model.blocks:
            block.forward=torch.compile(block.forward,fullgraph=False,dynamic=False)
    if engine=='paiton':
        from .vae import install
        vae=install(vae,Path(__file__).resolve().parents[1]/'artifacts/wan22_ti2v_fusions_gfx1201.so')
    return model,clip,vae
