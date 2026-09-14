# SPDX-License-Identifier: Apache-2.0
# Modified from the pinned FastVideo DMD implementation for ComfyUI.
"""FastWan FullAttn's text-only three-step DMD sampler for Comfy Wan models.

Algorithm: FastVideo's Apache-2.0 Wan DmdDenoisingStage and
pred_noise_to_pred_video. The training table has shift 8, independently of
inference configuration shift 5. This is not a three-step Euler schedule.
"""
import torch

TIMESTEPS = (1000,757,522)


def training_table(device='cpu'):
    # Identical to numpy.linspace(1,1000,1000,dtype=float32)[::-1].
    sigma=torch.arange(1000,0,-1,dtype=torch.float32,device=device)/1000
    sigma=8*sigma/(1+7*sigma)
    return sigma*1000,sigma


def sample(model,conditioning,seed,width,height,frames,callback=None):
    import comfy.model_management as mm
    if width%32 or height%32 or (frames-1)%4:
        raise ValueError('FastWan requires dimensions divisible by 32 and 4n+1 frames')
    device=mm.get_torch_device()
    mm.load_models_gpu([model],memory_required=3*1024**3)
    dm=model.get_model_object('diffusion_model')
    dtype=model.model.get_dtype_inference()
    if dtype!=torch.bfloat16:
        raise ValueError('The qualified FastWan DMD setting uses BF16')
    context=conditioning[0][0].to(device=device,dtype=dtype)
    # Preserve FastVideo's BTCHW CPU generator ordering and low-precision draws.
    shape=(1,(frames-1)//4+1,48,height//16,width//16)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    latents=torch.randn(shape,dtype=dtype,generator=generator).to(device)
    options=model.model_options.get('transformer_options',{}).copy()
    def predict(x,t):
        return dm(x,timestep=torch.tensor([t],dtype=torch.long,device=device),
                  context=context,transformer_options=options)
    latents=denoise(latents,predict,generator,callback)
    # Comfy's VAE receives raw latents, unlike its denoiser.
    return {'samples':model.model.process_latent_out(latents.permute(0,2,1,3,4).contiguous())}


def denoise(latents,predict,generator,callback=None):
    """Run the DMD update, retaining BTCHW noise order and FP64 clean prediction."""
    table,sigmas=training_table(latents.device)
    for i,t in enumerate(TIMESTEPS):
        if callback:callback(i,t)
        x=latents.permute(0,2,1,3,4).contiguous()
        pred=predict(x,t).permute(0,2,1,3,4)
        sigma=sigmas[(table.double()-t).abs().argmin()].double()
        clean=(latents.double()-sigma*pred.double()).to(pred.dtype)
        if i+1<len(TIMESTEPS):
            noise=torch.randn(latents.shape,dtype=clean.dtype,generator=generator).to(latents.device)
            sigma=sigmas[(table-TIMESTEPS[i+1]).abs().argmin()].reshape(1,1,1,1,1)
            latents=((1-sigma)*clean+sigma*noise).to(noise.dtype)
        else:
            latents=clean
    return latents
