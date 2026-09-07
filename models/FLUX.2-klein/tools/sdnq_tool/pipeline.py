# SPDX-License-Identifier: GPL-3.0-only
"""Pinned, fully resident Flux2 klein pipeline for the Radeon AI PRO R9700."""
from pathlib import Path
import os
import torch
from huggingface_hub import snapshot_download

MODEL = "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic"
REVISION = "45e9cc76cb70f84473ce5c6c2e2282d0ef3c6ecd"


def download_model():
    return Path(snapshot_download(
        MODEL, revision=REVISION, cache_dir=os.environ.get("HF_HUB_CACHE"), max_workers=2,
        allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json",
                        "text_encoder/model.safetensors", "transformer/config.json",
                        "transformer/diffusion_pytorch_model.safetensors", "vae/*"],
    ))


def load_pipeline(backend="stock", snapshot=None):
    import sdnq  # Registers the supported quantization loader.
    from sdnq.loader import apply_sdnq_options_to_model
    from sdnq.dequantizer import dequantize_sdnq_model
    from diffusers import Flux2KleinPipeline
    if backend != "stock":
        raise ValueError("This separate tool runs only stock")
    import torch._inductor.config as inductor_config
    inductor_config.freezing = os.environ.get("STOCK_WEIGHT_FREEZING", "0") == "1"
    torch.backends.cudnn.benchmark = os.environ.get("STOCK_CONV_AUTOTUNE", "0") == "1"
    gpu = torch.cuda.get_device_properties(0)
    if gpu.name != "AMD Radeon AI PRO R9700" or gpu.gcnArchName.split(":")[0] != "gfx1201":
        raise RuntimeError("This release is qualified for one Radeon AI PRO R9700, gfx1201")
    torch.set_num_threads(4)
    snapshot = download_model() if snapshot is None else Path(snapshot)
    pipe = Flux2KleinPipeline.from_pretrained(snapshot, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
    apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=False)
    pipe.text_encoder = dequantize_sdnq_model(pipe.text_encoder)
    if os.environ.get("STOCK_PREPARE_WEIGHTS", "1") == "1":
        from .prepare import prepare_transformer_weights
        prepare_transformer_weights(pipe.transformer)
    import diffusers.models.transformers.transformer_flux2 as flux2
    from sdnq.kernels.triton_atten import sdnq_triton_atten

    def dispatch(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if attn_mask is not None or dropout_p or is_causal:
            raise ValueError("The qualified profile requires unmasked non-causal attention")
        return sdnq_triton_atten(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                                scale=scale, matmul_dtype="int8", pv_matmul_dtype="disabled").transpose(1, 2)
    flux2.dispatch_attention_fn = dispatch
    pipe.transformer.set_attention_backend("native")
    pipe.vae.to(memory_format=torch.channels_last)
    with torch.inference_mode():
        pipe.text_encoder(input_ids=torch.zeros((1, 16), dtype=torch.long, device="cuda"),
                          output_hidden_states=True, use_cache=False)
    pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
    from .prompt_encoder import PromptEncoder
    pipe.text_encoder = torch.compile(PromptEncoder(pipe.text_encoder), mode="reduce-overhead", fullgraph=True)
    pipe.vae.decode = torch.compile(pipe.vae.decode, mode="reduce-overhead", fullgraph=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(pipe, prompt, seed=42, callback=None):
    if not prompt.strip() or len(prompt) > 8000:
        raise ValueError("Enter a prompt of 1 to 8000 characters")
    return pipe(prompt=prompt, height=1024, width=1024, num_inference_steps=4,
                guidance_scale=1.0, num_images_per_prompt=1, max_sequence_length=512,
                generator=torch.Generator(device="cuda").manual_seed(seed), callback_on_step_end=callback)
