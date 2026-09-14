"""Independent Diffusers/PyTorch inference from converted tensor files."""
import json
import os
from pathlib import Path
import torch

MODEL = "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic"
REVISION = "45e9cc76cb70f84473ce5c6c2e2282d0ef3c6ecd"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def download_model():
    """Resolve the tensor cache prepared by the separate conversion tool."""
    model_dir = Path(os.environ.get("PAITON_MODEL_DIR", "/models/flux2-klein-runtime"))
    manifest = model_dir / "conversion.json"
    if not manifest.is_file():
        raise RuntimeError("Run ./run.sh download once to prepare the pinned tensor cache")
    data = json.loads(manifest.read_text())
    if data.get("format_version") != 1 or data.get("source_revision") != REVISION or data.get("source_model") != MODEL:
        raise RuntimeError("The converted tensor cache does not match this release")
    return model_dir


def load_pipeline(backend="paiton", snapshot=None):
    if backend != "paiton":
        raise ValueError("Use the separate stock container through ./run.sh benchmark --backend stock")
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
    from .projection import Int8Linear, install_projections
    from .attention import install_paiton_attention
    from .decoder import install_decoder
    from .prompt_encoder import PromptEncoder
    gpu = torch.cuda.get_device_properties(0)
    if gpu.name != "AMD Radeon AI PRO R9700" or gpu.gcnArchName.split(":")[0] != "gfx1201":
        raise RuntimeError("This release requires one Radeon AI PRO R9700, gfx1201")
    torch.set_num_threads(4)
    snapshot = download_model() if snapshot is None else Path(snapshot)
    directory = snapshot / "transformer"
    config = json.loads((directory / "config.json").read_text())
    layers = json.loads((directory / "int8_layers.json").read_text())
    with init_empty_weights(include_buffers=True):
        transformer = Flux2Transformer2DModel.from_config(config).to(dtype=torch.bfloat16)
        for name, descriptor in layers.items():
            if descriptor["bias"] or descriptor["weight_dtype"] != "int8" or descriptor["scale_dtype"] != "bfloat16":
                raise ValueError("Unexpected converted linear layer contract")
            parent_name, child = name.rsplit(".", 1)
            parent = transformer.get_submodule(parent_name)
            setattr(parent, child, Int8Linear(*descriptor["shape"]))
    state = load_file(directory / "diffusion_pytorch_model.safetensors", device="cuda")
    transformer.load_state_dict(state, strict=True, assign=True)
    del state
    transformer.eval()
    pipe = Flux2KleinPipeline.from_pretrained(snapshot, transformer=transformer,
                                             torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    install_projections(ARTIFACT_DIR / "flux2_klein_projections_gfx1201.so")
    for module in pipe.transformer.modules():
        if isinstance(module,Int8Linear):
            module.prepare()
    install_paiton_attention(ARTIFACT_DIR / "flux2_klein_gfx1201.so")
    pipe.transformer.set_attention_backend("native")
    pipe.vae.to(memory_format=torch.channels_last)
    install_decoder(pipe.vae,ARTIFACT_DIR / 'flux2_klein_decoder_gfx1201.so')
    with torch.inference_mode():
        pipe.text_encoder(input_ids=torch.zeros((1,16),dtype=torch.long,device="cuda"),
                          output_hidden_states=True,use_cache=False)
    pipe.transformer = torch.compile(pipe.transformer,mode="reduce-overhead",fullgraph=True)
    pipe.text_encoder = torch.compile(PromptEncoder(pipe.text_encoder),mode="reduce-overhead",fullgraph=True)
    pipe.vae.decode = torch.compile(pipe.vae.decode,mode="reduce-overhead",fullgraph=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(pipe, prompt, seed=42, callback=None):
    if not prompt.strip() or len(prompt) > 8000:
        raise ValueError("Enter a prompt of 1 to 8000 characters")
    return pipe(prompt=prompt,height=1024,width=1024,num_inference_steps=4,guidance_scale=1.0,
                num_images_per_prompt=1,max_sequence_length=512,
                generator=torch.Generator(device="cuda").manual_seed(seed),callback_on_step_end=callback)
