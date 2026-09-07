"""Separate SDNQ conversion process; output contains tensors, never source code.

This conversion tool is licensed GPL-3.0-only when distributed with SDNQ.
It never loads Paiton artifacts or compiler/runtime modules.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
import sdnq
from sdnq.loader import apply_sdnq_options_to_model
from sdnq.dequantizer import dequantize_sdnq_model
from diffusers import Flux2KleinPipeline
from safetensors.torch import save_file


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = time.perf_counter()
    pipe = Flux2KleinPipeline.from_pretrained(args.snapshot, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=False)
    encoder = dequantize_sdnq_model(pipe.text_encoder)
    # Do not retain the quantization-loader metadata in a plain BF16 encoder.
    if hasattr(encoder.config, "quantization_config"):
        del encoder.config.quantization_config
    encoder.save_pretrained(args.output / "text_encoder", max_shard_size="2GB")
    del encoder, pipe.text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    transformer = pipe.transformer
    apply_sdnq_options_to_model(transformer, use_quantized_matmul=True)
    quantized = {}
    state = {}
    quantized_prefixes = []
    for name, module in transformer.named_modules():
        d = getattr(module, "sdnq_dequantizer", None)
        if d is None:
            continue
        if (not d.use_quantized_matmul or not d.re_quantize_for_matmul or d.quantized_matmul_dtype != "int8"
                or d.use_hadamard or module.svd_up is not None or module.original_class.__name__ != "Linear"):
            raise ValueError(f"Unexpected quantized layer contract: {name}, {vars(d)}")
        weight, scale = d.re_quantize_matmul(module.weight, module.scale, zero_point=module.zero_point)
        if weight.dtype != torch.int8 or scale.dtype != torch.bfloat16:
            raise ValueError(f"Unexpected prepared tensor types for {name}: {weight.dtype}, {scale.dtype}")
        quantized[name] = {"shape": list(d.original_shape), "weight_dtype": "int8", "scale_dtype": "bfloat16", "bias": module.bias is not None}
        state[name + ".weight"] = weight.t().contiguous().cpu()
        state[name + ".scale"] = scale.flatten().contiguous().cpu()
        if module.bias is not None:
            state[name + ".bias"] = module.bias.detach().cpu().contiguous()
        quantized_prefixes.append(name + ".")
    for key, value in transformer.state_dict().items():
        if not any(key.startswith(prefix) for prefix in quantized_prefixes):
            state[key] = value.detach().cpu().contiguous()
    target = args.output / "transformer"
    target.mkdir()
    save_file(state, target / "diffusion_pytorch_model.safetensors")
    config = dict(transformer.config)
    config.pop("quantization_config", None)
    config.pop("_name_or_path", None)
    config.pop("_use_default_values", None)
    (target / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    (target / "int8_layers.json").write_text(json.dumps(quantized, indent=2) + "\n")
    for name in ("vae", "tokenizer", "scheduler"):
        shutil.copytree(args.snapshot / name, args.output / name, symlinks=False)
    shutil.copy2(args.snapshot / "model_index.json", args.output / "model_index.json")
    files = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(16*1024**2), b""):
                    digest.update(chunk)
            files[str(path.relative_to(args.output))] = {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}
    manifest = {"format_version": 1, "source_model": "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
                "source_revision": "45e9cc76cb70f84473ce5c6c2e2282d0ef3c6ecd", "sdnq": sdnq.__version__,
                "torch": torch.__version__, "quantized_layers": len(quantized), "files": files,
                "conversion_seconds": time.perf_counter()-start}
    (args.output / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k:v for k,v in manifest.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
