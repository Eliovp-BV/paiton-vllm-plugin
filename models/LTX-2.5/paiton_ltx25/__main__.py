"""python -m paiton_ltx25 [serve|prepare]: download the weights on first use, then serve the HTTP API."""
import argparse
import os
from pathlib import Path

from tokenizers import Tokenizer

from . import server, weights

APP = Path(__file__).resolve().parent.parent
DIT_BF16 = "dit/ltx-2.5-22b-distilled-transformer-bf16.safetensors"   # the runner maps it for the few BF16 tensors left


def runner_command(lib, bundle):
    return [str(lib / "ltx_run"), "--mode", "gen", "--fps", "24", "--bundle", str(bundle / "dit"), "--dit", str(bundle / DIT_BF16),
            "--fp8", "1", "--gemm-so", str(lib / "ltx25_gemm_gfx1201.so"), "--rows-so", str(lib / "ltx25_rows_gfx1201.so"),
            "--attn-so", str(lib / "ltx25_attention_gfx1201.so"),
            "--te-so", str(lib / "ltx25_te_gfx1201.so"), "--te-dir", str(bundle / "te"),
            "--vae-so", str(lib / "ltx25_vae_gfx1201.so"), "--vae", str(bundle / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
            "--upsampler", str(bundle / "vae/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
            "--audio-so", str(lib / "ltx25_audio_gfx1201.so"), "--audio-vae", str(bundle / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
            "--k3-transient", "2", "--vram-cap-gib", os.environ.get("PAITON_VRAM_CAP_GIB", "27.5")]


def main():
    parser = argparse.ArgumentParser(prog="paiton_ltx25")
    parser.add_argument("command", nargs="?", default="serve", choices=("serve", "prepare"))
    parser.add_argument("--cache", type=Path, default=Path(os.environ.get("PAITON_CACHE_DIR", "/cache")))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PAITON_PORT_INTERNAL", "30000")))
    args = parser.parse_args()
    bundle = weights.prepare(args.cache)
    if args.command == "prepare":
        return
    runner = server.Runner(runner_command(APP / "lib", bundle))
    tokenizer = Tokenizer.from_file(str(bundle / "te" / "tokenizer.json"))
    server.serve(server.Engine(runner, tokenizer), args.host, args.port)


if __name__ == "__main__":
    main()
