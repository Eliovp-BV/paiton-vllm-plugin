"""python -m paiton_h3i [serve|prepare]: prepare the weights on first use, then serve the HTTP API."""
import argparse
import os
from pathlib import Path

from tokenizers import Tokenizer

from . import server, weights

APP = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(prog="paiton_h3i")
    parser.add_argument("command", nargs="?", default="serve", choices=("serve", "prepare"))
    parser.add_argument("--cache", type=Path, default=Path(os.environ.get("PAITON_CACHE_DIR", "/cache")))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PAITON_PORT_INTERNAL", "30000")))
    parser.add_argument("--te-resident", default=os.environ.get("PAITON_TE_RESIDENT", "12"),
                        help="text encoder layers kept on the card (the runner lowers it to what fits)")
    args = parser.parse_args()
    bundle = weights.prepare(args.cache)
    if args.command == "prepare":
        return
    lib = APP / "lib"
    runner = server.Runner([str(lib / "h3i_run"), "--weights", str(bundle / "dit"), "--libs", str(lib), "--te", str(bundle / "te"),
                            "--te-resident", args.te_resident, "--vae", str(bundle / "dec8.bin"), "--vae-enc", str(bundle / "enc.bin"),
                            "--vae-tiles", "64", "--serve"])
    tokenizer = Tokenizer.from_file(str(args.cache / "minimax-h3-image" / "tokenizer" / "tokenizer.json"))
    server.serve(server.Engine(runner, tokenizer), args.host, args.port)


if __name__ == "__main__":
    main()
