"""Local generation, a small interface, and reproducible measurement."""
import argparse
import json
from pathlib import Path
import time
import torch
from .pipeline import load_pipeline, generate, download_model, MODEL, REVISION


def main():
    parser = argparse.ArgumentParser(description="FLUX.2 klein 4B on Radeon AI PRO R9700")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Download only the pinned quantized model")
    gen = sub.add_parser("generate", help="Generate a 1024-square image with four steps")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--output", type=Path, default=Path("/outputs/image.png"))
    gen.add_argument("--backend", choices=["paiton"], default="paiton")
    gen.add_argument("--count", type=int, default=1, help="Consecutive seeds in one loaded process")
    ui = sub.add_parser("serve", help="Keep the pipeline loaded in a local browser interface")
    ui.add_argument("--port", type=int, default=7860)
    ui.add_argument("--backend", choices=["paiton"], default="paiton")
    bench = sub.add_parser("benchmark", help="Two warmups and two measured full generations")
    bench.add_argument("--backend", choices=["paiton", "stock"], default="paiton")
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--prompt-index", type=int, choices=range(3), default=0)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--suite", action="store_true", help="Run all three retained prompts and seeds")
    args = parser.parse_args()
    if args.command == "download":
        print(download_model())
        return
    if args.command == "benchmark":
        from .benchmark import run_benchmark
        run_benchmark(args)
        return
    start = time.perf_counter()
    pipe = load_pipeline(args.backend)
    print(json.dumps({"model": MODEL, "revision": REVISION, "load_seconds": time.perf_counter()-start}), flush=True)
    if args.command == "serve":
        from .interface import serve
        serve(pipe, args.port)
        return
    if args.count < 1 or args.count > 100:
        parser.error("--count must be between 1 and 100")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.count):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = generate(pipe, args.prompt, args.seed + index)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        path = args.output if args.count == 1 else args.output.with_name(f"{args.output.stem}-{index}{args.output.suffix}")
        result.images[0].save(path)
        print(json.dumps({"file": str(path), "seed": args.seed+index, "generation_seconds": elapsed,
                          "includes_first_use_compilation": index == 0}), flush=True)


if __name__ == "__main__":
    main()
