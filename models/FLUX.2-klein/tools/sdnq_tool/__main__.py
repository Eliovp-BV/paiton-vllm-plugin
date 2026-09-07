# SPDX-License-Identifier: GPL-3.0-only
import argparse
from pathlib import Path
import sys
import json
import tempfile


def main():
    parser=argparse.ArgumentParser(description="Separate model conversion and stock measurement tools")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("convert")
    bench=sub.add_parser("benchmark")
    bench.add_argument("--backend",choices=["stock"],default="stock")
    bench.add_argument("--suite",action="store_true")
    bench.add_argument("--output",type=Path,required=True)
    bench.add_argument("--prompt-index",type=int,choices=range(3),default=0)
    bench.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if args.command=="benchmark":
        from .benchmark import run_benchmark
        run_benchmark(args)
    else:
        from .pipeline import download_model
        from .convert import main as convert
        output=Path("/models/flux2-klein-runtime")
        if (output/"conversion.json").is_file():
            metadata=json.loads((output/"conversion.json").read_text())
            if metadata.get("source_revision") != "45e9cc76cb70f84473ce5c6c2e2282d0ef3c6ecd" or metadata.get("format_version") != 1 or metadata.get("source_model") != "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic":
                raise RuntimeError("An incompatible tensor cache is present")
            from huggingface_hub.errors import LocalEntryNotFoundError
            try:
                download_model(local_files_only=True)
            except LocalEntryNotFoundError:
                print("Restoring the pinned source cache for the stock engine.")
                download_model()
            print("Pinned tensor cache already prepared.")
            return
        snapshot=download_model()
        output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():
            raise RuntimeError('The tensor cache is incomplete. Move or remove flux2-klein-runtime in the cache before retrying. The source download is preserved.')
        with tempfile.TemporaryDirectory(prefix='.flux2-conversion-',dir=output.parent) as staging:
            prepared=Path(staging)/'tensors'
            sys.argv=["convert","--snapshot",str(snapshot),"--output",str(prepared)]
            convert()
            prepared.rename(output)


if __name__=="__main__":
    main()
