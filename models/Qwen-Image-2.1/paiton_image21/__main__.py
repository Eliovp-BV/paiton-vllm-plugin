import argparse
import json
from pathlib import Path
import sys

def main():
    parser=argparse.ArgumentParser(description="Paiton Image 2.1: local packed image generation")
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--backend",choices=("native","reference"),default="native")
    parser.add_argument("--native-fusions",action="store_true",help="Enable the qualified gfx1201 BF16 regions; native backend only")
    parser.add_argument("--precision-profile",choices=("exact","exact-w64","schedule-int8","schedule-int8-full8","schedule-fp8"),default=None,
                        help="exact (default) or a candidate low-precision schedule; requires --native-fusions")
    commands=parser.add_subparsers(dest="command",required=True)
    generate=commands.add_parser("generate")
    generate.add_argument("--prompt",required=True)
    generate.add_argument("--mode",choices=("text-to-image","rgba","edit"),default="text-to-image")
    generate.add_argument("--image",type=Path)
    generate.add_argument("--size",type=int,choices=(1024,2048),default=2048)
    generate.add_argument("--seed",type=int,default=42)
    generate.add_argument("--output",type=Path,required=True)
    serve=commands.add_parser("serve")
    serve.add_argument("--host",default="127.0.0.1")
    serve.add_argument("--port",type=int,default=8191)
    args=parser.parse_args()
    from .runtime import ImageEngine,validate_request
    source=None
    if args.command=="generate":
        from PIL import Image
        if args.output.exists():
            parser.error("Output already exists; choose a new path")
        source=Image.open(args.image) if args.image else None
        validate_request(args.prompt,args.size,args.size,seed=args.seed,mode=args.mode,image=source)
    print("Verifying checkpoint files and loading the GPU pipeline...",file=sys.stderr,flush=True)
    engine=ImageEngine(args.model_dir,args.backend,native_fusions=args.native_fusions,precision_profile=args.precision_profile)
    print(f"Pipeline ready in {engine.load_seconds:.2f} seconds.",file=sys.stderr,flush=True)
    if args.command=="serve":
        from .server import serve
        serve(engine,args.host,args.port)
    else:
        image,png,report=engine.generate(args.prompt,width=args.size,height=args.size,seed=args.seed,mode=args.mode,image=source)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("xb") as f:
            f.write(png)
        args.output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(dict(image=str(args.output.resolve()),load_seconds=engine.load_seconds,**report),indent=2))

if __name__=="__main__":
    main()
