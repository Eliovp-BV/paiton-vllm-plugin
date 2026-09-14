"""Local H3 generation and reproducible qualification commands."""
import argparse
from pathlib import Path
import runpy
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["download","generate","benchmark","inspect","report"])
    parser.add_argument("--engine",choices=["stock","paiton"],default="paiton")
    parser.add_argument("--preset",choices=["base20","turbo8","turbo4"],default="turbo8")
    args,extra=parser.parse_known_args()
    if args.command in ("download","inspect","report"):
        module={"download":"download","inspect":"inspect_clip","report":"report_runs"}[args.command]
        sys.argv=[module,*extra]
        runpy.run_module("paiton_video."+module,run_name="__main__")
        return
    package=Path(__file__).resolve().parents[1]
    defaults=["--attention","ck","--fast-disk","--smart-memory",
              "--denoiser","minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
              "--video-vae","minimax_h3_video_vae_int8_convrot.safetensors",
              "--runs","3" if args.command=="benchmark" else "1"]
    if args.preset in ("turbo8","turbo4"):
        steps=args.preset.removeprefix("turbo")
        defaults += ["--lora",f"minimax_h3_fl2v_turbo_{steps}step_v1.0_768p_comfyui_bf16.safetensors",
                     "--lora-mode","bypass","--sampler","euler","--steps",steps,
                     "--shift-video","6","--shift-audio","3"]
    if args.engine=="paiton":
        defaults += ["--paiton-runtime",str(package),"--paiton-projections",
                     str(package/"artifacts/minimax_h3_projections_gfx1201.so")]
    sys.argv=["qualification",*defaults,*extra]
    runpy.run_module("paiton_video.qualification",run_name="__main__")


if __name__=="__main__":
    main()
