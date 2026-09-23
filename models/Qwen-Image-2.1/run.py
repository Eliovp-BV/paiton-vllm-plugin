#!/usr/bin/env python3
"""Prepare the pinned model, then run the isolated image engine."""

import argparse
import os
from pathlib import Path
import sys


PRECISION_PROFILES = ("exact", "exact-w32", "schedule-int8", "schedule-int8-11", "schedule-fp8")
DEFAULT_PROFILE = "schedule-int8"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model-dir", type=Path, help="Verify and reuse an existing published checkpoint")
    parser.add_argument("--cache-dir", type=Path, default=os.environ.get("HF_HUB_CACHE"))
    parser.add_argument("--offline", action="store_true", help="Require already cached files")
    parser.add_argument("--download-only", action="store_true", help="Fetch and verify without importing the GPU runtime")
    parser.add_argument("--no-native-fusions", action="store_true", help="Use the qualified unpack-only fallback")
    commands = parser.add_subparsers(dest="command")
    serve = commands.add_parser("serve", help="Run the image HTTP API (default)")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8191)
    generate = commands.add_parser("generate", help="Write one PNG directly")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--mode", choices=("text-to-image", "rgba", "edit"), default="text-to-image")
    generate.add_argument("--image", type=Path)
    generate.add_argument("--size", type=int, choices=(1024, 2048), default=2048)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output", type=Path, default=Path("/outputs/image.png"))
    # Accept preparation options after the command too, including through the
    # generate-docker helper. Suppressed defaults preserve options before it.
    for command_parser in (serve, generate):
        for flag in ("--offline", "--download-only", "--no-native-fusions"):
            command_parser.add_argument(flag, action="store_true", default=argparse.SUPPRESS)
        command_parser.add_argument("--model-dir", type=Path, default=argparse.SUPPRESS)
        command_parser.add_argument("--cache-dir", type=Path, default=argparse.SUPPRESS)
        command_parser.add_argument("--precision-profile", choices=PRECISION_PROFILES, default=argparse.SUPPRESS,
                                    help="exact keeps every step bit-exact; schedule-int8 (default) runs steps 8-40 in low precision (measured 103.6 s warm)")
    args = parser.parse_args(argv)
    if args.command == "generate":
        if args.output.exists():
            parser.error("Output already exists; choose a new path")
        if args.mode == "edit" and (args.image is None or args.size != 1024):
            parser.error("Editing requires --image and --size 1024")
        if args.mode != "edit" and args.image is not None:
            parser.error("--image requires --mode edit")
        if args.image is not None and not args.image.is_file():
            parser.error("Input image does not exist")
    from paiton_image21.checkpoint import prepare

    checkpoint = prepare(args.model_dir, args.cache_dir, args.offline)
    if args.download_only:
        print(f"Verified checkpoint ready: {checkpoint}")
        return 0
    command = [sys.executable, "-u", "-m", "paiton_image21", "--model-dir", str(checkpoint)]
    if not args.no_native_fusions:
        command.append("--native-fusions")
        profile = getattr(args, "precision_profile", os.environ.get("PAITON_IMAGE21_PROFILE", DEFAULT_PROFILE))
        command += ["--precision-profile", profile]
    if args.command in (None, "serve"):
        command += ["serve", "--host", getattr(args, "host", "0.0.0.0"),
                    "--port", str(getattr(args, "port", 8191))]
    else:
        command += ["generate", "--prompt", args.prompt, "--mode", args.mode,
                    "--size", str(args.size), "--seed", str(args.seed), "--output", str(args.output)]
        if args.image is not None:
            command += ["--image", str(args.image)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        sys.exit(2)
