#!/usr/bin/env python3
"""Launch the pinned GHCR release with verified target and DFlash2 snapshots."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def mount_path(path):
    value = str(path.resolve())
    if any(char in value for char in (":", "\n", "\r")):
        raise ValueError("Docker mount paths cannot contain colons or newlines")
    return value


def command(args, image):
    cmd = ["docker", "run", "--rm", "--name", args.name]
    if not args.download_only:
        cmd += ["--device", "/dev/kfd", "--device", "/dev/dri",
                "--group-add", "video", "--shm-size", "2g",
                "-p", f"127.0.0.1:{args.port}:8000"]
    if args.detach:
        cmd.append("-d")
    cache = mount_path(args.cache) if args.cache else "paiton-qwen38-mxfp4-cache"
    cmd += ["-v", f"{cache}:/models/cache"]
    entry_args = []
    for key in ("target", "draft"):
        path = getattr(args, key)
        if path is not None:
            dest = f"/models/{key}"
            cmd += ["-v", f"{mount_path(path)}:{dest}:ro"]
            entry_args += [f"--{key}", dest]
    if args.offline:
        entry_args.append("--offline")
    if args.download_only:
        entry_args.append("--download-only")
    return [*cmd, image, *entry_args]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, help="Optional existing pinned target snapshot")
    parser.add_argument("--draft", type=Path, help="Optional existing pinned DFlash2 snapshot")
    parser.add_argument("--cache", type=Path, help="Persistent cache directory; default is a Docker volume")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--name", default="paiton-qwen38-mxfp4")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use only cached or mounted model snapshots")
    parser.add_argument("--download-only", action="store_true", help="Download and verify models without GPU access")
    parser.add_argument("--dry-run", action="store_true", help="Print Docker argv without downloading or launching")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,90}", args.name):
        parser.error("Invalid port or container name")
    base = Path(__file__).resolve().parent
    runtime = json.loads((base / "runtime.lock.json").read_text())
    image = runtime.get("registry_image") or runtime["planned_registry_image"]
    try:
        cmd = command(args, image)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(cmd, indent=2))
        return
    if not re.fullmatch(r"ghcr\.io/eliovp/paiton-vllm-plugin@sha256:[0-9a-f]{64}", image):
        parser.error("The release image digest is not finalized in runtime.lock.json yet")
    for path in (args.target, args.draft):
        if path is not None and not path.is_dir():
            parser.error(f"Snapshot directory does not exist: {path}")
    if args.cache is not None:
        args.cache.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
