#!/usr/bin/env python3
"""Send a local image request using only the Python standard library."""

import argparse
import base64
import json
from pathlib import Path
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", default="http://127.0.0.1:8191")
    p.add_argument("--prompt", required=True)
    p.add_argument("--model", help="Optional API model ID; by default use the server's loaded model")
    p.add_argument("--mode", choices=("text-to-image", "rgba", "edit"), default="text-to-image")
    p.add_argument("--size", type=int, choices=(1024, 2048), default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image", type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/image.png"))
    args = p.parse_args()
    if args.output.exists():
        p.error("Output already exists; choose a new path")
    if args.mode == "edit" and (args.image is None or args.size != 1024):
        p.error("Editing requires --image and --size 1024")
    if args.mode != "edit" and args.image is not None:
        p.error("--image requires --mode edit")
    body = dict(prompt=args.prompt, mode=args.mode,
                size=f"{args.size}x{args.size}", seed=args.seed, n=1)
    if args.model is not None:
        body["model"] = args.model
    endpoint = "generations"
    if args.image is not None:
        body["image_b64"] = base64.b64encode(args.image.read_bytes()).decode()
        endpoint = "edits"
    request = urllib.request.Request(
        args.server.rstrip("/") + "/v1/images/" + endpoint,
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        p.exit(1, error.read().decode() + "\n")
    png = base64.b64decode(result["data"][0]["b64_json"], validate=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(png)
    print(json.dumps(dict(image=str(args.output.resolve()), metrics=result.get("metrics")), indent=2))


if __name__ == "__main__":
    main()
