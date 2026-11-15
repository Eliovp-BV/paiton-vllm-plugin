#!/usr/bin/env python3
"""Small client for the server. Standard library only.

    client.py generate PROMPT [--size 1280x704] [--frames 121] [--seed 42] [--image first-frame.png] [--out video.mp4]
    client.py benchmark [--size 1280x704] [--runs 3]
"""
import argparse
import base64
import json
from pathlib import Path
import time
import urllib.request

BENCH_PROMPT = ("A slow dolly shot through a rainy neon-lit street market at night, steam rising from food stalls, "
                "people with umbrellas walking past, the sound of rain and distant chatter.")


def generate(url, prompt, size, frames, seed, image=None):
    body = dict(prompt=prompt, size=size, num_frames=frames, seed=seed)
    if image:
        body["image"] = base64.b64encode(Path(image).read_bytes()).decode()
    request = urllib.request.Request(f"{url}/v1/videos/generations", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("generate", "benchmark"))
    p.add_argument("prompt", nargs="?")
    p.add_argument("--url", default="http://127.0.0.1:30000")
    p.add_argument("--size", default="1280x704")
    p.add_argument("--frames", type=int, default=121)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image")
    p.add_argument("--out", default="video.mp4")
    p.add_argument("--runs", type=int, default=3)
    a = p.parse_args()
    if a.command == "benchmark":
        for i in range(a.runs):
            t = time.time()
            m = generate(a.url, a.prompt or BENCH_PROMPT, a.size, a.frames, a.seed + i, a.image)["metrics"]
            print(json.dumps(dict(run=i, client_s=round(time.time() - t, 3), **m)), flush=True)
    elif not a.prompt:
        p.error("a prompt is required")
    else:
        payload = generate(a.url, a.prompt, a.size, a.frames, a.seed, a.image)
        Path(a.out).write_bytes(base64.b64decode(payload["data"][0]["b64_json"]))
        print(json.dumps(dict(out=a.out, **payload.get("metrics", {}))))


if __name__ == "__main__":
    main()
