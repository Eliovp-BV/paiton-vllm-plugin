#!/usr/bin/env python3
"""Small client for the server. Standard library only.

    client.py generate PROMPT [--size 2048x2048] [--seed 42] [--out image.png]
    client.py edit PROMPT --image ref.png [--size 2048x2048] [--seed 1234] [--out edit.png]
    client.py benchmark [--size 2048x2048] [--runs 3]
"""
import argparse
import base64
import json
from pathlib import Path
import time
import urllib.request
import uuid

BENCH_PROMPT = ("A photorealistic overhead photograph of a rustic farm table with a sourdough loaf, a copper bowl of "
                "vegetables, a cheese board and a glass of red wine, warm sunlight from the left.")


def post(url, body, content_type):
    request = urllib.request.Request(url, data=body, headers={"Content-Type": content_type})
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)


def generate(url, prompt, size, seed):
    return post(f"{url}/v1/images/generations", json.dumps(dict(prompt=prompt, size=size, seed=seed)).encode(), "application/json")


def edit(url, prompt, images, size, seed):
    boundary = f"paiton-{uuid.uuid4().hex}"
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in dict(prompt=prompt, size=size, seed=seed).items()]
    for path in images:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
                     f"Content-Type: application/octet-stream\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return post(f"{url}/v1/images/edits", b"".join(parts), f"multipart/form-data; boundary={boundary}")


def save(payload, out):
    Path(out).write_bytes(base64.b64decode(payload["data"][0]["b64_json"]))
    print(json.dumps(dict(out=str(out), **payload.get("metrics", {}))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("generate", "edit", "benchmark"))
    p.add_argument("prompt", nargs="?")
    p.add_argument("--url", default="http://127.0.0.1:30000")
    p.add_argument("--size", default="2048x2048")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image", type=Path, action="append")
    p.add_argument("--out", default="image.png")
    p.add_argument("--runs", type=int, default=3)
    a = p.parse_args()
    if a.command == "benchmark":
        for i in range(a.runs):
            t = time.time()
            m = generate(a.url, a.prompt or BENCH_PROMPT, a.size, a.seed + i)["metrics"]
            print(json.dumps(dict(run=i, size=a.size, client_s=round(time.time() - t, 3), **m)))
    elif not a.prompt:
        p.error("a prompt is required")
    elif a.command == "generate":
        save(generate(a.url, a.prompt, a.size, a.seed), a.out)
    elif not a.image:
        p.error("edit needs --image")
    else:
        save(edit(a.url, a.prompt, a.image, a.size, a.seed), a.out)


if __name__ == "__main__":
    main()
