#!/usr/bin/env python3
"""Measure complete HTTP responses; run against an otherwise idle worker."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

NEON = 'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement'
TEAPOT = 'A blue ceramic teapot on a wooden table, soft daylight, studio photograph'


def run(server, output, suite=False, edit_input=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with urlopen(server + "/health", timeout=5) as response:
        health = json.load(response)
    report = dict(status="running", health=health, requests=[],
                  timing="Request submission through complete HTTP body receipt; startup/download excluded")
    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    requests = [dict(prompt=NEON, size="2048x2048", seed=42) for _ in range(2)]
    if suite:
        requests += [
            dict(prompt=TEAPOT, size="1024x1024", seed=51),
            dict(prompt="A cute cartoon dragon sticker.", mode="rgba", size="2048x2048", seed=52),
            dict(prompt="Change the blue teapot to bright red. Keep the table and composition.",
                 mode="edit", size="1024x1024", seed=53),
            dict(prompt=TEAPOT, size="1024x1024", seed=51),
        ]
    try:
        for index, body in enumerate(requests):
            recorded = dict(body)
            endpoint = "/v1/images/generations"
            if body.get("mode") == "edit":
                data = Path(edit_input or output / "request-2.png").read_bytes()
                body["image_b64"] = base64.b64encode(data).decode()
                recorded["input_png_sha256"] = hashlib.sha256(data).hexdigest()
                endpoint = "/v1/images/edits"
            request = Request(server + endpoint, data=json.dumps(body).encode(),
                              headers={"Content-Type": "application/json"})
            start = time.perf_counter()
            try:
                with urlopen(request, timeout=900) as response:
                    payload = response.read()
            except HTTPError as error:
                (output / f"request-{index}-error.json").write_bytes(error.read())
                raise
            end = time.perf_counter()
            result = json.loads(payload)
            png = base64.b64decode(result["data"][0]["b64_json"], validate=True)
            (output / f"request-{index}.png").write_bytes(png)
            row = dict(index=index, request=recorded, http_body_seconds=end-start,
                       request_begin_monotonic=start, request_end_monotonic=end,
                       png_sha256=hashlib.sha256(png).hexdigest(), **result["metrics"])
            report["requests"].append(row)
            save()
            print(json.dumps(row), flush=True)
        report["repeat_png_identical"] = report["requests"][0]["png_sha256"] == report["requests"][1]["png_sha256"]
        if not report["repeat_png_identical"]:
            raise RuntimeError("Repeated identical requests produced different PNG bytes")
        if suite:
            report["aba_png_identical"] = report["requests"][2]["png_sha256"] == report["requests"][5]["png_sha256"]
            if not report["aba_png_identical"]:
                raise RuntimeError("A-B-A request isolation failed")
            alpha = report["requests"][3]["alpha_extrema"]
            if not (alpha[0] < 64 and alpha[1] > 192):
                raise RuntimeError("RGBA alpha range check failed")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8191")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", action="store_true", help="Also run RGBA, editing and mixed-request A-B-A checks")
    parser.add_argument("--edit-input", type=Path, help="Reuse identical PNG bytes for paired editing comparisons")
    args = parser.parse_args()
    run(args.server.rstrip("/"), args.output, args.suite, args.edit_input)
