"""HTTP API of the MiniMax-H3-Image examples (/v1/images/generations JSON, /v1/images/edits multipart) over the native
runner. The runner process keeps the weights loaded and takes one request per stdin line."""
import base64
import email.parser
import email.policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
from PIL import Image, UnidentifiedImageError
from tokenizers import Tokenizer

from . import host

MODEL_ID = "MiniMax-H3-Image"
MAX_BODY = 256 << 20
MAX_CANVAS = 4736 * 3552           # largest canvas measured on 32 GB (16.8 MP); the runner also refuses what does not fit
MAX_IMAGE_PIXELS = 64 << 20        # decoded reference size guard
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class Runner:
    def __init__(self, command):
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1)
        self.load = self._read_until(("READY",))   # the runner prints READY once the weights are on the card

    def _read_until(self, ends):
        lines = []
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            print(line, file=sys.stderr, flush=True)
            lines.append(line)
            if line.split(" ", 1)[0] in ends:
                return lines
        raise RuntimeError(f"the native runner exited (code {self.proc.wait()})")

    def run(self, args):
        self.proc.stdin.write(" ".join(args) + "\n")
        self.proc.stdin.flush()
        lines = self._read_until(("DONE", "ERROR"))
        if lines[-1].startswith("ERROR"):
            raise ValueError(lines[-1][6:])
        return {k: float(v) for line in lines if line.startswith("TIMING ") for k, v in (kv.split("=") for kv in line.split()[1:])}


class Engine:
    def __init__(self, runner, tokenizer):
        self.runner, self.tokenizer = runner, tokenizer
        self.busy = threading.Lock()

    def generate(self, prompt, size, seed, references=()):
        width, height = _size(size)
        if len(references) > 1:
            raise ValueError("this build takes one reference image per edit")   # shortcut: runner edit path has one reference
        ch, cw = host.canvas_size(height, width)
        if ch * cw > MAX_CANVAS:
            raise ValueError(f"size {width}x{height} needs a {cw}x{ch} canvas, above the {MAX_CANVAS:,} pixel limit of this build")
        lh, lw = ch // 16, cw // 16
        refs = [host.load_reference(image) for image in references]
        vision = [host.vision_pixels(vis) for _, vis in refs]
        ids = host.token_ids(self.tokenizer, prompt, [grid for _, grid in vision])
        ref_grids = [(vae.height // 16, vae.width // 16) for vae, _ in refs]
        target, eps, noise = host.request_noise(seed, lh, lw, ref_grids)
        with self.busy, tempfile.TemporaryDirectory(prefix="h3i-") as tmp:
            t = Path(tmp)
            ids.tofile(t / "ids.bin")
            ref_rows = sum(h // 2 * (w // 2) for h, w in ref_grids)   # the runner fills the reference rows itself
            np.concatenate([np.zeros((ref_rows, 96), np.float32), target]).tofile(t / "noise.bin")
            args = ["--lat", f"{lh}x{lw}", "--ids", str(t / "ids.bin"), "--noise", str(t / "noise.bin"), "--png", str(t / "out.png")]
            if refs:
                (vae, _), (pix, (gh, gw)) = refs[0], vision[0]
                pix.tofile(t / "pixels.bin")
                host.vae_pixels(vae).tofile(t / "image.bin")
                eps[0].tofile(t / "eps.bin")
                noise[0].tofile(t / "ref_noise.bin")
                args += ["--refs", f"{ref_grids[0][0]}x{ref_grids[0][1]}", "--edit-pixels", str(t / "pixels.bin"),
                         "--edit-grid", f"{gh}x{gw}", "--edit-image", str(t / "image.bin"),
                         "--edit-size", f"{vae.height // 16 * 16}x{vae.width // 16 * 16}",
                         "--edit-eps", str(t / "eps.bin"), "--edit-noise", str(t / "ref_noise.bin")]
            started = time.time()
            metrics = self.runner.run(args)
            png = (t / "out.png").read_bytes()
        if (ch, cw) != (height, width):
            buffer = io.BytesIO()
            host.unscale(Image.open(io.BytesIO(png)).convert("RGB"), height, width).save(buffer, "PNG", compress_level=1)
            png = buffer.getvalue()
        metrics.update(wall_s=round(time.time() - started, 3), canvas=f"{cw}x{ch}", size=f"{width}x{height}", text_tokens=int(ids.size))
        return png, metrics


def _size(value):
    try:
        width, height = (int(v) for v in str(value).lower().split("x"))
    except ValueError:
        raise ValueError("size must be WIDTHxHEIGHT") from None
    if width < 32 or height < 32:
        raise ValueError("width and height must be at least 32")
    return width, height


def _seed(value):
    seed = int(value)
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _common(fields):
    if fields.get("response_format", "b64_json") != "b64_json":
        raise ValueError("response_format must be b64_json")
    if fields.get("output_format", "png") != "png":
        raise ValueError("output_format must be png")
    if fields.get("generator_device", "cpu") != "cpu":
        raise ValueError("generator_device must be cpu")
    if int(fields.get("n", 1)) != 1:
        raise ValueError("n must be 1")
    if fields.get("model", MODEL_ID) != MODEL_ID:
        raise ValueError(f"model must be {MODEL_ID}")


def _multipart(content_type, body):
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    if not message.is_multipart():
        raise ValueError("expected multipart/form-data")
    fields, images = {}, []
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        data = part.get_payload(decode=True) or b""
        if name in ("image", "image[]"):
            images.append(Image.open(io.BytesIO(data)))
        elif name:
            fields[name] = data.decode()
    return fields, images


def serve(engine, host_addr, port):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            payload = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, dict(status="ready", busy=engine.busy.locked(), model=MODEL_ID))
            elif self.path == "/v1/models":
                self.reply(200, dict(object="list", data=[dict(id=MODEL_ID, object="model", owned_by="MiniMaxAI")]))
            else:
                self.reply(404, dict(error="unknown endpoint"))

        def do_POST(self):
            if self.path not in ("/v1/images/generations", "/v1/images/edits"):
                return self.reply(404, dict(error="unknown endpoint"))
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError(f"request body must be 1 byte to {MAX_BODY >> 20} MiB")
                body = self.rfile.read(length)
                if self.path.endswith("generations"):
                    fields, images = json.loads(body), []
                    if not isinstance(fields, dict):
                        raise ValueError("JSON body must be an object")
                else:
                    fields, images = _multipart(self.headers.get("Content-Type", ""), body)
                    if not images:
                        raise ValueError("an edit needs at least one image field")
                _common(fields)
                png, metrics = engine.generate(fields.get("prompt"), fields.get("size", "2048x2048"),
                                               _seed(fields.get("seed", 42)), images)
                self.reply(200, dict(created=int(time.time()), data=[dict(b64_json=base64.b64encode(png).decode())], metrics=metrics))
            except (ValueError, TypeError, json.JSONDecodeError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                self.reply(400, dict(error=str(error)))
            except RuntimeError as error:
                self.reply(500, dict(error=str(error)))

        def log_message(self, fmt, *args):
            print("%s %s" % (self.address_string(), fmt % args), file=sys.stderr, flush=True)

    server = ThreadingHTTPServer((host_addr, port), Handler)
    server.daemon_threads = True
    print(f"READY http://{host_addr}:{port}", flush=True)
    server.serve_forever()
