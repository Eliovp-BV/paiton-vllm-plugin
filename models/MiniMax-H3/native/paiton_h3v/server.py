"""HTTP API of MiniMax H3 Ref2VA v1.2 over the native runner: POST /v1/videos/generations (multipart or JSON) returns
the MP4. The runner keeps the weights of one preset loaded and takes one request at a time."""
import base64
import email.parser
import email.policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

from PIL import Image, UnidentifiedImageError

import front
import h3v

MODEL_ID = "MiniMax-H3-Ref2VA"
MAX_BODY = 512 << 20
MAX_REFS = 9                       # gated: up to 9 references (images, at most one of them a video)
MAX_PIXELS = 1344 * 768            # largest canvas the runner was gated on
MAX_FRAMES = 362                   # 15.1 s at 24 fps; longer requests are also refused by the runner's token cap
Image.MAX_IMAGE_PIXELS = 64 << 20  # decoded reference size guard


class Runner:
    def __init__(self, command):
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1)
        self._read_until(("READY",))   # the runner prints READY once the weights are loaded

    def _read_until(self, ends):
        lines = []
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            print(line, file=sys.stderr, flush=True)
            lines.append(line)
            if line.split(" ", 1)[0] in ends:
                return lines
        raise RuntimeError(f"the native runner exited (code {self.proc.wait()})")

    def run(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        lines = self._read_until(("DONE", "ERROR"))
        if lines[-1].startswith("ERROR"):
            raise ValueError(lines[-1][6:])
        return {k: v for line in lines if line.startswith(("TIMING ", "VRAM ")) for k, v in
                (kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)}


def _int(fields, name, default):
    try:
        return int(fields.get(name, default))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None


def check(fields, images, videos, preset):
    """Validate a request before anything touches the GPU; returns (prompt, width, height, frames, seed)."""
    prompt = fields.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if fields.get("model", MODEL_ID) != MODEL_ID:
        raise ValueError(f"model must be {MODEL_ID}")
    if fields.get("preset", preset) != preset:
        raise ValueError(f"this server runs preset {preset}; start another server for {fields['preset']}")
    if not 1 <= len(images) + len(videos) <= MAX_REFS:
        raise ValueError(f"send 1 to {MAX_REFS} references (image fields, plus at most one video)")
    if len(videos) > 1:
        raise ValueError("at most one video reference")
    width, height = _int(fields, "width", 864), _int(fields, "height", 480)
    if width % 32 or height % 32 or width < 256 or height < 256 or width * height > MAX_PIXELS:
        raise ValueError(f"width and height must be multiples of 32, at least 256, at most {MAX_PIXELS:,} pixels (1344x768)")
    if "seconds" in fields and "num_frames" not in fields:
        frames = max(1, round((float(fields["seconds"]) * 24 - 5) / 17)) * 17 + 5
    else:
        frames = _int(fields, "num_frames", 124)
    if frames % 17 != 5 or not 22 <= frames <= MAX_FRAMES:
        raise ValueError(f"num_frames must be 17 n + 5, from 22 to {MAX_FRAMES} (124 = 5.2 s, 362 = 15.1 s)")
    seed = _int(fields, "seed", 0)
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be a non-negative integer")
    return prompt, width, height, frames, seed


def parse(content_type, body):
    """multipart/form-data (image / image[] / video file fields + text fields) or JSON
    {"prompt": .., "images": [base64, ..], "video": base64, ..} -> (fields, image bytes, video bytes)."""
    if content_type.startswith("multipart/form-data"):
        message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
        fields, images, videos = {}, [], []
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            data = part.get_payload(decode=True) or b""
            if name in ("image", "image[]"):
                images.append(data)
            elif name == "video":
                videos.append(data)
            elif name:
                fields[name] = data.decode()
        return fields, images, videos
    fields = json.loads(body)
    if not isinstance(fields, dict):
        raise ValueError("JSON body must be an object")
    images = fields.pop("images", [])
    if not isinstance(images, list):
        raise ValueError("images must be a list of base64 strings")
    video = fields.pop("video", None)
    try:
        return fields, [base64.b64decode(i, validate=True) for i in images], [base64.b64decode(video, validate=True)] if video else []
    except (TypeError, ValueError):
        raise ValueError("images and video must be base64") from None


class Engine:
    def __init__(self, home, preset, extra):
        self.home, self.preset = home, preset
        self.runner = Runner(h3v.runner_cmd(home, preset, extra + ["--serve"]))
        self.busy = threading.Lock()

    def generate(self, fields, images, videos):
        prompt, width, height, frames, seed = check(fields, images, videos, self.preset)
        with tempfile.TemporaryDirectory(prefix="h3v-") as tmp:
            refs = []
            for i, data in enumerate(images):   # decode here so a bad file is a 400, re-saved as PNG for the front end
                path = os.path.join(tmp, f"ref{i}.png")
                Image.open(io.BytesIO(data)).convert("RGB").save(path)
                refs.append(path)
            for data in videos:
                path = os.path.join(tmp, "video.mp4")
                open(path, "wb").write(data)
                try:
                    readable = any(st.get("codec_type") == "video" for st in front.probe(path))
                except Exception:
                    readable = False
                if not readable:
                    raise ValueError("the video reference cannot be read by ffmpeg")
                refs.append(path)
            req = dict(prompt=prompt, refs=refs, width=width, height=height, num_frames=frames, seed=seed,
                       out=os.path.join(tmp, "clip.mp4"))
            with self.busy:
                started = time.time()
                try:
                    line = h3v.request_line(self.home, self.preset, req, os.path.join(tmp, "work"))
                except AssertionError as error:   # front-end checks, e.g. a soundtrack that is not 32 kHz
                    raise ValueError(str(error)) from None
                metrics = self.runner.run(line)
                metrics.update(wall_s=f"{time.time() - started:.3f}", size=f"{width}x{height}", num_frames=str(frames),
                               preset=self.preset, seed=str(seed))
            return open(req["out"], "rb").read(), metrics


def serve(engine, host_addr, port):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value, content_type="application/json", headers=()):
            payload = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            for key, val in headers:
                self.send_header(key, val)
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                alive = engine.runner.proc.poll() is None
                self.reply(200 if alive else 503, dict(status="ready" if alive else "runner exited", busy=engine.busy.locked(), model=MODEL_ID, preset=engine.preset,
                                     max_references=MAX_REFS))
            else:
                self.reply(404, dict(error="unknown endpoint"))

        def do_POST(self):
            if self.path != "/v1/videos/generations":
                return self.reply(404, dict(error="unknown endpoint"))
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError(f"request body must be 1 byte to {MAX_BODY >> 20} MiB")
                fields, images, videos = parse(self.headers.get("Content-Type", ""), self.rfile.read(length))
                mp4, metrics = engine.generate(fields, images, videos)
                self.reply(200, mp4, "video/mp4", [("X-Paiton-Metrics", json.dumps(metrics))])
            except (ValueError, TypeError, json.JSONDecodeError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                self.reply(400, dict(error=str(error)))
            except RuntimeError as error:
                self.reply(500, dict(error=str(error)))

        def log_message(self, fmt, *args):
            print("%s %s" % (self.address_string(), fmt % args), file=sys.stderr, flush=True)

    server = ThreadingHTTPServer((host_addr, port), Handler)
    server.daemon_threads = True
    print(f"READY http://{host_addr}:{port} preset {engine.preset}", flush=True)
    server.serve_forever()


def main(args):
    """http [--preset turbo4] [--port 30000] [--bind 0.0.0.0] [--runner-args "..."]; H3V_HOME is set by __main__."""
    import argparse
    ap = argparse.ArgumentParser(prog="http")
    ap.add_argument("--preset", default="turbo4", choices=tuple(front.PRESETS))
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--bind", default="0.0.0.0", help="inside the container; launch.sh publishes it on 127.0.0.1")
    ap.add_argument("--runner-args", default="")
    a = ap.parse_args(args)
    serve(Engine(os.environ["H3V_HOME"], a.preset, a.runner_args.split()), a.bind, a.port)
