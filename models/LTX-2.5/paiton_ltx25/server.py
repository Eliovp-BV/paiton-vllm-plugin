"""HTTP API for LTX-2.5 text-to-video and image-to-video with audio over the native runner.

POST /v1/videos/generations takes JSON (text to video, or image to video with a base64 "image") or multipart/form-data
with an "image" file, and answers with one H.264 + AAC mp4 as base64. One request runs at a time."""
import array
import base64
import binascii
import email.parser
import email.policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

MODEL_ID = "LTX-2.5"
MAX_BODY = 64 << 20
MAX_WIDTH, MAX_HEIGHT = 1920, 1088  # the runner's buffers are sized once for this box (largest size measured on 32 GB)
MIN_SIDE = 256
MAX_FRAMES = 121                   # 5 s at 24 fps, the length measured at both sizes
MAX_TOKENS, BOS = 1024, 2          # LTXGemmaTokenizer: at most 1024 ids, <bos> first, no padding (the encoder runs these rows)


class Runner:
    """The native runner (--serve 1) keeps the weights loaded and takes one request per stdin line."""

    def __init__(self, command):
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1)
        self._read_until(("READY",))   # printed once the weights are on the card

    def _read_until(self, ends):
        lines = []
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if not line.startswith("STEP"):
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
        return parse_request_line(next(line for line in lines if line.startswith("REQUEST ")))


def parse_request_line(line):
    """'REQUEST 0: 18.96 s | te 0.45 s1 5.89 ... write 0.17 | VRAM 24.64 GiB, peak 25.98 GiB | prev mux 0.00 s'
    -> {runner_s, te_s, s1_s, ..., vram_gib, vram_peak_gib}."""
    head, stages, vram = line.split("|")[:3]
    words, v = stages.split(), vram.replace(",", " ").split()
    metrics = {f"{k}_s": float(v) for k, v in zip(words[::2], words[1::2])}
    return dict(runner_s=float(head.split()[2]), **metrics, vram_gib=float(v[1]), vram_peak_gib=float(v[4]))


def token_ids(tokenizer, prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    ids = tokenizer.encode(prompt.strip()).ids[:MAX_TOKENS]
    return array.array("i", ids if ids and ids[0] == BOS else [BOS, *ids][:MAX_TOKENS])


class Engine:
    def __init__(self, runner, tokenizer):
        self.runner, self.tokenizer = runner, tokenizer
        self.busy = threading.Lock()

    def generate(self, prompt, size, frames, seed, image=None):
        width, height = _size(size)
        frames = _frames(frames)
        ids = token_ids(self.tokenizer, prompt)
        with self.busy, tempfile.TemporaryDirectory(prefix="ltx-") as tmp:
            t = Path(tmp)
            with (t / "ids.i32").open("wb") as f:
                ids.tofile(f)
            args = ["--height", str(height), "--width", str(width), "--frames", str(frames), "--seed", str(seed),
                    "--ids", str(t / "ids.i32"), "--out", str(t)]
            if image is not None:
                (t / "image").write_bytes(image)
                _check_image(t / "image")
                args += ["--image", str(t / "image")]
            started = time.time()
            metrics = self.runner.run(args)
            mp4 = (t / "out.mp4").read_bytes()
        metrics.update(wall_s=round(time.time() - started, 3), size=f"{width}x{height}", num_frames=frames, fps=24,
                       mode="i2v" if image is not None else "t2v", text_tokens=len(ids))
        return mp4, metrics


def _size(value):
    try:
        width, height = (int(v) for v in str(value).lower().split("x"))
    except ValueError:
        raise ValueError("size must be WIDTHxHEIGHT, for example 1280x704") from None
    # two stages: stage 1 runs at half size on the 32-pixel latent grid, so both sides are multiples of 64
    if width % 64 or height % 64 or min(width, height) < MIN_SIDE:
        raise ValueError(f"width and height must be multiples of 64 and at least {MIN_SIDE}")
    if width > MAX_WIDTH or height > MAX_HEIGHT:
        raise ValueError(f"size {width}x{height} is outside this build's {MAX_WIDTH}x{MAX_HEIGHT} limit (no portrait sizes)")
    return width, height


def _frames(value):
    frames = int(value)
    if frames % 8 != 1 or not 9 <= frames <= MAX_FRAMES:
        raise ValueError(f"num_frames must be 8k+1 between 9 and {MAX_FRAMES}")
    return frames


def _seed(value):
    seed = int(value)
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _check_image(path):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0:s=x", str(path)], capture_output=True, text=True)
    sides = probe.stdout.strip().split("x")
    if probe.returncode or len(sides) != 2 or min(int(s) for s in sides) < 32 or int(sides[0]) * int(sides[1]) > 64 << 20:
        raise ValueError("image must be a readable picture (PNG, JPEG, WebP) of at least 32x32 and at most 64 MP")


def _common(fields):
    if fields.get("response_format", "b64_json") != "b64_json":
        raise ValueError("response_format must be b64_json")
    if int(fields.get("n", 1)) != 1:
        raise ValueError("n must be 1")
    if fields.get("model", MODEL_ID) != MODEL_ID:
        raise ValueError(f"model must be {MODEL_ID}")


def _multipart(content_type, body):
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    if not message.is_multipart():
        raise ValueError("expected multipart/form-data")
    fields, image = {}, None
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        data = part.get_payload(decode=True) or b""
        if name == "image":
            image = data
        elif name:
            fields[name] = data.decode()
    return fields, image


def parse_request(content_type, body):
    """-> (prompt, size, num_frames, seed, image bytes or None)."""
    if content_type.startswith("multipart/form-data"):
        fields, image = _multipart(content_type, body)
    else:
        fields = json.loads(body)
        if not isinstance(fields, dict):
            raise ValueError("JSON body must be an object")
        image = fields.get("image")
        if image is not None:
            try:
                image = base64.b64decode(image.split(",", 1)[-1], validate=True)   # plain base64 or a data: URL
            except (binascii.Error, AttributeError):
                raise ValueError("image must be base64") from None
    _common(fields)
    return (fields.get("prompt"), fields.get("size", "1280x704"), fields.get("num_frames", MAX_FRAMES),
            _seed(fields.get("seed", 42)), image or None)


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
                if engine.runner.proc.poll() is not None:   # the runner died: report it so the health check fails
                    return self.reply(503, dict(status="runner exited", model=MODEL_ID))
                self.reply(200, dict(status="ready", busy=engine.busy.locked(), model=MODEL_ID))
            elif self.path == "/v1/models":
                self.reply(200, dict(object="list", data=[dict(id=MODEL_ID, object="model", owned_by="Lightricks")]))
            else:
                self.reply(404, dict(error="unknown endpoint"))

        def do_POST(self):
            if self.path != "/v1/videos/generations":
                return self.reply(404, dict(error="unknown endpoint"))
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError(f"request body must be 1 byte to {MAX_BODY >> 20} MiB")
                request = parse_request(self.headers.get("Content-Type", ""), self.rfile.read(length))
                mp4, metrics = engine.generate(*request)
                self.reply(200, dict(created=int(time.time()), data=[dict(b64_json=base64.b64encode(mp4).decode())], metrics=metrics))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self.reply(400, dict(error=str(error)))
            except RuntimeError as error:
                self.reply(500, dict(error=str(error)))

        def log_message(self, fmt, *args):
            print("%s %s" % (self.address_string(), fmt % args), file=sys.stderr, flush=True)

    server = ThreadingHTTPServer((host_addr, port), Handler)
    server.daemon_threads = True
    print(f"READY http://{host_addr}:{port}", flush=True)
    server.serve_forever()
