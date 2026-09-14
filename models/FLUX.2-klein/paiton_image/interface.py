"""A local, single-user interface with no extra web framework."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs
from io import BytesIO
import base64
import html
import time
import torch
from .pipeline import generate

PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Paiton · FLUX.2 klein</title><style>
body{font:17px system-ui;max-width:880px;margin:40px auto;padding:0 24px;background:#111a1b;color:#edf6f3}
textarea,input,button{font:inherit;border-radius:8px;padding:12px;border:1px solid #708a82}
textarea{width:95%;height:100px;background:#20312e;color:inherit}button{background:#b5f0c5;cursor:pointer}
img{width:100%;border-radius:12px}small{color:#b5c9c1}form{margin:24px 0}h1{font-size:32px}</style>
<h1>Paiton · FLUX.2 klein</h1><p>Make a photograph, product concept or illustration on your Radeon.</p>
<small>1024 × 1024 · four steps · local generation. The first image compiles the pipeline.</small>
<form method="post"><label>Describe your image<br><textarea name="prompt" maxlength="8000" required>{prompt}</textarea></label>
<p><label>Seed <input name="seed" type="number" value="{seed}"></label> <button>Generate image</button></p></form>{result}</html>'''


def serve(pipe, port):
    class Handler(BaseHTTPRequestHandler):
        def render(self, prompt="", seed=42, result="", status=200):
            body = PAGE.replace("{prompt}", html.escape(prompt)).replace("{seed}", str(seed)).replace("{result}", result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.render()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length > 40000:
                self.render(result="<p>Prompt is too long.</p>", status=413)
                return
            values = parse_qs(self.rfile.read(length).decode())
            prompt = values.get("prompt", [""])[0]
            try:
                seed = int(values.get("seed", ["42"])[0])
                torch.cuda.synchronize()
                start = time.perf_counter()
                image = generate(pipe, prompt, seed).images[0]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                buffer = BytesIO()
                image.save(buffer, format="PNG", compress_level=1)
                data = base64.b64encode(buffer.getvalue()).decode()
                result = f'<p>{elapsed:.2f} seconds. Right-click the image to save it.</p><img alt="Generated image" src="data:image/png;base64,{data}">'
                self.render(prompt, seed, result)
            except (ValueError, OverflowError):
                self.render(prompt, result="<p>Enter a valid prompt and integer seed.</p>", status=400)

        def log_message(self, *args):
            pass  # Keep user prompts out of access logs.

    print(f"Local interface ready on port {port}", flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
