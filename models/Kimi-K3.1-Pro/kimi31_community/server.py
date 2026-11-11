#!/usr/bin/env python3
"""Small OpenAI-compatible server for Kimi-K3.1-Pro on top of model.py.

One request runs at a time; others wait in line. Set PAITON_API_KEY to require "Authorization: Bearer <key>" on /v1. Endpoints: GET /health, GET /v1/models,
POST /v1/chat/completions, POST /v1/completions (both with "stream": true or false).
"""
import argparse
import hmac
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from kimi31_community.model import KimiK31

ARGS = None
MODEL = None
TOK = None
LOCK = threading.Lock()
STOP_IDS: set[int] = set()
CTRL: dict[str, int] = {}


def load(args):
    global MODEL, TOK, STOP_IDS, CTRL
    sys.path.insert(0, args.model)  # the checkpoint ships its tokenizer and chat encoding
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    n = torch.cuda.device_count()
    if n == 0:
        sys.exit("no GPU visible")
    devices = [f"cuda:{i}" for i in range(n)]
    print(f"loading {args.model} on {n} GPU(s); this reads about 1.5 TB and takes a while", flush=True)
    t0 = time.time()
    MODEL = KimiK31(args.model, devices, reserve_gb=args.reserve_gb, log=lambda s: print(s, flush=True))
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    gen = json.load(open(os.path.join(args.model, "generation_config.json")))
    eos = gen.get("eos_token_id")
    STOP_IDS = {TOK.eos_token_id, *([eos] if isinstance(eos, int) else (eos or []))}
    for name in ("<|open|>", "<|close|>", "<|sep|>"):
        CTRL[name] = TOK.convert_tokens_to_ids(name)


class Channels:
    """Splits generated tokens into reasoning and answer text using the chat format's tags."""

    def __init__(self, thinking):
        self.stack = ["think" if thinking else "response"]
        self.mode = "body"
        self.tag = []
        self.done = False  # the response channel has closed

    def feed(self, tid):
        """Returns (channel, text) for visible text, or None."""
        if tid == CTRL["<|open|>"] or tid == CTRL["<|close|>"]:
            self.mode = "open" if tid == CTRL["<|open|>"] else "close"
            self.tag = []
            return None
        if tid == CTRL["<|sep|>"]:
            name = TOK.decode(self.tag).split()[0] if self.tag else ""
            if self.mode == "open":
                self.stack.append(name)
            elif self.stack:
                if self.stack.pop() == "response":
                    self.done = True
            self.mode = "body"
            return None
        if self.mode != "body":
            self.tag.append(tid)
            return None
        top = self.stack[-1] if self.stack else ""
        ch = {"think": "reasoning", "response": "content"}.get(top)
        if ch is None:
            return None
        return ch, tid


def sample(logits, temperature, top_p):
    if temperature <= 0:
        return int(logits.argmax())
    p = torch.softmax(logits.float() / temperature, -1)
    if top_p < 1.0:
        sp, si = p.sort(descending=True)
        keep = sp.cumsum(-1) - sp < top_p
        sp = sp * keep
        return int(si[torch.multinomial(sp / sp.sum(), 1)])
    return int(torch.multinomial(p, 1))


def generate(ids, max_tokens, temperature, top_p):
    """Yields token ids. The caller holds LOCK."""
    if len(ids) + max_tokens > ARGS.max_model_len:
        raise ValueError(f"prompt ({len(ids)}) + max_tokens ({max_tokens}) is over max_model_len {ARGS.max_model_len}")
    state = None
    t = torch.tensor(ids)
    for s in range(0, len(ids), ARGS.prefill_chunk):
        logits, state = MODEL.forward(t[s:s + ARGS.prefill_chunk], state)
    for _ in range(max_tokens):
        tid = sample(logits, temperature, top_p)
        yield tid
        if tid in STOP_IDS:
            return
        logits, state = MODEL.forward(torch.tensor([tid]), state)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _err(self, code, msg):
        self._json(code, {"error": {"message": msg, "type": "invalid_request_error", "code": code}})

    def _authorized(self):
        key = os.environ.get("PAITON_API_KEY")
        if not key or not self.path.startswith("/v1/"):
            return True
        if hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {key}"):
            return True
        self._err(401, "missing or wrong API key")
        return False

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        if not self._authorized():
            return
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": ARGS.served_model_name, "object": "model",
                                                               "owned_by": "kimi31-community",
                                                               "max_model_len": ARGS.max_model_len}]})
        self._err(404, "not found")

    def do_POST(self):
        if not self._authorized():
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError:
            return self._err(400, "body is not valid JSON")
        chat = self.path == "/v1/chat/completions"
        if not chat and self.path != "/v1/completions":
            return self._err(404, "not found")
        try:
            if chat:
                kw = dict(body.get("chat_template_kwargs") or {})
                thinking = bool(kw.pop("thinking", kw.pop("enable_thinking", False)))
                ids = TOK.apply_chat_template(body["messages"], tokenize=True, add_generation_prompt=True,
                                              thinking=thinking, **kw)
                if isinstance(ids, dict):
                    ids = ids["input_ids"]
            else:
                thinking = False
                p = body["prompt"]
                ids = p if isinstance(p, list) else TOK.encode(p)
        except Exception as e:  # bad messages or prompt
            return self._err(400, f"could not build the prompt: {e}")
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024)
        temperature = float(body.get("temperature", 1.0))
        top_p = float(body.get("top_p", 1.0))
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        created = int(time.time())
        stream = bool(body.get("stream"))

        def chunk(delta_or_text, finish=None):
            if chat:
                return {"id": rid, "object": "chat.completion.chunk", "created": created,
                        "model": ARGS.served_model_name,
                        "choices": [{"index": 0, "delta": delta_or_text, "finish_reason": finish}]}
            return {"id": rid, "object": "text_completion", "created": created, "model": ARGS.served_model_name,
                    "choices": [{"index": 0, "text": delta_or_text, "finish_reason": finish}]}

        with LOCK:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
            ch = Channels(thinking)
            out = {"reasoning": [], "content": []}
            sent = {"reasoning": 0, "content": 0}
            n_out, finish = 0, "length"
            try:
                for tid in generate(ids, max_tokens, temperature, top_p):
                    n_out += 1
                    if tid in STOP_IDS:
                        finish = "stop"
                        break
                    r = ch.feed(tid) if chat else ("content", tid)
                    if chat and ch.done:
                        finish = "stop"  # the answer closed; what follows is the chat format's closing tags
                        break
                    if r is None:
                        continue
                    name, t = r
                    out[name].append(t)
                    if stream:
                        text = TOK.decode(out[name])
                        if text.endswith("�"):
                            continue  # wait for the rest of a multi-byte character
                        new, sent[name] = text[sent[name]:], len(text)
                        if new:
                            key = "reasoning_content" if name == "reasoning" else "content"
                            payload = chunk({key: new} if chat else new)
                            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                            self.wfile.flush()
            except ValueError as e:
                if stream:
                    self.wfile.write(f"data: {json.dumps({'error': {'message': str(e)}})}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                return self._err(400, str(e))
            except (BrokenPipeError, ConnectionResetError):
                return
            usage = {"prompt_tokens": len(ids), "completion_tokens": n_out, "total_tokens": len(ids) + n_out}
            if stream:
                last = chunk({} if chat else "", finish)
                last["usage"] = usage
                self.wfile.write(f"data: {json.dumps(last)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
        content = TOK.decode(out["content"])
        if chat:
            msg = {"role": "assistant", "content": content}
            if out["reasoning"]:
                msg["reasoning_content"] = TOK.decode(out["reasoning"])
            resp = {"id": rid, "object": "chat.completion", "created": created, "model": ARGS.served_model_name,
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}
        else:
            resp = {"id": rid, "object": "text_completion", "created": created, "model": ARGS.served_model_name,
                    "choices": [{"index": 0, "text": content, "finish_reason": finish}], "usage": usage}
        self._json(200, resp)


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("MODEL_DIR", "/model"), help="checkpoint directory")
    ap.add_argument("--served-model-name", default="kimi-k3.1-pro")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=32768, help="prompt + output tokens per request")
    ap.add_argument("--prefill-chunk", type=int, default=512, help="prompt tokens per forward pass")
    ap.add_argument("--reserve-gb", type=float, default=40.0, help="GPU memory per GPU kept free for activations")
    ARGS = ap.parse_args()
    load(ARGS)
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"serving {ARGS.served_model_name} on http://{ARGS.host}:{ARGS.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
