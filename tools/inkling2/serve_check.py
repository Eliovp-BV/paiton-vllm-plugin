"""Serve Inkling 2 with `vllm serve`, ask the reference prompts over the OpenAI API and compare with fixtures.

Usage: python3 serve_check.py <model dir> <prompts.json> <fixture dir> <out.json> [extra vllm serve args...]
Per prompt: the token ids vLLM renders (/tokenize) against the fixture's input_ids, and the greedy completion
(token ids from logprobs) against the fixture's greedy ids. Then one chat prompt at reasoning efforts none, low and
default, and that prompt and the tool prompt once more as streams. Bring-up tool: one request at a time.

For parsed replies (reasoning apart from content, tool calls as `tool_calls`) pass these extra args:
    --reasoning-parser inkling2 --enable-auto-tool-choice --tool-call-parser inkling2
Without them the tool prompt gets HTTP 400 (vLLM 0.32 refuses tools with tool_choice auto and no tool parser).
"""
import json
import subprocess
import sys
import time
import urllib.request

import numpy as np

model_dir, prompts_json, fixture_dir, out_json, *extra = sys.argv[1:]
URL = "http://127.0.0.1:18971"


def post(path, body):
    request = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=900))


def body_of(prompt):
    body = {"model": "inkling2", "messages": prompt["messages"]}
    if "tools" in prompt:
        body["tools"] = prompt["tools"]
    if "reasoning_effort" in prompt:
        body["chat_template_kwargs"] = {"reasoning_effort": prompt["reasoning_effort"]}
    return body


def stream(body):
    """One streamed chat completion, deltas joined: reasoning, content, number of tool-call deltas."""
    request = urllib.request.Request(URL + "/v1/chat/completions", json.dumps({**body, "stream": True}).encode(),
                                     {"Content-Type": "application/json"})
    message = {"reasoning": "", "content": "", "tool_call_deltas": 0}
    for line in urllib.request.urlopen(request, timeout=900):
        if line.startswith(b"data: {"):
            for choice in json.loads(line[6:])["choices"]:
                delta = choice["delta"]
                message["reasoning"] += delta.get("reasoning") or delta.get("reasoning_content") or ""
                message["content"] += delta.get("content") or ""
                message["tool_call_deltas"] += len(delta.get("tool_calls") or [])
    return message


def parsed(message):
    """What the parsers made of a reply. vLLM 0.32 names the field `reasoning`; older clients read `reasoning_content`."""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    return {"reasoning_filled": bool(reasoning), "reasoning_chars": len(reasoning),
            "markup_in_content": "<|" in (message.get("content") or ""), "tool_calls": message.get("tool_calls") or []}


def main() -> None:
    server = subprocess.Popen(["vllm", "serve", model_dir, "--served-model-name", "inkling2", "--port", "18971",
                               "--host", "127.0.0.1", "--quantization", "inkling2", *extra])
    try:
        start = time.time()
        while True:
            assert server.poll() is None, "vllm serve exited"
            assert time.time() - start < 1500, "vllm serve not ready after 25 minutes"
            try:
                urllib.request.urlopen(URL + "/health", timeout=5)
                break
            except OSError:
                time.sleep(5)
        print(f"server ready after {time.time() - start:.0f} s", flush=True)
        results = []
        prompts = json.load(open(prompts_json))
        for prompt in prompts:
            fixture = np.load(f"{fixture_dir}/{prompt['name']}.npz")
            common = body_of(prompt)
            tokens = post("/tokenize", {**common, "add_generation_prompt": True})["tokens"]
            want = fixture["input_ids"].tolist()
            t0 = time.time()
            reply = post("/v1/chat/completions", {**common, "temperature": 0, "max_tokens": len(fixture["greedy"]),
                                                  "skip_special_tokens": False, "logprobs": True,
                                                  "return_tokens_as_token_ids": True})
            choice = reply["choices"][0]
            ids = [int(entry["token"].split(":")[1]) for entry in choice["logprobs"]["content"]]
            greedy = fixture["greedy"].tolist()
            same = [a == b for a, b in zip(ids, greedy)]
            result = {"name": prompt["name"], "prompt_tokens": len(tokens), "prompt_ids_equal_fixture": tokens == want,
                      "generated": len(ids), "greedy_prefix_equal_fixture": same.index(False) if False in same else len(same),
                      "fixture_generated": len(greedy), "finish_reason": choice["finish_reason"],
                      "seconds": round(time.time() - t0, 1), **parsed(choice["message"]), "message": choice["message"]}
            if tokens != want:
                result["prompt_ids"], result["fixture_ids"] = tokens, want
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in ("prompt_ids", "fixture_ids")})[:1500], flush=True)
        # One chat prompt at three reasoning efforts: no fixture, only what comes back and how it is split.
        chat = next(p for p in prompts if "tools" not in p and "reasoning_effort" not in p)
        for effort in ("none", "low", None):
            reply = post("/v1/chat/completions", {
                "model": "inkling2", "messages": chat["messages"], "temperature": 0, "max_tokens": 128,
                **({"chat_template_kwargs": {"reasoning_effort": effort}} if effort else {})})
            choice = reply["choices"][0]
            result = {"name": f"effort_{effort or 'default'}", "prompt": chat["name"],
                      "generated": reply["usage"]["completion_tokens"], "finish_reason": choice["finish_reason"],
                      **parsed(choice["message"]), "message": choice["message"]}
            results.append(result)
            print(json.dumps(result)[:1500], flush=True)
        # Streamed: here the parser decides from the prompt whether reasoning is still open (the `inkling2` rule).
        for prompt in prompts:
            if "tools" in prompt or prompt is chat:
                message = stream({**body_of(prompt), "temperature": 0, "max_tokens": 128})
                result = {"name": f"stream_{prompt['name']}", **parsed(message),
                          "tool_call_deltas": message["tool_call_deltas"], "message": message}
                results.append(result)
                print(json.dumps(result)[:1500], flush=True)
        json.dump(results, open(out_json, "w"), indent=1)
        print("SERVE CHECK DONE")
    finally:
        server.terminate()
        server.wait(60)


if __name__ == "__main__":
    main()
