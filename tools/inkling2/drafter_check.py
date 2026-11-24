"""Greedy replies of Inkling 2 with or without its drafter, and how many tokens a verify step yields.

    python3 drafter_check.py run <model dir> <out.json> <prompts.json>[,<prompts.json>...] [extra vllm serve args...]
    python3 drafter_check.py compare <off.json> <on.json>

`run` starts `vllm serve`, asks every text prompt once (temperature 0, the prompt's `max_new_tokens` or 64) and
stores the token ids. With a drafter it also stores vLLM's spec-decode counters per prompt: tokens per verify step
= 1 + accepted drafts / draft steps (the first token of a reply comes from the prefill, as in the reference
script `quality/dspark_ref.py`, whose figure this is compared with). `compare` prints, per prompt, where the two
replies first differ; exit code 1 when any differs. Bring-up tool: one request at a time.
"""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:18971"


def post(path, body):
    request = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=900))


def counters():
    """vLLM's spec-decode counters, summed over labels: drafts, draft_tokens, accepted_tokens."""
    out = {}
    for line in urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode().splitlines():
        if line.startswith("vllm:spec_decode_num_") and "_created" not in line:
            name = line.split("{")[0].split(" ")[0].removeprefix("vllm:spec_decode_num_").removesuffix("_total")
            if name in ("drafts", "draft_tokens", "accepted_tokens"):
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def run(model_dir, out_json, prompt_files, extra):
    server = subprocess.Popen(["vllm", "serve", model_dir, "--served-model-name", "inkling2", "--port", "18971",
                               "--host", "127.0.0.1", "--quantization", "inkling2", *extra])
    try:
        start = time.time()
        while True:
            assert server.poll() is None, "vllm serve exited"
            assert time.time() - start < 900, "vllm serve not ready after 15 minutes"
            try:
                urllib.request.urlopen(URL + "/health", timeout=5)
                break
            except OSError:
                time.sleep(3)
        print(f"server ready after {time.time() - start:.0f} s", flush=True)
        results = {}
        for prompt in (p for f in prompt_files.split(",") for p in json.load(open(f))):
            if prompt["name"].startswith("img_"):
                continue
            body = {"model": "inkling2", "messages": prompt["messages"], "temperature": 0,
                    "max_tokens": prompt.get("max_new_tokens", 64), "skip_special_tokens": False, "logprobs": True,
                    "return_tokens_as_token_ids": True}
            if "tools" in prompt:
                body["tools"] = prompt["tools"]
            if "reasoning_effort" in prompt:
                body["chat_template_kwargs"] = {"reasoning_effort": prompt["reasoning_effort"]}
            before, t0 = counters(), time.time()
            try:
                reply = post("/v1/chat/completions", body)
            except urllib.error.HTTPError as error:  # a prompt the API refuses (tool arguments as an object)
                print(prompt["name"], "skipped: HTTP", error.code, flush=True)
                continue
            seconds, after = time.time() - t0, counters()
            choice = reply["choices"][0]
            ids = [int(entry["token"].split(":")[1]) for entry in choice["logprobs"]["content"]]
            row = {"prompt_tokens": reply["usage"]["prompt_tokens"], "tokens": ids,
                   "finish_reason": choice["finish_reason"], "seconds": round(seconds, 2)}
            if after:
                delta = {k: after[k] - before.get(k, 0.0) for k in after}
                row.update(delta, tokens_per_step=round(1 + delta["accepted_tokens"] / max(delta["drafts"], 1), 3))
            results[prompt["name"]] = row
            print(prompt["name"], json.dumps({k: v for k, v in row.items() if k != "tokens"}), flush=True)
            json.dump(results, open(out_json, "w"), indent=1)
        drafts = sum(r.get("drafts", 0) for r in results.values())
        if drafts:
            accepted = sum(r["accepted_tokens"] for r in results.values())
            print(f"all: {1 + accepted / drafts:.3f} tokens per verify step ({accepted:.0f} accepted in {drafts:.0f} steps)")
        print("DRAFTER CHECK DONE", flush=True)
    finally:
        server.terminate()
        server.wait(60)


def compare(off_json, on_json):
    off, on = json.load(open(off_json)), json.load(open(on_json))
    differ = 0
    for name in off:
        a, b = off[name]["tokens"], on[name]["tokens"]
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        same = first is None and len(a) == len(b)
        differ += not same
        print(f"{name:14s} off {len(a):4d} on {len(b):4d} tokens  "
              + ("equal" if same else f"first difference at token {first if first is not None else min(len(a), len(b))}")
              + (f"  tokens/step {on[name]['tokens_per_step']}" if "tokens_per_step" in on[name] else ""))
    print("GREEDY EQUAL" if not differ else f"GREEDY DIFFERS on {differ} of {len(off)} prompts")
    return 1 if differ else 0


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        sys.exit(compare(*sys.argv[2:4]))
    run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
