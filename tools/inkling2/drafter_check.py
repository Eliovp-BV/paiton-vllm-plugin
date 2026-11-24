"""Greedy replies of Inkling 2 with or without its drafter, and how many tokens a verify step yields.

    python3 drafter_check.py run <model dir> <out.json> <prompts.json>[,<prompts.json>...] [extra vllm serve args...]
    python3 drafter_check.py compare <off.json> <on.json>

`run` starts `vllm serve`, asks every text prompt once (temperature 0, the prompt's `max_new_tokens` or 64), then
the reference script's long prompt (the ids of `code_low` 50 times), and stores the token ids. INK2_USERS=4 sends
four prompts at once (then the counters are per run, not per prompt). With a drafter it also stores vLLM's spec-decode counters per prompt: tokens per verify step
= 1 + accepted drafts / draft steps (the first token of a reply comes from the prefill, as in the reference
script `quality/dspark_ref.py`, whose figure this is compared with). `compare` prints, per prompt, where the two
replies first differ; exit code 1 when any differs. Bring-up tool: one request at a time.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:18971"
USERS = int(os.environ.get("INK2_USERS", "1"))  # requests sent at once


def post(path, body):
    request = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=900))


def counters():
    """vLLM's spec-decode counters, summed over labels: drafts, draft_tokens, accepted_tokens."""
    out = {}
    for line in urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode().splitlines():
        if line.startswith("vllm:spec_decode_num_") and "_created" not in line:
            name = line.split("{")[0].split(" ")[0].removeprefix("vllm:spec_decode_num_").removesuffix("_total")
            if name == "accepted_tokens_per_pos":  # accepted drafts at draft position 0, 1, ...
                name = "pos" + line.split('position="')[1].split('"')[0]
            if name in ("drafts", "draft_tokens", "accepted_tokens") or name.startswith("pos"):
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def ask(name, path, body, rows, i):
    """One greedy request; its token ids into rows[i]."""
    t0 = time.time()
    try:
        reply = post(path, body)
    except urllib.error.HTTPError as error:  # a prompt the API refuses (tool arguments as an object)
        print(name, "skipped: HTTP", error.code, flush=True)
        return
    choice = reply["choices"][0]
    tokens = [e["token"] for e in choice["logprobs"]["content"]] if "message" in choice else choice["logprobs"]["tokens"]
    rows[i] = {"prompt_tokens": reply["usage"]["prompt_tokens"], "tokens": [int(t.split(":")[1]) for t in tokens],
               "finish_reason": choice["finish_reason"], "seconds": round(time.time() - t0, 2)}


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
        results, requests, long_ids = {}, [], None
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
            requests.append((prompt["name"], "/v1/chat/completions", body))
            if prompt["name"] == "code_low":  # the reference script's long prompt: these ids 50 times (2,400 tokens)
                long_ids = post("/tokenize", {"model": "inkling2", "messages": prompt["messages"],
                                              "add_generation_prompt": True,
                                              "chat_template_kwargs": body["chat_template_kwargs"]})["tokens"] * 50
        if long_ids:
            requests.append(("long", "/v1/completions", {
                "model": "inkling2", "prompt": long_ids, "temperature": 0, "max_tokens": 64,
                "skip_special_tokens": False, "logprobs": 1, "return_tokens_as_token_ids": True}))
        drafts = accepted = wall = wall_tokens = 0.0
        first = counters()
        for start in range(0, len(requests), USERS):  # USERS requests at once
            group = requests[start : start + USERS]
            rows, before, t0 = [None] * len(group), counters(), time.time()
            threads = [threading.Thread(target=ask, args=(*request, rows, i)) for i, request in enumerate(group)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if start:  # the first group warms the server up
                wall, wall_tokens = wall + time.time() - t0, wall_tokens + sum(len(r["tokens"]) for r in rows if r)
            after = counters()
            delta = {k: after[k] - before.get(k, 0.0) for k in after}
            drafts, accepted = drafts + delta.get("drafts", 0.0), accepted + delta.get("accepted_tokens", 0.0)
            for (name, _, _), row in zip(group, rows):
                if row is None:
                    continue
                if delta and USERS == 1:
                    row.update(delta, tokens_per_step=round(1 + delta["accepted_tokens"] / max(delta["drafts"], 1), 3))
                results[name] = row
                print(name, json.dumps({k: v for k, v in row.items() if k != "tokens"}), flush=True)
            json.dump(results, open(out_json, "w"), indent=1)
        if drafts:
            print(f"all: {1 + accepted / drafts:.3f} tokens per verify step ({accepted:.0f} accepted in {drafts:.0f} steps)")
            last = counters()
            by_pos = [last[k] - first.get(k, 0.0) for k in sorted((k for k in last if k.startswith("pos")),
                                                                  key=lambda k: int(k[3:]))]
            # what a draft cut after k tokens would have yielded, from this run's accepted counts per position
            print("tokens per step if only the first k drafts were verified:",
                  ", ".join(f"k={k + 1}: {1 + sum(by_pos[: k + 1]) / drafts:.3f}" for k in range(len(by_pos))))
        seconds = sum(r["seconds"] for n, r in results.items() if n != next(iter(results)))
        tokens = sum(len(r["tokens"]) for n, r in results.items() if n != next(iter(results)))
        print(f"all but the first request: {tokens} tokens in {seconds:.2f} s of request time = {tokens / seconds:.1f} tokens/s")
        print(f"{USERS} at once, all but the first group: {wall_tokens:.0f} tokens in {wall:.2f} s = {wall_tokens / max(wall, 1e-9):.1f} tokens/s")
        print("DRAFTER CHECK DONE", flush=True)
    finally:
        server.terminate()
        server.wait(60)


def compare(off_json, on_json):
    off, on = json.load(open(off_json)), json.load(open(on_json))
    differ = 0
    for name in off:
        if name not in on:
            differ += 1
            print(f"{name:14s} missing in the second run")
            continue
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
