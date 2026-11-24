"""Serve Inkling 2 with `vllm serve`, ask the reference prompts over the OpenAI API and compare with fixtures.

Usage: python3 serve_check.py <model dir> <prompts.json> <fixture dir> <out.json> [extra vllm serve args...]
Per prompt: the token ids vLLM renders (/tokenize) against the fixture's input_ids, and the greedy completion
(text, token ids from logprobs) against the fixture's greedy ids. Bring-up tool: one request at a time.
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
        for prompt in json.load(open(prompts_json)):
            fixture = np.load(f"{fixture_dir}/{prompt['name']}.npz")
            common = {"model": "inkling2", "messages": prompt["messages"]}
            if "tools" in prompt:
                common["tools"] = prompt["tools"]
            if "reasoning_effort" in prompt:
                common["chat_template_kwargs"] = {"reasoning_effort": prompt["reasoning_effort"]}
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
                      "seconds": round(time.time() - t0, 1), "message": choice["message"]}
            if tokens != want:
                result["prompt_ids"], result["fixture_ids"] = tokens, want
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in ("prompt_ids", "fixture_ids")})[:1500], flush=True)
        json.dump(results, open(out_json, "w"), indent=1)
        print("SERVE CHECK DONE")
    finally:
        server.terminate()
        server.wait(60)


if __name__ == "__main__":
    main()
