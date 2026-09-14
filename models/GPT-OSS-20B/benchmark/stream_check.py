"""Measure first reasoning and first visible answer events, preserving raw SSE."""

import argparse, json, pathlib, time, urllib.request

p = argparse.ArgumentParser()
p.add_argument("--out", type=pathlib.Path, required=True)
p.add_argument("--url", default="http://127.0.0.1:8020")
a = p.parse_args()
body = {
    "model": "gpt-oss-20b",
    "messages": [
        {
            "role": "user",
            "content": "Explain why a binary search needs a sorted array in two sentences.",
        }
    ],
    "temperature": 0,
    "max_tokens": 512,
    "reasoning_effort": "low",
    "stream": True,
    "return_token_ids": True,
    "stream_options": {"include_usage": True},
}
r = {"request": body, "events": [], "reasoning": "", "answer": ""}
start = time.perf_counter()
try:
    req = urllib.request.Request(
        a.url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            text = line[6:].strip()
            if text == b"[DONE]":
                r["done"] = True
                continue
            e = json.loads(text)
            t = time.perf_counter() - start
            r["events"].append({"elapsed_s": t, "data": e})
            if e.get("usage"):
                r["usage"] = e["usage"]
            for c in e.get("choices", []):
                d = c.get("delta", {})
                reason = d.get("reasoning") or d.get("reasoning_content")
                answer = d.get("content")
                if reason:
                    r.setdefault("first_reasoning_s", t)
                    r.setdefault("ttft_s", t)
                    r["reasoning"] += reason
                if answer:
                    r.setdefault("first_visible_answer_s", t)
                    r.setdefault("ttft_s", t)
                    r["answer"] += answer
                if c.get("finish_reason"):
                    r["finish_reason"] = c["finish_reason"]
except Exception as e:
    r["error"] = f"{type(e).__name__}: {e}"
r["e2e_s"] = time.perf_counter() - start
a.out.write_text(json.dumps(r, indent=2))
print({k: v for k, v in r.items() if k not in ["events", "request"]})
