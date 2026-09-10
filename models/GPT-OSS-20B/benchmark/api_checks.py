"""Stream, effort, tool continuation and long-context API qualification."""

import argparse
import json
from pathlib import Path
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from harmony import count_channels


def call(url, body):
    start = time.perf_counter()
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    events = []
    saw_done = False
    with urllib.request.urlopen(request, timeout=240) as response:
        for line in response:
            if line.startswith(b"data: ") and line[6:].strip() == b"[DONE]":
                saw_done = True
            if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                events.append(
                    {
                        "elapsed_s": time.perf_counter() - start,
                        "data": json.loads(line[6:]),
                    }
                )
    content = ""
    reasoning = ""
    calls = {}
    usage = None
    first_answer = None
    first_reasoning = None
    for event in events:
        data = event["data"]
        if data.get("usage"):
            usage = data["usage"]
        for choice in data.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content"):
                content += delta["content"]
                if first_answer is None and delta["content"].strip():
                    first_answer = event["elapsed_s"]
            if delta.get("reasoning"):
                reasoning += delta["reasoning"]
                if first_reasoning is None:
                    first_reasoning = event["elapsed_s"]
            for part in delta.get("tool_calls", []):
                dst = calls.setdefault(
                    part["index"],
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if part.get("id"):
                    dst["id"] = part["id"]
                for key in ("name", "arguments"):
                    dst["function"][key] += part.get("function", {}).get(key) or ""
    id_events = [
        (e["elapsed_s"], c["token_ids"])
        for e in events
        for c in e["data"].get("choices", [])
        if c.get("token_ids")
    ]
    channels = count_channels(id_events) if id_events else {}
    return dict(
        channels,
        **{
            "stream_complete": saw_done,
            "request": body,
            "events": events,
            "elapsed_s": time.perf_counter() - start,
            "content": content,
            "reasoning": reasoning,
            "tool_calls": list(calls.values()),
            "usage": usage,
            "first_answer_s": first_answer,
            "first_reasoning_s": first_reasoning,
        }
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8020")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--long-cases", type=Path)
    a = p.parse_args()
    base = {
        "model": "gpt-oss-20b",
        "temperature": 0,
        "seed": 1201,
        "max_tokens": 512,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
        "reasoning_effort": "low",
    }
    results = []

    def save(record):
        record["passed"] = record["passed"] and record["stream_complete"]
        results.append(record)
        a.out.write_text(json.dumps(results, indent=2))
        print(record["id"], record["passed"], flush=True)

    for effort in ["low", "medium", "high"]:
        r = call(
            a.url,
            dict(
                base,
                reasoning_effort=effort,
                messages=[
                    {
                        "role": "user",
                        "content": "What is 17 plus 25? Reply with only the integer.",
                    }
                ],
            ),
        )
        r.update(
            id="stream-effort-" + effort,
            passed=r["content"].strip() == "42" and r["first_answer_s"] is not None,
        )
        save(r)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get the temperature for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city", "unit"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": "Use the weather tool for Paris in Celsius, then tell me the temperature in one sentence.",
        }
    ]
    r = call(a.url, dict(base, messages=messages, tools=tools, tool_choice="auto"))
    try:
        r["passed"] = (
            len(r["tool_calls"]) == 1
            and r["tool_calls"][0]["function"]["name"] == "weather"
            and json.loads(r["tool_calls"][0]["function"]["arguments"])
            == {"city": "Paris", "unit": "celsius"}
        )
    except Exception:
        r["passed"] = False
    r["id"] = "stream-tool"
    save(r)
    if r["passed"]:
        messages += [
            {"role": "assistant", "content": None, "tool_calls": r["tool_calls"]},
            {
                "role": "tool",
                "tool_call_id": r["tool_calls"][0]["id"],
                "content": '{"city":"Paris","temperature":18,"unit":"celsius"}',
            },
        ]
        r = call(a.url, dict(base, messages=messages, tools=tools, tool_choice="auto"))
        r.update(
            id="tool-continuation", passed="18" in r["content"] and not r["tool_calls"]
        )
        save(r)
    if a.long_cases:
        cases = json.loads(a.long_cases.read_text())

        def one(case):
            r = call(a.url, dict(base, messages=case["messages"]))
            r.update(
                id=case["id"],
                passed=r["content"].strip() == case["expected"],
                expected=case["expected"],
            )
            return r

        for case in cases:
            save(one(case))
        with ThreadPoolExecutor(max_workers=2) as pool:
            for r in pool.map(one, cases[:2]):
                r["id"] += "-concurrent"
                save(r)


if __name__ == "__main__":
    main()
