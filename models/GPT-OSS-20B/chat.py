#!/usr/bin/env python3
"""Terminal chat with a locally running GPT-OSS OpenAI-compatible API."""
import argparse
import json
import sys
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8020")
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"], default="low"
    )
    parser.add_argument("--show-reasoning", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("prompt", nargs="?")
    args = parser.parse_args()
    messages = []
    while True:
        try:
            prompt = args.prompt if args.prompt is not None else input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.strip().lower() in ("/quit", "/exit"):
            return
        if not prompt.strip():
            continue
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": "gpt-oss-20b",
            "messages": messages,
            "reasoning_effort": args.reasoning_effort,
            "max_tokens": args.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        req = urllib.request.Request(
            args.url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        answer = ""
        thinking = False
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                for line in response:
                    if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                        continue
                    event = json.loads(line[6:])
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        if args.show_reasoning and delta.get("reasoning"):
                            if not thinking:
                                print("[Reasoning] ", end="", flush=True)
                                thinking = True
                            print(delta["reasoning"], end="", flush=True)
                        if delta.get("content"):
                            if thinking:
                                print("\n[Answer] ", end="", flush=True)
                                thinking = False
                            answer += delta["content"]
                            print(delta["content"], end="", flush=True)
            print()
            if not answer:
                print("[No final answer was returned.]", file=sys.stderr)
            messages.append({"role": "assistant", "content": answer})
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"API request failed: {error}", file=sys.stderr)
            messages.pop()
            if args.prompt is not None:
                raise SystemExit(1)
        if args.prompt is not None:
            return


if __name__ == "__main__":
    main()
