"""Small terminal chat client for the local OpenAI-compatible Paiton server."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with a local Paiton server")
    parser.add_argument(
        "--url",
        default=os.environ.get(
            "PAITON_CHAT_URL", "http://127.0.0.1:8000/v1/chat/completions"
        ),
        help="OpenAI-compatible chat-completions URL",
    )
    parser.add_argument("--model", default=os.environ.get("PAITON_CHAT_MODEL", "qwen38"))
    parser.add_argument("--system", default=os.environ.get("PAITON_CHAT_SYSTEM", ""))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable the model's thinking mode (disabled by default)",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser


def _request_payload(
    messages: list[dict[str, str]], args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": args.thinking},
    }


def _stream_text(lines: Iterable[bytes]) -> Iterable[str]:
    for raw_line in lines:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        event = json.loads(data)
        choices = event.get("choices") or []
        if not choices:
            continue
        text = choices[0].get("delta", {}).get("content")
        if text:
            yield text


def _complete(
    messages: list[dict[str, str]], args: argparse.Namespace
) -> str:
    body = json.dumps(_request_payload(messages, args)).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    pieces: list[str] = []
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for text in _stream_text(response):
            pieces.append(text)
            print(text, end="", flush=True)
    return "".join(pieces)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be positive")
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print("Paiton Qwen3.8 chat: /help for commands, /quit to exit.")
    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        command = prompt.lower()
        if command in {"/quit", "/exit"}:
            return 0
        if command == "/help":
            print("/reset clears the conversation; /quit exits.")
            continue
        if command == "/reset":
            messages = (
                [{"role": "system", "content": args.system}] if args.system else []
            )
            print("Conversation cleared.")
            continue

        messages.append({"role": "user", "content": prompt})
        print("Paiton> ", end="", flush=True)
        try:
            answer = _complete(messages, args)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            print(f"\nServer returned HTTP {error.code}: {detail}", file=sys.stderr)
            messages.pop()
            continue
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"\nCould not reach the Paiton server: {error}", file=sys.stderr)
            messages.pop()
            continue
        print()
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    raise SystemExit(main())
