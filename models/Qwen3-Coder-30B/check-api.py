"""Check local coding API connectivity, arithmetic and streamed tool-call parsing."""

import argparse
import json
from pathlib import Path
import urllib.request


def check(url, output):
    records = []
    def request(body):
        req = urllib.request.Request(url + "/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as response:
            if body.get("stream"):
                result = [json.loads(line[6:]) for line in response
                          if line.startswith(b"data: ") and line.strip() != b"data: [DONE]"]
            else:
                result = json.load(response)
        records.append({"request": body, "response": result})
        output.write_text(json.dumps(records, indent=2) + "\n")
        return result

    answer = request({"model": "qwen3-coder", "messages": [{"role": "user",
        "content": "Compute 17 * 23 + 9. Reply with only the decimal integer."}],
        "max_tokens": 32, "temperature": 0, "seed": 1201})
    assert answer["choices"][0]["message"]["content"].strip() == "400"
    tool = {"type": "function", "function": {"name": "read_file",
        "description": "Read a source file before reviewing its code.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"], "additionalProperties": False}}}
    for stream in [False, True]:
        response = request({"model": "qwen3-coder", "messages": [{"role": "user",
            "content": "Use the read_file tool to read src/main.py. Do not guess the contents."}],
            "tools": [tool], "tool_choice": "auto", "temperature": 0, "seed": 1201,
            "max_tokens": 256, "stream": stream})
        if stream:
            calls = {}
            for event in response:
                for choice in event.get("choices", []):
                    for part in choice.get("delta", {}).get("tool_calls", []):
                        call = calls.setdefault(part["index"], {"name": "", "arguments": ""})
                        function = part.get("function") or {}
                        for key in ["name", "arguments"]:
                            call[key] += function.get(key) or ""
            functions = list(calls.values())
        else:
            functions = [call["function"] for call in response["choices"][0]["message"].get("tool_calls", [])]
        assert len(functions) == 1, functions
        assert functions[0]["name"] == "read_file", functions
        assert json.loads(functions[0]["arguments"]) == {"path": "src/main.py"}, functions
    print("API, arithmetic, non-streaming and streaming tool-call checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--output", type=Path, default=Path("qwen3-coder-api-check.json"))
    args = parser.parse_args()
    check(args.url.rstrip("/"), args.output)
