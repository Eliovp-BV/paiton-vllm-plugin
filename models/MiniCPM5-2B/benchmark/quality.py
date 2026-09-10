"""Retained deterministic chat checks; generated code runs in a bounded container."""

import argparse, json, pathlib, subprocess, time, urllib.request, uuid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8020")
    p.add_argument("--model", default="gpt-oss-20b")
    p.add_argument("--cases", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--code-image", required=True)
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--disable-thinking", action="store_true")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top-p", type=float, default=1)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--repetition-penalty", type=float, default=1)
    a = p.parse_args()
    results = []
    for case in json.loads(a.cases.read_text()):
        body = {
            "model": a.model,
            "messages": [{"role": "user", "content": case["prompt"]}],
            "temperature": a.temperature,
            "top_p": a.top_p, "top_k": a.top_k, "repetition_penalty": a.repetition_penalty,
            "seed": 1201,
            "max_tokens": 4096 if a.thinking else 1024,
            "return_token_ids": True,
        }
        if a.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": True}
        elif a.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if a.reasoning_effort:
            body["reasoning_effort"] = a.reasoning_effort
        if case.get("tools"):
            body.update(tools=case["tools"], tool_choice="auto")
        if case.get("structured"):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name", "age"],
                        "additionalProperties": False,
                    },
                },
            }
        rec = {"id": case["id"], "request": body, "passed": False}
        start = time.monotonic()
        try:
            req = urllib.request.Request(
                a.url + "/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=240) as r:
                response = json.load(r)
            rec.update(response=response, elapsed_s=time.monotonic() - start)
            msg = response["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            check = case["check"]
            if check == "exact":
                rec["passed"] = content == case["expected"]
            elif check == "json":
                rec["passed"] = json.loads(content) == case["expected"]
            elif check == "tool":
                calls = msg.get("tool_calls", [])
                rec["passed"] = (
                    len(calls) == 1
                    and calls[0]["function"]["name"] == "weather"
                    and json.loads(calls[0]["function"]["arguments"])
                    == case["expected"]
                )
            elif check == "code":
                code = content
                if code.startswith("```"):
                    code = "\n".join(code.splitlines()[1:-1])
                container_name = "paiton-code-eval-" + uuid.uuid4().hex[:12]
                command = [
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    container_name,
                    "--ulimit",
                    "cpu=5:5",
                    "-i",
                    "--network",
                    "none",
                    "--read-only",
                    "--memory",
                    "128m",
                    "--memory-swap",
                    "128m",
                    "--cpus",
                    "0.5",
                    "--pids-limit",
                    "32",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--user",
                    "65534:65534",
                    "--entrypoint",
                    "python3",
                    a.code_image,
                    "-I",
                    "-",
                ]
                try:
                    r = subprocess.run(
                        command,
                        input=code + "\n" + case["tests"] + "\n",
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                finally:
                    subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                    )
                rec.update(
                    code_exit=r.returncode,
                    code_stdout=r.stdout,
                    code_stderr=r.stderr,
                    code_command=command,
                )
                rec["passed"] = r.returncode == 0
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        results.append(rec)
        a.out.write_text(
            json.dumps(
                {
                    "cases": results,
                    "passed": sum(x["passed"] for x in results),
                    "total": len(results),
                },
                indent=2,
            )
        )
        print(case["id"], rec["passed"], flush=True)


if __name__ == "__main__":
    main()
