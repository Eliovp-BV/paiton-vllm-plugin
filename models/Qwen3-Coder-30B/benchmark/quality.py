"""Small retained quality suite; generated Python runs in a bounded container."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request


CASES = [
    {"id": "merge", "kind": "python", "prompt": "Return only one Python code block implementing merge_intervals(intervals). Merge overlapping closed integer intervals. Return sorted tuples. Do not mutate input. No imports except standard library.",
     "tests": "x=[(5,7),(1,3),(2,4),(9,9)]; assert merge_intervals(x)==[(1,4),(5,7),(9,9)]; assert x==[(5,7),(1,3),(2,4),(9,9)]; assert merge_intervals([])==[]; assert merge_intervals([(1,2),(2,3)])==[(1,3)]"},
    {"id": "lower_bound", "kind": "python", "prompt": "Return only one Python code block implementing lower_bound(values, target) in O(log n) time: the first index with value >= target, or len(values). The values are sorted. Do not mutate inputs.",
     "tests": "import bisect, random\nr=random.Random(1201)\nfor _ in range(100):\n a=sorted(r.randrange(-20,21) for _ in range(r.randrange(40)))\n for t in range(-22,23): assert lower_bound(a,t)==bisect.bisect_left(a,t)"},
    {"id": "brackets", "kind": "python", "prompt": "Return only one Python code block implementing balanced_brackets(text). Check proper nesting of (), [] and {}. Ignore all other characters. Return a bool.",
     "tests": "assert balanced_brackets('a{b[c(d)e]f}') is True; assert balanced_brackets('') is True; assert balanced_brackets('([)]') is False; assert balanced_brackets(']') is False; assert balanced_brackets('((') is False"},
    {"id": "dedup", "kind": "python", "prompt": "Return only one Python code block implementing stable_unique(items). Return a list retaining only the first occurrence of each item, preserving order. Support unhashable items such as lists and dictionaries. Do not mutate input.",
     "tests": "x=[[1],{'a':2},[1],{'a':2},[3]]; assert stable_unique(x)==[[1],{'a':2},[3]]; assert len(x)==5; assert stable_unique([])==[]; assert stable_unique([3,1,3,2,1])==[3,1,2]"},
    {"id": "json", "kind": "json", "prompt": "Return only a JSON object with exactly these keys: name is the string Paiton, gpu_count is the integer 1, and architectures is an array containing only gfx1201. No markdown or other text.",
     "expected": {"name": "Paiton", "gpu_count": 1, "architectures": ["gfx1201"]}},
    {"id": "instruction", "kind": "exact", "prompt": "Reply with exactly the following three lines, without quotes, markdown, or extra words:\nalpha\nbeta\ngamma", "expected": "alpha\nbeta\ngamma"},
    {"id": "arithmetic", "kind": "exact", "prompt": "Compute 17 * 23 + 9. Reply with only the decimal integer.", "expected": "400"},
    {"id": "sorting", "kind": "json", "prompt": "Sort [8, -2, 3, 8, 0, -2] in ascending order, retaining duplicates. Reply only with a JSON array, no explanation or markdown.", "expected": [-2, -2, 0, 3, 8, 8]},
]


def check_python(text, tests, image):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    code = blocks[0] if blocks else text
    harness = "import resource\nresource.setrlimit(resource.RLIMIT_CPU,(2,2))\nresource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))\n"
    harness += f"exec(compile({code!r},'<model>','exec'))\nexec(compile({tests!r},'<checks>','exec'))\nprint('CHECKS_PASSED')\n"
    command = ["docker", "run", "--rm", "-i", "--network", "none", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--user", "65534:65534", "--pids-limit", "32", "--memory", "256m",
               "--cpus", "1", "--entrypoint", "python3", image, "-I", "-"]
    start = time.time()
    try:
        result = subprocess.run(command, input=harness, text=True, capture_output=True, timeout=15)
        return {"passed": result.returncode == 0 and result.stdout.strip().endswith("CHECKS_PASSED"),
                "returncode": result.returncode, "stdout": result.stdout[-4096:], "stderr": result.stderr[-4096:],
                "code": code, "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "sandbox_command": command, "sandbox_seconds": time.time() - start}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "sandbox timeout", "code": code}


def run(args):
    records = []
    for case in CASES:
        body = {"model": "qwen3-coder", "messages": [{"role": "user", "content": case["prompt"]}],
                "max_tokens": 768, "temperature": 0, "top_p": 1, "top_k": -1,
                "seed": 1201, "repetition_penalty": 1, "stream": False}
        request = urllib.request.Request(args.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        start = time.time()
        with urllib.request.urlopen(request, timeout=900) as response:
            data = json.load(response)
        text = data["choices"][0]["message"]["content"]
        if case["kind"] == "python":
            check = check_python(text, case["tests"], args.image)
        elif case["kind"] == "json":
            try:
                check = {"passed": json.loads(text) == case["expected"]}
            except ValueError as error:
                check = {"passed": False, "error": str(error)}
        else:
            check = {"passed": text.strip() == case["expected"]}
        record = {"case": case, "request": body, "response": data, "elapsed_s": time.time() - start,
                  "check": check, "truncated": data["choices"][0]["finish_reason"] == "length"}
        records.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(case["id"], check["passed"], data["usage"], "truncated", record["truncated"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="c56baf54aca1")
    run(parser.parse_args())
