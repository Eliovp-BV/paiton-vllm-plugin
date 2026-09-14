"""Fixed, inspectable Qwen3-Coder workload. No model-generated input data."""

CODING_TASKS = [
    ("intervals", "Implement merge_intervals(intervals): merge overlapping closed integer intervals, return sorted tuples, and do not mutate the input."),
    ("toposort", "Implement topological_sort(graph) for a dict of node to successor list. Include successor-only nodes, return lexicographically smallest valid order, and raise ValueError for cycles."),
    ("lru", "Implement an LRUCache class with get(key) returning -1 for a miss and put(key,value). Handle updates, capacity zero and eviction in constant expected time."),
    ("csv", "Implement parse_csv_line(text) without using csv. Support quoted fields, escaped double quotes and commas inside quotes. Reject unterminated quoted fields."),
    ("retry", "Implement retry(fn, attempts, retryable) that returns fn(), retries only exceptions in retryable, preserves the final exception and rejects attempts below one."),
    ("window", "Implement sliding_window_max(values, k) in O(n) time using a deque. Reject invalid k and describe the deque invariant."),
    ("jsonpaths", "Implement flatten_json(value) returning a dict of dotted paths to scalar values, with array indices in brackets. Preserve empty dicts and lists."),
    ("binarysearch", "Implement lower_bound(values, target) returning the first index whose value is >= target. Include duplicates, empty inputs and targets beyond both ends."),
]

CONTEXT = """
You are reviewing a small Python library used by a code indexing service.
The library processes user supplied data and runs on Python 3.12. Prefer
standard library dependencies and explicit exceptions to assertions for
invalid public inputs. Do not perform file, network or shell operations.
The caller may reuse its input objects after the function returns, so avoid
mutating arguments. Results should be deterministic. Document the behavior
at boundary conditions and include small executable examples. Use readable
names, add type hints where useful, and explain the time and space complexity.
During review, pay special attention to empty inputs, repeated values, stable
ordering, and off-by-one errors. Separate validation from the core algorithm
when this improves readability. A short correct implementation is preferred
to an abstraction that hides the algorithm. Include a brief explanation of
why the algorithm works and mention any assumptions about input types.
""".strip()


def make_requests(tokenizer):
    requests = []
    for variant in range(2):
        for name, task in CODING_TASKS:
            messages = [
                {"role": "system", "content": "You are a precise Python coding assistant."},
                {"role": "user", "content": CONTEXT + "\n\n" + task +
                 ("\nInclude a focused test plan." if variant else "\nInclude runnable unit tests.")},
            ]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
            requests.append({
                "id": f"{name}-{variant}", "messages": messages,
                "body": {"model": "qwen3-coder", "prompt": ids, "max_tokens": 256,
                         "temperature": 0, "top_p": 1, "top_k": -1, "seed": 1201,
                         "repetition_penalty": 1, "ignore_eos": True,
                         "stream": True, "stream_options": {"include_usage": True},
                         "logprobs": 1},
            })
    return requests


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    args.output.write_text(json.dumps(make_requests(tokenizer), indent=2) + "\n")
