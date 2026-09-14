"""Generate pinned Harmony token requests without truncating control tokens."""

import argparse, json, pathlib
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--out", type=pathlib.Path, required=True)
p.add_argument("--input-tokens", type=int, default=512)
p.add_argument("--output-tokens", type=int, default=256)
p.add_argument("--count", type=int, default=16)
p.add_argument("--date", default="2026-09-09")
a = p.parse_args()
t = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
items = []
for i in range(a.count):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"Request {i}. Explain how a hash table handles collisions, with a small Python example. Background notes follow.\n",
        },
    ]
    ids = t.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort="low",
        return_dict=False,
        strftime_now=lambda _: a.date,
    )
    # Insert neutral content into the user message before its closing control tokens.
    text = t.encode(" Reference note.", add_special_tokens=False)
    n = a.input_tokens - len(ids)
    if n < 0:
        raise ValueError("Input budget too small for complete template")
    # Find final user end marker; preserve the complete assistant generation prefix.
    end_id = t.convert_tokens_to_ids("<|end|>")
    pos = max(j for j, x in enumerate(ids) if x == end_id)
    ids = ids[:pos] + (text * ((n + len(text) - 1) // len(text)))[:n] + ids[pos:]
    assert len(ids) == a.input_tokens
    items.append(
        {
            "id": f"p{a.input_tokens}-o{a.output_tokens}-{i}",
            "body": {
                "model": "gpt-oss-20b",
                "prompt": ids,
                "max_tokens": a.output_tokens,
                "temperature": 0,
                "seed": 1201,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
                "return_token_ids": True,
            },
            "decoded_prompt": t.decode(ids),
            "reasoning_effort": "low",
        }
    )
a.out.write_text(json.dumps(items, indent=2))
