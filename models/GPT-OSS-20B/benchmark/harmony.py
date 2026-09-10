"""Count actual generated token IDs for the pinned GPT-OSS vocabulary.

Control/header tokens are counted separately. This does not re-tokenize text.
"""


def count_channels(events):
    counts = {
        "reasoning_tokens": 0,
        "final_answer_tokens": 0,
        "other_body_tokens": 0,
        "format_tokens": 0,
        "post_stop_tokens": 0,
    }
    header = True
    channel = None
    want_channel = False
    first_visible = None
    finished = False
    in_format = False
    for timestamp, ids in events:
        for token in ids:
            if finished:
                counts["post_stop_tokens"] += 1
                continue
            if token in (200006, 200007, 200002, 200012):
                header = True
                channel = None
                in_format = False
                want_channel = False
                counts["format_tokens"] += 1
                if token in (200002, 200012):
                    finished = True
            elif token == 200005:
                header = True
                want_channel = True
                in_format = False
                counts["format_tokens"] += 1
            elif token == 200003:
                header = True
                in_format = True
                counts["format_tokens"] += 1
            elif token == 200008:
                header = False
                counts["format_tokens"] += 1
            elif header:
                counts["format_tokens"] += 1
                if want_channel:
                    channel = {35644: "analysis", 17196: "final"}.get(token, "other")
                    want_channel = False
                elif channel is not None and not in_format:
                    channel = "other"
            else:
                key = {
                    "analysis": "reasoning_tokens",
                    "final": "final_answer_tokens",
                }.get(channel, "other_body_tokens")
                counts[key] += 1
                if channel == "final" and first_visible is None:
                    first_visible = timestamp
    return dict(counts, first_visible_answer_s=first_visible)
