#!/usr/bin/env python3
"""Write a bounded A/B/C profile for the immutable published 64K image.

Standard library only. This does not start a server, load a model, or change the
installed plugin. Arm D requires an unreleased adapter and is not generated.
"""

import argparse
import hashlib
import json
from pathlib import Path


GDN_FLAGS = (
    "PAITON_EXPERIMENTAL_GDN_REPLAY",
    "PAITON_EXPERIMENTAL_GDN_PREFILL",
    "PAITON_EXPERIMENTAL_GDN_CONV_PREFILL",
)


def make_profile(arm):
    source = Path(__file__).resolve().parents[3] / "engine-profile-agentic-64k-v1.json"
    source_bytes = source.read_bytes()
    profile = json.loads(source_bytes)
    profile["profile"] = f"apc-report-arm-{arm.lower()}-64k-c1"
    profile["based_on"] = source.name
    profile["source_profile_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    profile["validation_scope"] = (
        "Bounded APC report reproduction: 65536 context, 5 GiB FP8 cache, "
        "one active sequence; see the 2026-09-17-prefix-caching report."
    )
    arguments = profile["arguments"]
    for flag, value in (
        ("--max-model-len", "65536"),
        ("--max-num-seqs", "1"),
        ("--max-num-batched-tokens", "4096"),
        ("--kv-cache-memory-bytes", "5368709120"),
        ("--mamba-cache-mode", "align" if arm == "C" else "none"),
    ):
        arguments[arguments.index(flag) + 1] = value
    if arm == "C":
        arguments.remove("--no-enable-prefix-caching")
        arguments.append("--enable-prefix-caching")
    draft_index = arguments.index("--speculative-config") + 1
    draft = json.loads(arguments[draft_index])
    draft["max_model_len"] = 65536
    arguments[draft_index] = json.dumps(draft)
    for name in GDN_FLAGS:
        profile["environment"][name] = "1" if arm == "A" else "0"
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B", "C"), default="C")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    profile = make_profile(args.arm)
    # Preserve earlier profiles and evidence rather than silently replacing them.
    with args.out.open("x") as handle:
        json.dump(profile, handle, indent=2)
        handle.write("\n")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
