"""Verify paired request settings and summarize retained complete-inference runs."""

import argparse
import json
from pathlib import Path

from benchmark import distribution


def tokens(record):
    result = []
    for event in record["events"]:
        for choice in event["data"].get("choices", []):
            logprobs = choice.get("logprobs") or {}
            result.extend(zip(logprobs.get("tokens", []), logprobs.get("token_logprobs", [])))
    return result


def summarize(root):
    result = {"settings": {}, "comparisons": {}}
    for concurrency in [1, 2]:
        paired = {}
        for label in ["stock-final", "paiton-final"]:
            runs = [json.loads((root / f"{label}-c{concurrency}-r{run}.json").read_text()) for run in [1, 2]]
            records = [r for run in runs for r in run["requests"]]
            for run in runs:
                summary = run["summary"]
                assert summary["completed_requests"] == summary["requested"] == 16
                assert summary["actual_output_tokens"] == 4096
                assert summary["errors_or_count_mismatches"] == 0
            elapsed = sum(run["summary"]["elapsed_s"] for run in runs)
            samples = [sample for run in runs for sample in run["telemetry"]]
            data = {"run_output_tokens_s": [run["summary"]["aggregate_output_tokens_s"] for run in runs],
                    "aggregate_output_tokens_s": 8192 / elapsed,
                    "completed_requests_s": len(records) / elapsed,
                    "requests": len(records), "input_tokens": sum(r["usage"]["prompt_tokens"] for r in records),
                    "output_tokens": 8192,
                    "peak_sampled_vram_bytes": max(int(s["mem_info_vram_used"]) for s in samples),
                    "telemetry_samples": len(samples)}
            for metric in ["e2e_s", "ttft_s", "tpot_s", "output_tokens_s", "decode_tokens_s"]:
                data[metric] = distribution([r[metric] for r in records])
            data["telemetry"] = {key: distribution([s[key] for s in samples if key in s])
                for key in ["temp1_input", "temp2_input", "temp3_input", "freq1_input", "freq2_input", "power1_average"]}
            result["settings"][f"{label}-c{concurrency}"] = data
            paired[label] = records
        differences = []
        prefix_lengths = []
        identical = 0
        for stock, paiton in zip(paired["stock-final"], paired["paiton-final"]):
            assert stock["id"] == paiton["id"] and stock["body"] == paiton["body"]
            assert stock["usage"]["prompt_tokens"] == paiton["usage"]["prompt_tokens"]
            assert stock["usage"]["completion_tokens"] == paiton["usage"]["completion_tokens"] == 256
            a, b = tokens(stock), tokens(paiton)
            assert len(a) == len(b) == 256
            n = 0
            for left, right in zip(a, b):
                if left[0] != right[0]:
                    break
                n += 1
                differences.append(abs(left[1] - right[1]))
            prefix_lengths.append(n)
            identical += a == b
        stock = result["settings"][f"stock-final-c{concurrency}"]
        paiton = result["settings"][f"paiton-final-c{concurrency}"]
        result["comparisons"][str(concurrency)] = {
            "throughput_gain_percent": (paiton["aggregate_output_tokens_s"] / stock["aggregate_output_tokens_s"] - 1) * 100,
            "mean_latency_change_percent": (paiton["e2e_s"]["mean"] / stock["e2e_s"]["mean"] - 1) * 100,
            "common_generated_prefix_tokens": distribution(prefix_lengths),
            "common_prefix_logprob_abs_difference": distribution(differences),
            "identical_tokens_and_logprobs_requests": identical,
            "scope": "Common-prefix logprobs only; not a full-vocabulary or teacher-forced numerical equivalence test."}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.evidence), indent=2) + "\n")
