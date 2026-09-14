"""Closed-loop streamed benchmark with retained requests, events and telemetry.

Aggregate throughput is actual output tokens / wall time from the first
request submission to the last completion. Per-request output rate is output
tokens / end-to-end latency. Decode rate is (output tokens - 1) / (last-token
time - first-token time). TTFT excludes empty/header/usage-only stream chunks.
TPOT is (end-to-end latency - TTFT) / (output tokens - 1), so it includes final
server/client completion overhead. ITL is reported only for single-token
chunks; token counts come from server usage and are checked against logprobs.
"""

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import time

import aiohttp


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    def pct(p):
        x = (len(values) - 1) * p
        lo = int(x)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)
    return {"n": len(values), "mean": statistics.mean(values), "p50": pct(.5),
            "p90": pct(.9), "p95": pct(.95), "p99": pct(.99),
            "min": values[0], "max": values[-1]}


def gpu_device(address=None):
    root = Path("/sys/bus/pci/devices")
    if address:
        candidate = root / address
        if not candidate.joinpath("mem_info_vram_used").is_file():
            raise ValueError(f"No AMD GPU telemetry at {candidate}")
        return candidate
    candidates = []
    for candidate in root.iterdir():
        if candidate.joinpath("mem_info_vram_used").is_file() and candidate.joinpath("vendor").read_text().strip() == "0x1002":
            candidates.append(candidate)
    if len(candidates) != 1:
        raise ValueError("Set --gpu-pci-device to select one AMD GPU for telemetry")
    return candidates[0]


def telemetry(base):
    result = {"unix_s": time.time(), "monotonic_s": time.perf_counter()}
    names = ["mem_info_vram_used", "gpu_busy_percent", "mem_busy_percent",
             "power_dpm_force_performance_level", "pp_dpm_sclk", "pp_dpm_mclk"]
    for name in names:
        try:
            result[name] = (base / name).read_text().strip()
        except OSError:
            pass
    for hw in base.glob("hwmon/hwmon*"):
        for pattern in ["temp*_input", "freq*_input", "power1_average", "fan1_input"]:
            for path in hw.glob(pattern):
                try:
                    result[path.name] = int(path.read_text())
                except (ValueError, OSError):
                    pass
    return result


async def request(session, url, item, epoch):
    start = time.perf_counter()
    record = {"id": item["id"], "body": item["body"], "start_s": start - epoch,
              "events": [], "text": "", "usage": None, "error": None}
    try:
        async with session.post(url + "/v1/completions", json=item["body"]) as response:
            response.raise_for_status()
            async for line in response.content:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    continue
                event = json.loads(data)
                record["events"].append({"elapsed_s": time.perf_counter() - start, "data": event})
                if event.get("usage"):
                    record["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    record["text"] += choice.get("text", "")
                    if choice.get("finish_reason"):
                        record["finish_reason"] = choice["finish_reason"]
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["e2e_s"] = time.perf_counter() - start
    token_events = []
    for event in record["events"]:
        choices = event["data"].get("choices", [])
        count = sum(len((c.get("logprobs") or {}).get("tokens", [])) for c in choices)
        if count:
            token_events.append((event["elapsed_s"], count))
    if record["usage"] and token_events:
        usage = record["usage"]
        count = usage["completion_tokens"]
        record["token_count_match"] = count == sum(n for _, n in token_events)
        record["input_count_match"] = usage["prompt_tokens"] == len(item["body"]["prompt"])
        record["ttft_s"] = token_events[0][0]
        record["tpot_s"] = (record["e2e_s"] - record["ttft_s"]) / max(count - 1, 1)
        record["output_tokens_s"] = count / record["e2e_s"]
        record["decode_tokens_s"] = (count - 1) / max(token_events[-1][0] - token_events[0][0], 1e-9)
        record["single_token_itl_s"] = [b[0] - a[0] for a, b in zip(token_events, token_events[1:]) if a[1] == b[1] == 1]
        record["multi_token_chunks"] = sum(n > 1 for _, n in token_events)
    return record


async def run(args):
    items = json.loads(args.requests.read_text())
    device = gpu_device(args.gpu_pci_device)
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for item in items[:args.warmup]:
            warm = await request(session, args.url, item, time.perf_counter())
            if warm["error"]:
                raise RuntimeError(warm["error"])
        for run_index in range(1, args.runs + 1):
            epoch = time.perf_counter()
            samples = []
            stop = asyncio.Event()
            async def monitor():
                while not stop.is_set():
                    samples.append(telemetry(device))
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=.5)
                    except TimeoutError:
                        pass
            mon = asyncio.create_task(monitor())
            queue = asyncio.Queue()
            for i, item in enumerate(items):
                queue.put_nowait((i, item))
            records = [None] * len(items)
            async def worker():
                while not queue.empty():
                    i, item = queue.get_nowait()
                    records[i] = await request(session, args.url, item, epoch)
            await asyncio.gather(*(worker() for _ in range(args.concurrency)))
            elapsed = time.perf_counter() - epoch
            stop.set()
            await mon
            good = [r for r in records if r["error"] is None and r.get("token_count_match") and r.get("input_count_match")]
            output_tokens = sum(r["usage"]["completion_tokens"] for r in good)
            summary = {
                "label": args.label, "run": run_index, "concurrency": args.concurrency,
                "gpu_pci_device": device.name,
                "elapsed_s": elapsed, "completed_requests": len(good), "requested": len(items),
                "aggregate_output_tokens_s": output_tokens / elapsed,
                "completed_requests_s": len(good) / elapsed, "actual_output_tokens": output_tokens,
                "actual_input_tokens": sum(r["usage"]["prompt_tokens"] for r in good),
                "length_finishes": sum(r.get("finish_reason") == "length" for r in good),
                "errors_or_count_mismatches": len(records) - len(good),
                "peak_sampled_vram_bytes": max((int(s.get("mem_info_vram_used", 0)) for s in samples), default=0),
            }
            for metric in ["e2e_s", "ttft_s", "tpot_s", "output_tokens_s", "decode_tokens_s"]:
                summary[metric] = distribution([r[metric] for r in good])
            summary["single_token_itl_s"] = distribution([v for r in good for v in r["single_token_itl_s"]])
            dest = args.output / f"{args.label}-c{args.concurrency}-r{run_index}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps({"summary": summary, "requests": records, "telemetry": samples}, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
            if len(good) != len(items):
                raise RuntimeError(f"incomplete run: see {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--concurrency", type=int, choices=[1, 2], required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--gpu-pci-device", help="PCI address for telemetry; auto-selects a single AMD GPU")
    asyncio.run(run(parser.parse_args()))
