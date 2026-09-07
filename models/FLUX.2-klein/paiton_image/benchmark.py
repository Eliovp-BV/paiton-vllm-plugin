"""Complete generation timings and retained quality/telemetry evidence."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import threading
import time
import torch
from .pipeline import load_pipeline, generate, download_model, MODEL, REVISION, ARTIFACT_DIR

PROMPTS = [
    "A small red fox standing on a mossy stone bridge in a misty woodland at sunrise, soft golden light, detailed natural fur, cinematic wildlife photograph.",
    'A clean studio product photograph of a teal ceramic coffee cup beside a yellow lemon on a cream background, soft window light from the left, a small card that reads "PAITON", realistic shadows.',
    "A cozy bookshop on a rainy evening, warm light through tall windows, a cyclist wearing a red raincoat outside, watercolor and ink illustration, carefully composed wide view.",
]


def telemetry():
    row = {"monotonic_s": time.perf_counter(), "unix_s": time.time()}
    for device in Path("/sys/class/drm").glob("card*/device"):
        try:
            if (device / "device").read_text().strip() != "0x7551":
                continue
        except OSError:
            continue
        for name in ("mem_info_vram_used", "gpu_busy_percent", "power_dpm_force_performance_level", "pp_dpm_sclk", "pp_dpm_mclk", "pp_power_profile_mode"):
            try:
                row[name] = (device / name).read_text().strip()
            except OSError:
                pass
        for hwmon in (device / "hwmon").glob("hwmon*"):
            for pattern in ("temp*_input", "temp*_label", "power*_average", "freq*_input"):
                for path in hwmon.glob(pattern):
                    try:
                        row[path.name] = path.read_text().strip()
                    except OSError:
                        pass
    return row


def run_benchmark(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    snapshot = download_model()
    download_seconds = time.perf_counter() - started
    manifest = {"model": MODEL, "revision": REVISION, "backend": args.backend, "argv": sys.argv,
                "stock_weight_freezing": False, "suite": args.suite,
                "prompt_encoder_outputs": [9, 18, 27],
                "prompt": PROMPTS[args.prompt_index], "seed": args.seed, "resolution": [1024, 1024],
                "steps": 4, "guidance": 1.0, "batch": 1, "max_sequence_length": 512, "offload": False,
                "attention": "INT8 QK, BF16 PV, FP32 softmax", "download_or_cache_check_seconds": download_seconds,
                "versions": {n: importlib.metadata.version(n) for n in ("torch", "triton", "diffusers", "transformers", "accelerate", "huggingface-hub", "safetensors")},
                "device": str(torch.cuda.get_device_properties(0)), "initial_telemetry": telemetry(),
                "pipeline_source_sha256": hashlib.sha256(Path(__file__).with_name("pipeline.py").read_bytes()).hexdigest()}
    manifest["artifacts"] = [json.loads(path.read_text())["artifact"] for path in sorted(ARTIFACT_DIR.glob("*.manifest.json"))]
    conversion = json.loads((snapshot / "conversion.json").read_text())
    (args.output / "conversion.json").write_text(json.dumps(conversion, indent=2))
    stop = threading.Event()

    def poll():
        with (args.output / "telemetry.jsonl").open("w") as out:
            while not stop.is_set():
                out.write(json.dumps(telemetry()) + "\n")
                out.flush()
                stop.wait(0.25)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        started = time.perf_counter()
        pipe = load_pipeline(args.backend, snapshot)
        torch.cuda.synchronize()
        manifest["load_seconds"] = time.perf_counter() - started
        manifest["resident_allocated_bytes"] = torch.cuda.memory_allocated()
        manifest["resident_reserved_bytes"] = torch.cuda.memory_reserved()
        manifest["component_parameter_bytes"] = {name: sum(t.numel()*t.element_size() for t in getattr(pipe, name).parameters())
                                                  for name in ("transformer", "text_encoder", "vae")}
        manifest["component_buffer_bytes"] = {name: sum(t.numel()*t.element_size() for t in getattr(pipe, name).buffers())
                                               for name in ("transformer", "text_encoder", "vae")}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        events = []
        for name, module, method in (("text_encoder", pipe.text_encoder, "forward"), ("transformer", pipe.transformer, "forward"), ("vae_decode", pipe.vae, "decode")):
            original = getattr(module, method)

            def timed(*a, _original=original, _name=name, **kw):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                result = _original(*a, **kw)
                end.record()
                events.append((_name, begin, end))
                return result
            setattr(module, method, timed)
        final_latents = None

        def retain(_pipe, _step, _timestep, values):
            nonlocal final_latents
            final_latents = values["latents"].detach()
            return values

        rows = []
        cases = [(0, 42), (1, 31415), (2, 2026)] if args.suite else [(args.prompt_index, args.seed)]
        manifest["cases"] = [{"prompt_index": i, "prompt": PROMPTS[i], "seed": seed} for i, seed in cases]
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        for total_index in range(4 * len(cases)):
            index = total_index % 4
            prompt_index, seed = cases[total_index // 4]
            case_output = args.output / f"prompt-{prompt_index}" if args.suite else args.output
            case_output.mkdir(exist_ok=True)
            phase = "warmup" if index < 2 else "measured"
            events.clear()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = generate(pipe, PROMPTS[prompt_index], seed, retain)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            components = {}
            for name, begin, end in events:
                components[name] = components.get(name, 0.) + begin.elapsed_time(end)/1000
            finite = bool(torch.isfinite(final_latents).all())
            if not finite:
                raise RuntimeError("Non-finite final latents; reject this benchmark")
            row = {"index": index, "phase": phase, "seconds": elapsed, "begin_monotonic_s": start,
                   "prompt_index": prompt_index, "seed": seed,
                   "end_monotonic_s": start+elapsed, "component_seconds": components,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "host_max_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "latents_finite": finite}
            result.images[0].save(case_output / f"{phase}-{index}.png")
            if phase == "measured":
                torch.save(final_latents.cpu(), case_output / f"latents-{index}.pt")
            rows.append(row)
            (args.output / "timings.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
    finally:
        stop.set()
        thread.join(timeout=2)
