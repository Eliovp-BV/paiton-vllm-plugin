"""Start-time item A11, in-worker half: compile the fp8 block-scale GEMM for every shape class the server can produce.

The DFlash drafter's fp8 linears run aiter's Triton `_gemm_a8w8_blockscale_preshuffle_kernel`. Its code object depends on
the config aiter picks per M bucket (M_LEQ_8 … M_LEQ_512, any, per N/K) and on Triton's specialization of the integer
argument M (== 1, divisible by 16, other), so a fresh server compiles it again for every new (bucket, class) a request mix
produces: 0.3-0.45 s stalls during the first minutes after a start, exactly when the plugin's JIT monitor warns. This
runs, after vLLM's own kernel warm-up and before graph capture, one forward per fp8 block-quant Linear (target and
drafter) for EVERY M <= max_num_seqs x (num_speculative_tokens + 1) (every row count a decode/verify step can produce) and,
up to max_num_batched_tokens, every multiple of 16 and its successor: the compiled variant also depends on the grid constant
GRID_MN = cdiv(M, BLOCK_SIZE_M) x cdiv(N, BLOCK_SIZE_N), one per band of BLOCK_SIZE_M rows and per integer class of M. Same kernels, same arguments as production, so outputs do
not change; the cost is paid once per cache namespace. PAITON_FP8_WARMUP=0 disables it.
"""
import os
import time

import torch

from vllm.logger import init_logger

# vLLM installs its handlers on the 'vllm' logger hierarchy only; a plugin-named logger would print nothing below WARNING
logger = init_logger('vllm.paiton.warmup')


def _models(worker):
    runner = worker.model_runner
    out = [("target", runner.model)]
    for attr in ("drafter", "speculator"):          # the old runner's drafter, the new runner's speculator (DFlash2)
        holder = getattr(runner, attr, None)
        if holder is None:
            continue
        if isinstance(holder, torch.nn.Module):
            out.append((attr, holder))
        for name, value in vars(holder).items():
            if isinstance(value, torch.nn.Module):
                out.append((f"{attr}.{name}", value))
    return out


def _fp8_block_layers(models):
    layers = []
    for owner, model in models:
        for name, mod in model.named_modules():
            qm = getattr(mod, "quant_method", None)
            if qm is None or not getattr(qm, "block_quant", False):
                continue
            n = getattr(mod, "output_size_per_partition", None) or getattr(mod, "output_size", None)
            k = getattr(mod, "input_size_per_partition", None) or getattr(mod, "input_size", None)
            if n and k:
                layers.append((f"{owner}.{name}", mod, int(n), int(k)))
    return layers


def _m_values(worker):
    cfg = worker.vllm_config
    sched = cfg.scheduler_config
    spec = cfg.speculative_config
    k = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec is not None else 0
    m_decode = int(sched.max_num_seqs) * (k + 1)                 # verify rows of a decode step; >= the drafter's block rows
    m_max = max(int(sched.max_num_batched_tokens), m_decode)
    # The kernel's compiled variant depends on M through the aiter config bucket, Triton's integer specialisation of M
    # (1, multiple of 16, other) and the heuristic constant GRID_MN = cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N), i.e.
    # one variant per band of BLOCK_SIZE_M (16 or 64 here) rows and per class. Every M up to m_decode, then every multiple
    # of 16 and its successor up to m_max cover all bands in both classes; already-compiled variants cost a cache hit.
    ms = set(range(1, m_decode + 1))
    for b in range(16, m_max + 1, 16):
        ms.add(b)
        ms.add(min(m_max, b + 1))
    ms.add(m_max)
    return sorted(ms), m_decode, m_max


def fp8_blockscale_warmup(worker):
    if os.environ.get("PAITON_FP8_WARMUP", "1") != "1":
        return None
    t0 = time.time()
    layers = _fp8_block_layers(_models(worker))
    if not layers:
        logger.info("[paiton.warmup] fp8 block-scale warm-up: no fp8 block-quant linear found")
        return {"layers": 0}
    ms, m_decode, m_max = _m_values(worker)
    plan = [(name, mod, n, k, m) for name, mod, n, k in layers for m in ms]
    device = worker.device
    dtype = getattr(worker.model_config, "dtype", torch.bfloat16)
    cuda = torch.cuda.is_available() and torch.device(device).type == "cuda"
    mem = {}
    if cuda:  # the warm-up runs after the KV pool is allocated: record its own allocator peak against the cap
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        mem["allocated_before_mib"] = round(torch.cuda.memory_allocated(device) / 2**20)
        mem["reserved_before_mib"] = round(torch.cuda.memory_reserved(device) / 2**20)
    t_start = time.time()
    done = 0
    with torch.inference_mode():
        for name, mod, n, k, m in plan:
            x = torch.zeros(m, k, dtype=dtype, device=device)
            out = mod(x)
            del out, x
            done += 1
        if cuda:
            torch.cuda.synchronize(device)
    t_end = time.time()
    if cuda:
        mem["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20)
        mem["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved(device) / 2**20)
        mem["reserved_after_mib"] = round(torch.cuda.memory_reserved(device) / 2**20)
        try:
            total = torch.cuda.get_device_properties(device).total_memory
            mem["cap_mib"] = round(torch.cuda.get_per_process_memory_fraction(device) * total / 2**20)
        except Exception:  # noqa: BLE001
            pass
    logger.info("[paiton.warmup] fp8 block-scale GEMM warm-up window epoch %.3f-%.3f memory %s", t_start, t_end, mem)
    shapes = sorted({(n, k) for _, _, n, k, _ in plan})
    logger.info("[paiton.warmup] fp8 block-scale GEMM warm-up: %d linears, %d (N,K) shapes %s, M <= %d exhaustive + bucket "
                "boundaries to %d (%d M values), %d forwards in %.1f s",
                len(layers), len(shapes), shapes, m_decode, m_max, len(ms), done, time.time() - t0)
    return {"layers": len(layers), "shapes": shapes, "calls": done, "seconds": round(time.time() - t0, 1), "memory": mem}
