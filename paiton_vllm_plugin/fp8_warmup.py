"""Start-time item A11, in-worker half: compile the fp8 block-scale GEMM for every shape class the server can produce.

The DFlash drafter's fp8 linears run aiter's Triton `_gemm_a8w8_blockscale_preshuffle_kernel`. Its code object depends on
the config aiter picks per M bucket (M_LEQ_8 … M_LEQ_512, any, per N/K) and on Triton's specialization of the integer
argument M (== 1, divisible by 16, other), so a fresh server compiles it again for every new (bucket, class) a request mix
produces: 0.3-0.45 s stalls during the first minutes after a start, exactly when the plugin's JIT monitor warns. This
runs, after vLLM's own kernel warm-up and before graph capture, one forward per fp8 block-quant Linear (target and
drafter) for EVERY M <= max_num_seqs x (num_speculative_tokens + 1) (every row count a decode/verify step can produce; no
bucketing assumptions, so split-K or other M-derived kernel parameters are covered too) plus the config-bucket boundaries up
to max_num_batched_tokens, each in a %16 and a non-%16 variant. Same kernels, same arguments as production, so outputs do
not change; the cost is paid once per cache namespace. PAITON_FP8_WARMUP=0 disables it.
"""
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _models(worker):
    runner = worker.model_runner
    out = [("target", runner.model)]
    drafter = getattr(runner, "drafter", None)
    if drafter is not None:
        for name, value in vars(drafter).items():
            if isinstance(value, torch.nn.Module):
                out.append((f"drafter.{name}", value))
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
    ms = set(range(1, m_decode + 1))
    for b in (65, 128, 129, 256, 257, 512, 513, 1024, 2048, 4096):
        if b <= m_max:
            ms.add(b)
            ms.add(min(m_max, b + (-b) % 16))                     # the next multiple of 16 in the same bucket
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
    done = 0
    with torch.inference_mode():
        for name, mod, n, k, m in plan:
            x = torch.zeros(m, k, dtype=dtype, device=device)
            out = mod(x)
            del out, x
            done += 1
        torch.cuda.synchronize(device)
    shapes = sorted({(n, k) for _, _, n, k, _ in plan})
    logger.info("[paiton.warmup] fp8 block-scale GEMM warm-up: %d linears, %d (N,K) shapes %s, M <= %d exhaustive + bucket "
                "boundaries to %d (%d M values), %d forwards in %.1f s",
                len(layers), len(shapes), shapes, m_decode, m_max, len(ms), done, time.time() - t0)
    return {"layers": len(layers), "shapes": shapes, "calls": done, "seconds": round(time.time() - t0, 1)}
