# Paiton: earlier announcements

The "What's new" notes that used to be on the [front page](README.md), newest first. Each model's earlier results
are in its own history file: [Qwen3.8 27B](models/Qwen3.8-MXFP4-DFlash2/HISTORY.md),
[Qwen3.8 Flash Next](models/Qwen3.8-Flash-Next/RELEASE-NOTES.md), [Qwen-Image-2.1](models/Qwen-Image-2.1/HISTORY.md).
Releases with their notes: [GitHub Releases](https://github.com/Eliovp-BV/paiton-vllm-plugin/releases).

## What's new (11 November)

Kimi-K3.1-Pro community build: Moonshot AI's 1.56 TB model on 4 AMD Instinct MI355X with one container. It is plain
PyTorch with no Paiton engine, so it is slow (2 to 5 seconds per token, one request at a time), and it is meant for
trying the model and reading its forward pass. [Guide](models/Kimi-K3.1-Pro/README.md) ·
[benchmarks against Paiton](models/Kimi-K3.1-Pro/BENCHMARKS.md)

Update 12 November: the build read KDA's decay scale `A_log` as one value per channel; the checkpoint stores one value
per head (96 values, padded to 128 with zeros). Short prompts were only slightly off, but answers to long prompts
(about 100K tokens and up) lost track of the question. `model.py` now reads it per head, and `tests/test_tiny.py`
checks it. The container image will be rebuilt with the fix.

## What's new (7 November)

Qwen3.8 27B has new 3-bit weights, retuned on coding-agent conversations. On 80 SWE-bench Verified tasks the agent looped
on 7 tasks instead of 16 (borderline, p 0.078) and solved 49 instead of 41 (not significant on 80 tasks, p 0.12). GSM8K,
HumanEval and MMLU-Pro are unchanged within noise. Run the 3-bit download again.

Update 8 November: on all 500 SWE-bench Verified tasks the new 3-bit weights solved 323 against 284 for the 24 October
weights (p 0.0006), and 165 against 134 on the 250 tasks held out from training (p 0.0002). Loops fell from 89 to 67
tasks. The 16-bit model solved 390, so a significant gap remains. See the
[27B guide](models/Qwen3.8-MXFP4-DFlash2/HISTORY.md#3-bit-weights-fewer-agent-loops).

The new 27B image, r2, moves the speculative-decoding drafter and parts of the 3-bit model to native HIP kernels. Each
decode step is about 3.5 % cheaper and overall speed is about the same. `prompt_logprobs` no longer stops the server.
See the [27B guide](models/Qwen3.8-MXFP4-DFlash2/README.md).

Qwen3.8 Flash Next can now keep prefix-cached sessions on an NVMe/SSD with `--extend-cache` (opt-in), so they survive a
server restart. After a restart, 33K-90K-token sessions on the 4-bit came back in about 1 s instead of 5 to 15 s.
See [session hibernation](models/Qwen3.8-Flash-Next/README.md#session-hibernation-extend-cache).

## What's new (24 October)

- Qwen3.8 Flash Next has new 3-bit weights: GSM8K, HumanEval and MMLU-Pro scores with no significant difference to the
  16-bit model, at full speed from VRAM. The launcher downloads them automatically into a new folder. [3-bit page](models/Qwen3.8-Flash-Next/README-3bit.md)
- Qwen3.8 Flash Next 4-bit gets prefix caching with speculative decoding (`--fn-worker native --prefix-caching`):
  262K-token window, repeated long prompts in under 1 s. [Flash Next guide](models/Qwen3.8-Flash-Next/README.md)
- Qwen3.8 27B 3-bit accuracy update: closer to the 16-bit model, same speed, task scores unchanged within noise. Run
  the 3-bit download again. [27B guide](models/Qwen3.8-MXFP4-DFlash2/README.md)
