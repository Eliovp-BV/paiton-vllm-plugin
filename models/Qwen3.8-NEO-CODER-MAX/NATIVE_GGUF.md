# Native GGUF through vLLM on AMD RDNA4

GGUF weights can now run through Paiton’s native AMD execution path inside
vLLM. Our qualified Qwen3.8 NEO CODER MAX release brings the author’s mixed
Q4_K_M fine-tune—including image input—to vLLM’s model loader, scheduler,
sampling and streaming API.

The result is **native GGUF execution through vLLM with slightly lower
complete-request latency medians than a fresh working llama.cpp baseline**
on all three tested 128-output-token text workloads. The long workload is
close to parity. This is a result for one pinned model and deployment profile,
not a claim that every GGUF file or vLLM workload has become faster.

## Same model, a native execution path

GGUF is a container for tensors and metadata, not a single arithmetic format.
This file mixes Q4_K, Q6_K, Q8_0, FP32 and BF16 tensors. Its output head has
higher precision than most projection weights. Treating it as uniform INT4,
or silently replacing it with an AWQ checkpoint, would change the contract.

Paiton reads the author’s packed weights directly and preserves their decoded
values. Native HIP kernels and ROCm libraries execute the compiled language
and vision paths. We did not substitute the base Qwen model, re-quantize the
author’s source checkpoint, or put a llama.cpp server behind an OpenAI API.
Requests enter the actual vLLM integration. llama.cpp is our independent
comparison engine.

The Paiton compiler and its native runtime artifacts do not require PyTorch
or Triton. Existing vLLM and image preprocessing remain in the external
serving stack. The public package contains allowlisted runtime binaries and
metadata; the proprietary compiler and generated implementation source remain
private.

## What improved

We optimized packed-weight decode, bounded prefill matrix operations, graph
replay and the transfer of native results into vLLM. The final profile processes
up to 2,048 prefill tokens per chunk while keeping matrix operations bounded
to the qualified arithmetic path.

That distinction mattered. One experimental profile generated the same greedy
tokens but failed our full-logit tolerance. We rejected it. The selected final
prefill profile preserved all 31,784,960 compared logits exactly against the
preceding qualified profile, plus every compared text token/log-probability
event and image response. Those checks cover this optimization step; they do
not mean every engine or activation profile is bit-equivalent.

The release uses Q8 activation decode for
large Q4 projections. Weight values are unchanged, but activation arithmetic
is different from the FP32 decode reference profile. Its held-out perplexity
increase was 0.1301% over 770 predictions, below the predefined 1% threshold.
That is a narrow measured result, not evidence of zero accuracy difference
on every task.

## The measured result

Complete streaming HTTP request latency, median seconds; fixed output length
of 128 tokens, one active sequence, identical input tokens and greedy sampling:

| Input tokens | Paiton + vLLM | llama.cpp | Paiton latency difference |
| ---: | ---: | ---: | ---: |
| 128 | **4.925** | 5.264 | 6.4% lower |
| 1,024 | **5.631** | 5.933 | 5.1% lower |
| 4,096 | **9.023** | 9.099 | 0.8% lower |

The llama.cpp comparison uses
fresh serialized runs on the same idle Radeon AI PRO R9700 and pinned ROCm
runtime. Both use the same GGUF, tokenizer, sampling controls, context budget
and fixed token counts; engine arithmetic differs. It is a cross-engine
comparison, not an equivalent-kernel experiment.

Image input is also working: a 1,024 × 1,024 image with 128 output tokens took
6.279 seconds versus llama.cpp’s 6.455 seconds. Two queued 128/128 requests
completed in 9.905 seconds versus 10.466 seconds. These medians come from five
measured repetitions after warmup. Some single-output, prefill-dominated cases
still favor llama.cpp. The 0.8% long-request difference is small and should
not be generalized into a substantial speed advantage.

[The benchmark report](BENCHMARKS.md) includes p95 latency, time to first token,
decode throughput, regressions, quality results and reproducibility details.
Output lengths and reasoning settings were held fixed; shorter answers did
not create the reported speedup.

## What you can deploy

The qualified package targets one R9700 / gfx1201 GPU, an 8,192-token total
context and one active sequence with additional requests queued. It supports
text, chat, streaming, reasoning controls, tool parsing and one PNG or JPEG
image. Native language and vision execution are included. MTP, video and prefix
caching are not advertised: speculative draft verification and correct KV/GDN
commit or rollback still need implementation and measurement.

Plan for a roughly 32 GiB GPU. The measured peak device allocation was
23.74 GiB, compared with 19.03 GiB for the reference profile; host process RSS
was 5.11 GiB versus 10.15 GiB. This release prioritizes the measured latency
profile, and its memory contract is part of the deployment requirements.

The [model guide](README.md) provides the immutable image, launch command,
image API example, exact model pins and rollback instructions. This is a concrete native GGUF deployment through
Paiton and vLLM, with a tested support boundary and room to extend it.
