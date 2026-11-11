# Kimi-K3.1-Pro community build: benchmarks

Measured 11 November 2026 on one node with 8 AMD Instinct MI355X (288 GB each), 2 AMD EPYC 9575F and 2.2 TB of RAM.
The community build ran on 4 of the GPUs, **each capped at 800 W** (rated 1,400 W). Temperature 0.

## Community build against Paiton on the same 4 GPUs

| 4x MI355X, 800 W per GPU | Community build | Paiton | Paiton is |
| --- | --- | --- | --- |
| One request, time per output token | 1.9-4.6 s | 26 ms | 70-180x faster |
| Prefill speed | ~5 tok/s (1,175 tokens in 239 s) | ~3,200 tok/s (8K tokens in 2.6 s) | ~650x faster |
| Requests served at once | 1 | 16 | 16x more |
| 8K/1K throughput, input + output tokens | ~2 tok/s (one request at a time) | 911 tok/s (16 at once) | ~450x more |
| One 8K-in, 1K-out request, start to finish | about 1 to 2 hours (estimate) | 29 s | |
| Model load | 6-25 min | a few minutes | |

The 8K/1K rows for the community build are estimated from its measured prefill and decode speed; we did not time a
two-hour request. Paiton's numbers come from AMD's benchmark method (`vllm bench serve`, random 8K-token prompts, 1K
output tokens, 16 concurrent requests) on the same GPUs. The time per token varies with how much of the host-memory
expert traffic is already in the page cache: 4.6 s on the first request after loading, 1.9 s once warm.

## Correctness checks

Answers from the community build on 4x MI355X, greedy, thinking off:

| Prompt | Answer | Time |
| --- | --- | --- |
| What is the capital of France? Answer in one word. | Paris | 434 s (first request after loading) |
| Write a haiku about GPUs. | Silicon hearts race fast, / Pixels bloom in silent streams— / Dreams in parallel. | 179 s |
| GSM8K: Natalia sold clips to 48 friends in April and half as many in May. How many altogether? | 72 | 145 s |
| Count from 1 to 5 (streamed, 12-token limit) | 1, 2, 3, 4, | first text after 47 s |

`tests/test_tiny.py` checks that a prompt run at once and the same prompt run token by token give the same logits
(relative error 3e-7 on CPU). The forward pass matches our internal fp32 reference implementation of Kimi-K3.1-Pro
exactly on the same tiny checkpoint.
