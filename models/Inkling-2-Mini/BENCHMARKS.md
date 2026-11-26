# Inkling 2 Mini benchmarks

Measured on 25 and 26 November 2026 on one Radeon AI PRO R9700 (gfx1201, 32 GB) at a 300 W power cap, with the image
`inkling-2-mini-rdna4-v1.0.0` (ROCm 10.2, vLLM 0.32) and the checkpoint `thinkingmachines/Inkling-2-Mini` at
revision `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`, with the launcher's defaults unless a row says otherwise: the
4-bit cache (`--kv-cache kv4`) and speculative decoding with the default drafter (its checkpoint and revision are in
[NOTICE.md](NOTICE.md)).

vLLM has no path for this model on the R9700, so there is no stock number to compare with. Every comparison here is
with Paiton's own first complete build of 24 November ("first build" below): the same weights and arithmetic on
Paiton's first compiled expert, attention and operator libraries, with the layers called one by one from Python and
without the drafter.

## How the numbers were taken

- Every arm is a fresh container and a fresh server. Comparisons run in mirrored order (old, new, new, old) and the
  tables show both runs of each arm.
- Each table says whether its requests are greedy (temperature 0) or at temperature 1.0, the model's default
  (its `generation_config.json`: temperature 1.0, no top-p or top-k, which the server uses when a request sets none).
  Sampled runs use seed 1, and their replies differ from the greedy ones in length.
- "Real text" is 11 requests: chat, code and tool-call prompts with 64 to 160 new tokens each, and one 2,400-token
  prompt with 64 new tokens. The first request or group warms the server up and is not counted. Tokens per second
  is the number of generated tokens divided by the wall time of the requests, prompt processing included.
- `vllm bench serve` is vLLM's benchmark client in the image: real-text prompts (Python standard library source
  and a coding task) that each start with their own line, so no request reuses another's cached tokens, and
  replies run to 512 tokens. The guide's headline speeds come from it.
- The numbers come from a build with the same native libraries as the release image. Run again on the release
  image, the `vllm bench serve` cells came within 0.4 % of them and the 1,038,311-token prompt read at 1,174 tok/s
  against 1,172; the 1,038,311-token row is the release image's own run.
- No other job ran on this card during an arm. The second card of the host ran other work at the same time.
- Compare numbers within one card: the same image and arm ran 2.6 % faster on the host's second R9700 than on the
  first. Every table here comes from one card; the `vllm bench serve` table is from the second card.

## Context and cache formats, greedy

The real-text set with each `--kv-cache` format, mirrored (fp8, bf16, bf16, fp8; the `kv4` row is the default's
mirror against an earlier build):

| `--kv-cache` | Tokens per request | 11 requests, tok/s at 1 / 4 | Tokens per check step at 1 / 4 |
| --- | ---: | --- | --- |
| `kv4` (default) | 1,048,576 | 142.7 / 240.7 and 142.6 / 241.0 | 4.03 / 3.68 |
| `fp8` | 558,496 | 140.1 / 237.1 and 140.6 / 238.1 | 3.95 / 3.68 |
| `bf16` | 279,248 | 146.5 / 244.2 and 146.7 / 243.1 | 4.15 / 3.81 |

Ten facts hidden in a long text, one request, reply budget 4,000 tokens, one run each. Each cell: prompt speed,
facts found, generation speed after the prompt:

| `--kv-cache` | 260,049 tokens | 522,196 tokens | 1,038,311 tokens |
| --- | --- | --- | --- |
| `kv4` (default) | 2,436 tok/s, 10 of 10, 136.1 tok/s | 1,772 tok/s, 10 of 10, 125.6 tok/s | 1,174 tok/s, 10 of 10, 122.3 tok/s |
| `fp8` | 2,397 tok/s, 9 of 10, 158.0 tok/s | 1,713 tok/s, 10 of 10, 120.8 tok/s | does not fit |
| `bf16` | 2,380 tok/s, 10 of 10, 120.0 tok/s | does not fit | does not fit |

The fp8 miss at 260,049 tokens is a reply that spent its 4,000-token budget on reasoning and never started the
answer; its reasoning names 9 of the 10 facts.

## `vllm bench serve`, 1,024-token prompts

vLLM's own benchmark client from the image, on the host's second card. Off is `--no-drafter`, on is the default; four
fresh servers in the order off, on, on, off; no prompt tokens came from the prefix cache. Requests per cell: 8 at one
and two clients, 16 at four. Total output tok/s:

| Requests at once | Mode | Off, runs 1 / 4 | On, runs 2 / 3 | On / off | First token p50, off / on | Time per token p50, off / on |
| ---: | --- | --- | --- | ---: | --- | --- |
| 1 | greedy | 68.8 / 68.8 | 151.3 / 151.3 | 2.20x | 290 / 317 ms | 14.0 / 6.0 ms |
| 2 | greedy | 112.1 / 112.1 | 200.9 / 200.8 | 1.79x | 473 / 380 ms | 16.9 / 8.8 ms |
| 4 | greedy | 170.8 / 170.5 | 285.0 / 284.8 | 1.67x | 635 / 417 ms | 22.3 / 12.8 ms |
| 1 | temperature 1.0 | 68.9 / 68.9 | 129.4 / 129.4 | 1.88x | 291 / 300 ms | 14.0 / 7.2 ms |
| 2 | temperature 1.0 | 112.1 / 112.2 | 199.5 / 201.4 | 1.79x | 474 / 377 ms | 16.9 / 9.1 ms |
| 4 | temperature 1.0 | 169.5 / 169.4 | 296.0 / 294.3 | 1.74x | 635 / 461 ms | 22.5 / 12.4 ms |

At four requests the default drafter checks 3.38 (greedy) and 3.49 (temperature 1.0) tokens per step, at one
request 4.07 and 3.40.

## Long prompts, greedy

One request per length, 320 new tokens, one run each (not mirrored), prompt not in the prefix cache:

| Prompt tokens | First token | Prompt speed | Generation, no drafter | Generation, default drafter |
| ---: | ---: | ---: | ---: | ---: |
| 4,285 | 1.1 s | 3,824 tok/s | 69.0 tok/s | 120.7 tok/s |
| 17,225 | 4.4 s | 3,921 tok/s | 68.0 tok/s | 110.8 tok/s |
| 71,411 | 20.3 s | 3,524 tok/s | 64.7 tok/s | 130.3 tok/s |
| 143,840 | 46.9 s | 3,068 tok/s | 61.2 tok/s | 106.6 tok/s |
| 218,915 | 80.6 s | 2,718 tok/s | 57.8 tok/s | 117.1 tok/s |
| 522,196 | 294.7 s | 1,772 tok/s | not run | 125.6 tok/s (2,684 tokens, the fact check above) |

First token and prompt speed are from the run without the drafter; with the drafter they are within 2 %. At every
length up to 218,915 tokens, and at a 43-token prompt, the reply with the drafter is the same text as the reply
without it. Each length is one run with its own reply, so generation speed does not fall steadily with the length.

Four requests of 247,027 prompt tokens each at once (988,108 in the cache together) all finished: the card's memory
peaked at 29.89 of 31.86 GiB, system memory use did not grow and vLLM preempted no request.

BetterBench 0.6.0, standard decode and concurrency runs (temperature 0.7, top-p 0.95, top-k 20, prompts of changing
length), on the release image: card memory peaked at 30.71 GiB (decode, 160 of 160 requests) and 30.63 GiB
(concurrency, 48 of 48 at every level), with the launcher's 1 GiB allocator headroom. Without it the same runs grew
to 31.39 and 31.45 GiB.

## Release image against the first build, real text, greedy

| Arm | 1 request, tok/s | 4 at once, tok/s |
| --- | --- | --- |
| First build | 25.8 / 24.8 | 72.1 / 66.7 |
| Release image, `--no-drafter` | 66.6 / 66.6 | 138.6 / 138.5 |
| Release image (default drafter) | 142.7 / 142.6 | 240.7 / 241.0 |

The default-drafter row comes from a mirrored run against an earlier build of this release, which drew its drafts
greedily: 137.1 / 137.5 and 230.7 / 229.8 tok/s there with the same greedy tokens, so +3.9 % for one request and
+4.6 % for four.

The three drafter settings of the release image, greedy:

| Arm | 1 request, tok/s | 4 at once, tok/s | Tokens per check step |
| --- | --- | --- | --- |
| `--no-drafter` | 66.6 / 66.6 | 138.6 / 138.5 | 1 |
| `--drafter dspark` | 109.3 / 109.4 | 181.1 / 182.2 | 3.02 at one request, 2.88 at four |
| Default drafter | 142.7 / 142.6 | 240.7 / 241.0 | 4.03 at one request, 3.68 at four |

The `--no-drafter` and `--drafter dspark` rows come from one mirrored run (no drafter, DSpark, DSpark, no drafter).

At temperature 1.0, where the model accepts fewer draft tokens, mirrored the same way:

| Arm | 1 request, tok/s | 4 at once, tok/s | Tokens per check step |
| --- | --- | --- | --- |
| `--no-drafter` | 66.6 / 66.6 | 136.7 / 136.7 | 1 |
| `--drafter dspark` | 98.8 / 98.8 | 171.5 / 171.7 | 2.69 at one request, 2.53 at four |
| Default drafter | 137.8 / 137.7 | 231.5 / 231.5 | 3.87 at one request, 3.42 at four |

The default drafter's drafts are sampled, so more of them are accepted at temperature 1.0: the earlier build, which drew its
drafts greedily, gave 116.2 / 116.2 and 198.9 / 199.1 tok/s in the same mirror (3.34 and 3.06 tokens per check step),
so +18.5 % for one request and +16.4 % for four.

In the greedy runs the token ids of all 11 requests are the same with each drafter and without one, alone and four
at once. The launcher drafts five tokens for one request and four when requests run side by side.

## Quality

The weights are the publisher's and run in the arithmetic they were trained in, so quality checks here look for
port faults, not for quantization loss.

- Greedy replies of the served model match the PyTorch reference's token ids on four fixture prompts for the first
  47, 50, 13 and 78 tokens (the full fixture length for the first and last). The image prompt matches for 97
  tokens.
- Task set, 35 tasks, greedy, defaults, through the launcher: **34 of 35**. Arithmetic 10 of 10, code output 10 of
  10, tool calls 9 of 10, needle in a long text 5 of 5 (two at 31,000 tokens, answered in 10.2 and 12.2 s; three at
  130,000 tokens, 45.2 to 46.1 s). The miss is a translation request that the model answers directly and correctly
  without calling the offered tool.
- Ten facts hidden in texts of 260,049, 522,196 and 1,038,311 tokens, greedy,
  defaults: all ten found each time.
- The 4-bit cache against a bf16 reference (our quality study, 2K to 512K-token documents): about 0.005 next-token
  KL, of which 0.001 comes from the 4-bit values; no loss of top-1 agreement or of a needle. fp8 showed no change.
- 8-bit output head (the default): next-token KL against the bf16 head 0.00045 on the check set.
- `--int8-attention` (off by default): KL about 0.003 against the default. Greedy, mirrored against the default on the
  build of 25 November (bf16 cache, the draft head in the model's checkpoint), not measured with the default drafter:

  | Arm | 1 request, tok/s | 4 at once, tok/s |
  | --- | --- | --- |
  | Default, drafter on | 111.9 / 111.7 | 187.1 / 186.8 |
  | `--int8-attention`, drafter on | 115.6 / 115.3 | 194.6 / 193.6 |
  | Default, `--no-drafter` | 68.4 / 68.3 | 134.1 / 134.0 |
  | `--int8-attention`, `--no-drafter` | 78.2 / 78.3 | 151.5 / 151.3 |

  The context gets smaller with it: 235,600 tokens with the drafter (254,224 without the option) and 309,120
  without the drafter (330,736). Prompt speed stays within 1.3 % at 4,285 to 143,840 tokens, and the drafter reply is
  the same text as the plain reply at those lengths. It is not run through the task set and not measured at
  temperature 1.0.

## Limits of these numbers

- One host, one card, one power cap. A card at a higher cap or a faster host will differ.
- The real-text set is 11 requests. It shows the size of an effect; it is too small for a second decimal.
- The long-prompt and fact-check tables are one run per cell.
- Four requests at once is the most we ran.
- The quality checks are greedy. Sampled requests are checked for running and for plausible replies only.
