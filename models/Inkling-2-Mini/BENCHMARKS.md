# Inkling 2 Mini benchmarks

Measured on 25 and 26 November 2026 on a single Radeon AI PRO R9700 (gfx1201, 32 GB) at a 300 W power cap,
with the images `inkling-2-mini-rdna4-v1.2.0` (default and `--fast`) and `inkling-2-mini-w3-rdna4-v1.2.0` (super fast)
or the earlier builds a table names (ROCm 10.2, vLLM 0.32), and the checkpoint `thinkingmachines/Inkling-2-Mini` at
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
- The default-setting tables from "Context and cache formats" to "Release image against the first build" come from
  v1.0.0 and the build before it. Run again on v1.0.0, the `vllm bench serve` cells came within 0.4 % of them and the
  1,038,311-token prompt read at 1,174 tok/s against 1,172; the 1,038,311-token rows are v1.0.0's own runs, except the
  fact cell of the cache-format table, which is v1.2.0's. v1.2.0 has
  the same model libraries except the expert library, whose prompt kernels are faster and give the same bits: with
  the default settings its greedy tokens are the same as v1.0.0's on the 11 check requests, alone and four at once.
  A v1.2.0 run of the 1,024-token `vllm bench serve` cells with the drafter on came within 1 % of the "on" columns
  below (151.8 / 129.8 tok/s with one request, 287.7 / 296.2 with four; one run, not mirrored). The
  [fast mode and super fast](#fast-mode---fast-and-super-fast-3-bit-image) numbers come from the v1.2.0 images.
- No other job ran on this card during an arm. The second card of the host ran other work at the same time.
- Compare numbers within the same card: the same image and arm ran 2.6 % faster on the host's second R9700 than on
  the first. All rows of a table come from the same card; the `vllm bench serve` table is from the second card.

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
| `kv4` (default) | 2,436 tok/s, 10 of 10, 136.1 tok/s | 1,772 tok/s, 10 of 10, 125.6 tok/s | 1,185 tok/s, 8 of 10 without an answer, 103.6 tok/s |
| `fp8` | 2,397 tok/s, 9 of 10, 158.0 tok/s | 1,713 tok/s, 10 of 10, 120.8 tok/s | does not fit |
| `bf16` | 2,380 tok/s, 10 of 10, 120.0 tok/s | does not fit | does not fit |

The fp8 miss at 260,049 tokens is a reply that spent its 4,000-token budget on reasoning and never started the
answer; its reasoning names 9 of the 10 facts. The 1,038,311-token `kv4` cell is a v1.2.0 run, on a text with other random facts: greedy, the reasoning named
8 of the 10 facts, then repeated a list until the 4,000-token cap, so no answer came. At temperature 1.0 the reply
gave 9 of 10 (1,187 tok/s, generation 117.4 tok/s).

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
length), on v1.0.0 and again on this image: card memory peaked at 30.71 GiB (decode, 160 of 160 requests) and 30.63 GiB
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

## Fast mode (`--fast`) and super fast (3-bit image)

`--fast` sets 2,048-token prompt steps (default 512) and four drafts while one request runs (default five); the
weights, the arithmetic and the draft check stay those of the default. The launcher then fits 524,800
tokens per request instead of 1,048,576. Super fast, the image `inkling-2-mini-w3-rdna4-v1.2.0`, adds 3-bit experts
(all 24 layers), 8-bit attention projections and the relaxed draft check (`0.1,0.3`, see the
[guide](README.md#super-fast-3-bit-image)), with 1,048,576 tokens per request.

LocalMaxxing's inference matrix (lmx 0.1.66 with the site's settings: its prose prompt at temperature 1.0, 30 s per
cell), one fresh server per version on the same card, one run each (not mirrored). Prompt speed, one request, tok/s:

| Prompt | Default | `--fast` | Super fast |
| ---: | ---: | ---: | ---: |
| 8K | 3,897 | 5,562 | 5,702 |
| 16K | 3,550 | 5,322 | 5,321 |
| 32K | 3,737 | 4,888 | 5,295 |
| 64K | 3,521 | 4,697 | 4,876 |
| 128K | 3,088 | 4,022 | 4,153 |

Generation, total tok/s at 1 / 2 / 4 requests at once:

| Context | Default | `--fast` | Super fast |
| ---: | --- | --- | --- |
| none | 93.9 / 153.0 / 249.7 | 102.4 / 153.1 / 244.0 | 179.7 / 287.3 / 400.5 |
| 4K | 95.2 / 169.6 / 243.0 | 100.0 / 166.8 / 245.5 | 156.3 / 255.2 / 400.8 |
| 16K | 93.1 / 164.1 / 245.4 | 105.8 / 164.0 / 230.1 | 160.5 / 272.3 / 405.8 |
| 32K | 98.3 / 149.8 / 242.6 | 106.3 / 154.8 / 237.1 | 159.2 / 269.9 / 390.8 |
| 64K | 102.3 / 163.6 / 234.6 | 96.6 / 153.9 / 240.8 | 167.5 / 257.8 / 383.1 |
| 128K | 93.8 / 156.7 / 224.2 | 96.8 / 142.6 / - | 149.4 / 240.2 / 363.1 |

Fast mode has no 4-user number at 128K: four requests with 128K tokens of context each need more than its
524,832-token cache.

- Tokens per check step on this prose: 2.49 with the default, 2.44 with `--fast` and 3.09 with super fast (vLLM's draft
  metrics over each whole run; super fast keeps more drafts through its relaxed check). Each context fill starts with its
  own random text, so the generation cells move by a few percent from run to run through the accepted drafts; the
  prompt cells repeat within 0.3 %.
- LocalMaxxing's two short speed tests (512-token replies, one request), `code-v1` / `reasoning-v1`:
  default 164.6 / 169.9 tok/s, the mean of two fresh servers (`code-v1` 163.7 and 165.4 tok/s at draft acceptance
  0.608 and 0.614; `reasoning-v1` 163.1 and 176.8 at 0.580 and 0.646); `--fast` 152.0 / 166.6 (one run, first
  token after 137 / 144 ms); super fast 199.5 / 201.3 (one run, first token after
  109 / 103 ms). The accepted drafts move these short replies by
  several percent from server to server.
- Ten facts hidden in a long text, temperature 1.0: super fast found all ten in a 1,038,311-token prompt (prompt
  788.7 s, 1,317 tok/s; generation 125.5 tok/s, 1,434 tokens). `--fast` at its own context, one run each: at
  temperature 1.0 (516,065 tokens, prompt 238.5 s, 2,164 tok/s) the reasoning named all ten facts, but the 4,000-token
  budget ran out before the answer started, as in the fp8 case above; greedy (516,063 tokens, prompt 241.2 s,
  2,140 tok/s) the reply gave all ten, with 1,203 tokens at 114.4 tok/s.

`vllm bench serve` with `--fast` on the v1.2.0 image, same method as the 1,024-token table above (512-token replies,
new prompts in every cell), one run per prompt length. Total tok/s and latencies:

1,024-token prompts:

| Users | Sampling | Total tok/s | First token p50 / p99 | Time per token p50 / p99 |
| ---: | --- | ---: | --- | --- |
| 1 | greedy | 148.2 | 216 ms / 219 ms | 6.4 / 7.7 ms |
| 1 | temperature 1.0 | 134.0 | 215 ms / 218 ms | 7.1 / 7.8 ms |
| 2 | greedy | 207.7 | 249 ms / 402 ms | 8.9 / 9.7 ms |
| 2 | temperature 1.0 | 205.5 | 250 ms / 415 ms | 9.1 / 10.2 ms |
| 4 | greedy | 298.1 | 276 ms / 748 ms | 12.5 / 13.9 ms |
| 4 | temperature 1.0 | 310.5 | 337 ms / 747 ms | 11.8 / 13.8 ms |
| 8 (4 run, 4 wait) | greedy | 307.8 | 6,700 ms / 8,028 ms | 12.3 / 14.9 ms |
| 8 (4 run, 4 wait) | temperature 1.0 | 301.7 | 6,987 ms / 7,970 ms | 12.7 / 15.1 ms |

8,192-token prompts:

| Users | Sampling | Total tok/s | First token p50 / p99 | Time per token p50 / p99 |
| ---: | --- | ---: | --- | --- |
| 1 | greedy | 99.7 | 1,505 ms / 1,515 ms | 7.4 / 8.0 ms |
| 1 | temperature 1.0 | 92.6 | 1,499 ms / 1,504 ms | 8.0 / 8.8 ms |
| 2 | greedy | 132.6 | 1,562 ms / 2,881 ms | 12.1 / 14.8 ms |
| 2 | temperature 1.0 | 129.7 | 1,568 ms / 2,865 ms | 12.0 / 13.2 ms |
| 4 | greedy | 167.4 | 1,595 ms / 5,658 ms | 19.9 / 22.5 ms |
| 4 | temperature 1.0 | 165.2 | 1,698 ms / 5,667 ms | 18.8 / 23.7 ms |
| 8 (4 run, 4 wait) | greedy | 169.3 | 13.7 s / 17.8 s | 20.3 / 23.6 ms |
| 8 (4 run, 4 wait) | temperature 1.0 | 170.6 | 13.3 s / 17.6 s | 20.0 / 24.0 ms |

32,768-token prompts:

| Users | Sampling | Total tok/s | First token p50 / p99 | Time per token p50 / p99 |
| ---: | --- | ---: | --- | --- |
| 1 | greedy | 50.2 | 6,308 ms / 6,624 ms | 7.3 / 9.6 ms |
| 1 | temperature 1.0 | 48.2 | 6,374 ms / 6,405 ms | 8.5 / 8.5 ms |
| 2 | greedy | 58.4 | 9,030 ms / 12.5 s | 16.7 / 21.6 ms |
| 2 | temperature 1.0 | 57.8 | 9,619 ms / 12.6 s | 15.5 / 22.4 ms |
| 4 | greedy | 65.7 | 10.9 s / 25.6 s | 38.0 / 45.6 ms |
| 4 | temperature 1.0 | 64.5 | 12.6 s / 24.8 s | 41.2 / 49.9 ms |
| 8 (4 run, 4 wait) | greedy | 66.9 | 36.7 s / 55.0 s | 45.9 / 52.9 ms |
| 8 (4 run, 4 wait) | temperature 1.0 | 66.3 | 37.2 s / 55.4 s | 42.7 / 48.8 ms |

The default on the same image, one run, 1,024-token prompts: 151.8 / 129.8 tok/s (greedy / temperature 1.0) with one
request, 287.7 / 296.2 with four; 8,192-token prompts: 90.6 / 86.5 and 147.5 / 145.3. Super fast
on the 3-bit image, one run per prompt length, total tok/s and latencies.

1,024-token prompts:

| Users | Sampling | Total tok/s | First token p50 / p99 | Time per token p50 / p99 |
| ---: | --- | ---: | --- | --- |
| 1 | greedy | 187.4 | 196 ms / 199 ms | 5.1 / 5.3 ms |
| 1 | temperature 1.0 | 201.8 | 190 ms / 199 ms | 4.6 / 5.0 ms |
| 2 | greedy | 272.8 | 224 ms / 368 ms | 6.7 / 7.6 ms |
| 2 | temperature 1.0 | 305.4 | 223 ms / 362 ms | 6.0 / 6.7 ms |
| 4 | greedy | 401.0 | 248 ms / 718 ms | 9.1 / 10.9 ms |
| 4 | temperature 1.0 | 422.3 | 246 ms / 712 ms | 8.6 / 10.2 ms |
| 8 (4 run, 4 wait) | greedy | 402.5 | 5,209 ms / 6,094 ms | 9.3 / 11.3 ms |
| 8 (4 run, 4 wait) | temperature 1.0 | 433.1 | 4,882 ms / 5,501 ms | 8.6 / 10.1 ms |

8,192-token prompts:

| Users | Sampling | Total tok/s | First token p50 / p99 | Time per token p50 / p99 |
| ---: | --- | ---: | --- | --- |
| 1 | greedy | 118.9 | 1,440 ms / 1,449 ms | 5.4 / 6.3 ms |
| 1 | temperature 1.0 | 128.6 | 1,443 ms / 1,446 ms | 5.1 / 5.4 ms |
| 2 | greedy | 160.7 | 1,604 ms / 2,803 ms | 9.1 / 11.9 ms |
| 2 | temperature 1.0 | 176.4 | 1,612 ms / 2,854 ms | 8.1 / 9.0 ms |
| 4 | greedy | 199.0 | 1,761 ms / 5,472 ms | 16.6 / 19.0 ms |
| 4 | temperature 1.0 | 210.3 | 1,518 ms / 5,467 ms | 15.4 / 16.9 ms |
| 8 (4 run, 4 wait) | greedy | 197.4 | 11.9 s / 15.6 s | 16.5 / 23.1 ms |
| 8 (4 run, 4 wait) | temperature 1.0 | 211.7 | 11.2 s / 14.9 s | 15.9 / 18.9 ms |

32,768-token prompts:

| Users | Sampling | Total tok/s | First token p50 / p99 | Time per token p50 / p99 |
| ---: | --- | ---: | --- | --- |
| 1 | greedy | 59.1 | 6,086 ms / 6,134 ms | 5.0 / 5.8 ms |
| 1 | temperature 1.0 | 59.1 | 6,168 ms / 6,178 ms | 4.8 / 5.5 ms |
| 2 | greedy | 64.7 | 7,669 ms / 12.0 s | 12.8 / 24.8 ms |
| 2 | temperature 1.0 | 66.4 | 9,322 ms / 12.2 s | 11.9 / 17.3 ms |
| 4 | greedy | 71.0 | 8,330 ms / 24.0 s | 39.1 / 44.3 ms |
| 4 | temperature 1.0 | 73.0 | 10.8 s / 23.8 s | 30.5 / 44.9 ms |
| 8 (4 run, 4 wait) | greedy | 70.9 | 35.0 s / 52.5 s | 39.6 / 51.9 ms |
| 8 (4 run, 4 wait) | temperature 1.0 | 73.1 | 33.9 s / 52.1 s | 41.5 / 44.6 ms |

`--fast`, output against the default:

- Greedy: the token ids of the 11 check requests are the same as with the defaults, alone and four at once. A prompt
  longer than 512 tokens is read in larger pieces, which changes the order of some sums in attention, so a greedy
  reply can part from the default's after some tokens: in our check of 10 prompts from 2,063 to 32,783 tokens, 256 greedy tokens each, none parted.
- Temperature 1.0: distribution test against serving without a drafter (5-token replies, 3 prompts, 2,500 seeds per
  prompt, the token frequencies at positions 2 to 5, 12 cells): 0 of 12 cells differ, total variation mean 0.045 and
  max 0.079, the size of the test's own noise. The defaults drew the same samples on the same image.

Super fast, output against the default:

- Greedy token ids against the image's own exact path for the same 3-bit weights: equal on all 11 check requests, alone and four
  at once.
- 3-bit experts against the checkpoint's own experts (our quality study: teacher-forced, the 3-bit weights decoded
  from the published files, 62 prompts from short chat, code, tool and image prompts to 512K-token documents; floor:
  the reference run again with only the order of the attention sums changed). The 3-bit experts pick the reference's
  top token 1.0 to 3.2 points less often than the floor does up to 128K tokens, 1.7 points less at 256K and 2.4 at
  512K. In the 2K to 512K documents they give every token of all 68 planted codes the highest probability, as the
  reference does. Over the six groups up to 128K: 95.28 % against 97.18 %, 0.0189 KL per token above the floor.

  | Prompts | Count | Scored tokens | Same top token, 3-bit | Same top token, reference run again | Added KL per token (95 % interval) |
  |---|---:|---:|---:|---:|---|
  | Short chat, code and tool prompts | 21 | 2,572 | 94.87 % | 98.06 % | 0.0235 (0.0189 to 0.0285) |
  | Short prompts with an image | 3 | 385 | 97.40 % | 98.38 % | 0.0123 (0.0074 to 0.0180) |
  | 2K tokens | 12 | 5,003 | 93.34 % | 95.91 % | 0.0271 (0.0227 to 0.0318) |
  | 8K tokens | 7 | 4,043 | 95.45 % | 96.94 % | 0.0106 (0.0082 to 0.0136) |
  | 32K tokens | 6 | 3,623 | 96.69 % | 97.90 % | 0.0171 (0.0119 to 0.0228) |
  | 128K tokens | 5 | 2,085 | 93.91 % | 95.88 % | 0.0228 (0.0171 to 0.0296) |
  | 256K tokens | 5 | 2,085 | 94.24 % | 95.94 % | 0.0137 (0.0098 to 0.0174) |
  | 512K tokens | 3 | 1,251 | 94.56 % | 96.96 % | 0.0152 (0.0104 to 0.0204) |

  Added KL is the KL divergence from the reference minus that of the floor, in nats per token. The intervals cover
  token noise within a row, not the choice of prompts; the 256K and 512K rows rest on five and three documents.
- Task set, 35 tasks: greedy 34 of 35, as with the default. Temperature 1.0, seeds 1 to 3: 35, 34 and 33 of 35
  (102 of 105); the default 32, 32 and 32 (96), `--fast` 32, 33 and 34 (99). Three seeds can move such a total by
  about 5 (the default's arithmetic scored 99 on the v1.0.0 image), so the totals don't rank the versions; the
  next-token numbers above do.
- The relaxed draft check alone, measured with the 4-bit experts: in the distribution test above 10 of 12 cells
  differ (mean 0.230, max 0.430), and 42 % of the sampled tokens were the token that serving without a drafter picks
  most often, against 29 % with the standard check. The 3-bit image was not run through this test.

Memory with BetterBench decode and concurrency (temperature 0.7, top-p 0.95, top-k 20): `--fast` peaked at 29.38 GiB (decode,
160 of 160 requests) and 29.47 GiB (concurrency, 48 of 48 at every level); super fast at 29.33 and 29.42 GiB, with
the same counts.

## Quality

With the default and `--fast` the weights are the publisher's and run in the arithmetic they were trained in, so
the quality checks in this section look for port faults, not for quantization loss. Super fast's checks are in the
section above.

- Greedy replies of the served model match the PyTorch reference's token ids on four fixture prompts for the first
  47, 50, 13 and 78 tokens (the full fixture length for the first and last). The image prompt matches for 97
  tokens.
- Task set, 35 tasks, greedy, defaults, through the launcher: **34 of 35**. Arithmetic 10 of 10, code output 10 of
  10, tool calls 9 of 10, needle in a long text 5 of 5 (two at 31,000 tokens, answered in 10.2 and 12.2 s; three at
  130,000 tokens, 45.2 to 46.1 s). The miss is a translation request that the model answers directly and correctly
  without calling the offered tool.
- Ten facts hidden in texts of 260,049 and 522,196 tokens, greedy, defaults: all ten found each time. At 1,038,311
  tokens on v1.2.0: greedy 8 of 10 in the reasoning without an answer, 9 of 10 at temperature 1.0.
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

- Every run used the same host and a 300 W power cap. A card at a higher cap or a faster host gives other numbers.
- The real-text set is 11 requests. It shows the size of an effect; it is too small for a second decimal.
- The long-prompt and fact-check tables are one run per cell.
- Four requests at once is the most we ran.
- The quality checks of the default are greedy. Sampled requests are checked for running and for plausible replies,
  and through the distribution test.
- Super fast's task set is three seeds at temperature 1.0; a gap of a few tasks is within what sampling can move.
