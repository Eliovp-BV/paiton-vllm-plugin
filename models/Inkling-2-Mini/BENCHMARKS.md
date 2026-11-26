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
  1,038,311-token prompt read at 1,174 tok/s against 1,172; the 1,038,311-token row is v1.0.0's own run. v1.2.0 has
  the same model libraries except the expert library, whose prompt kernels are faster and give the same bits: with
  the default settings its greedy tokens are the same as v1.0.0's on the 11 check requests, alone and four at once.
  [TBD default tables: rerun on v1.2.0, or keep v1.0.0's with this note.] The
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
| 4K | 95.2 / 169.6 / 243.0 | 100.0 / 166.8 / 245.5 | 156.3 / 255.2 / 388.7 |
| 16K | 93.1 / 164.1 / 245.4 | 105.8 / 164.0 / 230.1 | 160.5 / 263.2 / 405.8 |
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
  788.7 s, 1,317 tok/s; generation 125.5 tok/s, 1,434 tokens). [TBD --fast at its context.]

[TBD FAST-GRID: `vllm bench serve`, 1,024-, 8,192- and 32,768-token prompts at 4 and 8 users, --fast and super fast,
against the default rows above.]

`--fast`, output against the default:

- Greedy: the token ids of the 11 check requests are the same as with the defaults, alone and four at once. A prompt
  longer than 512 tokens is read in larger pieces, which changes the order of some sums in attention, so a greedy
  reply can part from the default's after some tokens: in our check of 10 prompts from 2,063 to 32,783 tokens, 256 greedy tokens each, none parted.
- Temperature 1.0: distribution test against serving without a drafter (5-token replies, 3 prompts, 2,500 seeds per
  prompt, the token frequencies at positions 2 to 5, 12 cells): 0 of 12 cells differ, total variation mean 0.045 and
  max 0.079, the size of the test's own noise. The defaults drew the same samples on the same image.

Super fast, output against the default:

- Greedy token ids against the image's own exact path for the same 3-bit weights: [TBD w1 gate: equal on the check
  requests at 1 and 4].
- 3-bit experts in our quality study (the published files, teacher-forced on the model's own replies, six groups from
  short prompts, three of them with an image, to 128K-token documents; reference: the default's experts in the
  arithmetic it serves): top-1 agreement 95.81 %, against 97.33 % for two reference runs that differ only in the
  order of some attention sums; worst group 2K, 93.48 % (95.80 %); 128K 95.52 % (96.64 %). KL per token 0.0182 above
  that floor. One 256K and one 512K prompt: top-1 97.12 % (98.08 %) and 99.52 % (99.16 %), KL 0.0038 and 0.0030
  above the floor.
- Task set, 35 tasks: greedy 34 of 35, as with the default. Temperature 1.0, seeds 1 to 3: 35, 34 and 33 of 35
  (102 of 105); the default 32, 32 and 32 (96), `--fast` 32, 33 and 34 (99). Three seeds can move such a total by
  about 5 (the default's arithmetic scored 99 on the v1.0.0 image), so the totals don't rank the versions; the
  next-token numbers above do.
- The relaxed draft check alone, measured with the 4-bit experts: in the distribution test above 10 of 12 cells
  differ (mean 0.230, max 0.430), and 42 % of the sampled tokens were the token that serving without a drafter picks
  most often, against 29 % with the standard check. The 3-bit image was not run through this test.

Memory with BetterBench decode and concurrency (temperature 0.7, top-p 0.95, top-k 20): [TBD gates: peaks with --fast
and super fast, every request ok].

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

- Every run used the same host and a 300 W power cap. A card at a higher cap or a faster host gives other numbers.
- The real-text set is 11 requests. It shows the size of an effect; it is too small for a second decimal.
- The long-prompt and fact-check tables are one run per cell.
- Four requests at once is the most we ran.
- The quality checks of the default are greedy. Sampled requests are checked for running and for plausible replies,
  and through the distribution test.
- Super fast's task set is three seeds at temperature 1.0; a gap of a few tasks is within what sampling can move.
