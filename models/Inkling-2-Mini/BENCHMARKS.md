# Inkling 2 Mini benchmarks

Measured on 24 and 25 November 2026 on one Radeon AI PRO R9700 (gfx1201, 32 GB) at a 300 W power cap, with the
image `inkling-2-mini-rdna4-v1.0.0` (ROCm 10.2, vLLM 0.32) and the checkpoint `thinkingmachines/Inkling-2-Mini` at
revision `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`.

vLLM has no path for this model on the R9700, so there is no stock number to compare with. Every comparison here is
with Paiton's own first complete build of 24 November ("first build" below): the same weights and arithmetic on
Paiton's first compiled expert, attention and operator libraries, with the layers called one by one from Python and
without the drafter.

## How the numbers were taken

- Every arm is a fresh container and a fresh server. Comparisons run in mirrored order (old, new, new, old) and the
  tables show both runs of each arm.
- Requests are greedy (temperature 0) unless a table says temperature 1.0, which is the model's default. Sampled
  runs use a fixed seed, and their replies differ from the greedy ones in length.
- "Real text" is 11 requests: chat, code and tool-call prompts with 64 to 160 new tokens each, and one 2,400-token
  prompt with 64 new tokens. The first request or group warms the server up and is not counted. Tokens per second
  is the number of generated tokens divided by the wall time of the requests, prompt processing included.
- "Bench text" is a 64-token prompt with 192 new tokens, sent by 1, 2 and 4 clients at once. The numbers are
  generated tokens of all clients per second. The text repeats itself, which flatters a drafter, so the drafter
  results are given on real text only.
- No other job ran on the card during an arm. The second card of the host was in use by other work.

## Release image against the first build, real text

All rows greedy:

| Arm | 1 request, tok/s | 4 at once, tok/s |
| --- | --- | --- |
| First build | 25.9 / 26.3 | 72.0 / 70.9 |
| Release image, `--no-drafter` | 68.4 / 68.4 | 134.5 / 134.4 |
| Release image (drafter on) | 111.5 / 111.6 | 186.7 / 186.6 |

The drafter row comes from its own mirrored run against `--no-drafter`, greedy:

| Arm | 1 request, tok/s | 4 at once, tok/s | Tokens per check step |
| --- | --- | --- | --- |
| `--no-drafter` | 68.5 / 68.4 | 134.3 / 134.4 | 1 |
| Drafter on | 111.5 / 111.6 | 186.7 / 186.6 | 3.14 at one request, 2.97 at four |

The same mirror at temperature 1.0, where the model accepts fewer draft tokens:

| Arm | 1 request, tok/s | 4 at once, tok/s | Tokens per check step |
| --- | --- | --- | --- |
| `--no-drafter` | 68.7 / 68.7 | 141.5 / 141.5 | 1 |
| Drafter on | 96.7 / 96.7 | 169.4 / 170.0 | 2.64 at one request, 2.59 at four |

In the greedy runs the token ids of all 11 requests are the same in the four cases: without the drafter alone and
four at once, and with the drafter alone and four at once. The longest of these prompts has 2,400 tokens; longer
prompts are in [the next section](#long-prompts-with-the-drafter-known-issue).

## Long prompts with the drafter (known issue)

Same prompt with and without the drafter, greedy, 320 new tokens, one request, one run per length:

| Prompt tokens | Reply with the drafter |
| ---: | --- |
| 43 | same text as without |
| 4,285 | differs |
| 17,225 | differs |
| 71,411 | differs |
| 143,840 | differs |
| 199,028 | differs |

The cause is in the attention of the window layers. A check step that crosses a 64-key step of the 1,024-key window
adds up the partial sums of some rows in another order than a one-token step does. The last bits differ, and a greedy
reply can then take another path. A request is only affected once its prompt and reply together pass 1,024 tokens
(the 2,400-token check prompt above still gave the same tokens). The task scores below,
with the drafter on, include prompts of 31,000 and 130,000 tokens. A fixed attention build gave the same text as
without the drafter at all six lengths in the same check (greedy, one run); it is not in the release image yet.

The launcher drafts five tokens for one request and four when requests run side by side. Five against seven for one
request, mirrored (7, 5, 5, 7), measured one build before the release image (its drafter library was about 1 %
slower, which is why the 5-token row is below the tables above):

| Draft tokens for one request | Greedy, tok/s | Temperature 1.0, tok/s |
| --- | --- | --- |
| 7 | 106.4 / 106.4 | 90.4 / 90.3 |
| 5 (default) | 110.4 / 110.5 | 95.6 / 95.6 |

A longer draft gets more tokens accepted per step (3.31 against 3.14, greedy), but every drafted token is one more
row for the model to check, and the last ones are rarely accepted.

## What each piece added, bench text, no drafter

All rows greedy. Each row is one mirrored run against the row above it. The last row was run against the first build, which gave
27.1 / 27.5, 55.7 / 51.9 and 108.0 / 100.3 in that run.

| Step | 1 client, tok/s | 2 clients | 4 clients |
| --- | --- | --- | --- |
| First build | 26.9 / 25.8 | 52.7 / 53.6 | 104.5 / 105.4 |
| One compiled call per forward pass in place of the Python loop | 57.3 / 57.2 | 101.2 / 101.2 | 201.1 / 201.2 |
| New operator and expert builds, 8-bit output head | 61.0 | 107 | 207.6 |
| Attention decode in one call, fused operator launches | 72.6 | 125 | 240.5 |
| Attention that reads the same way at every batch size | 72.4 | 123.4 | 235.4 |
| Release image (prefill kernels, final attention and expert builds) | 72.8 / 72.9 | 124.1 / 124.1 | 236.8 / 237.3 |

The last row is a cost we chose: before it, a request could get different greedy tokens when another request ran
next to it, on 6 of 11 check prompts. After it the count was 1 of 11, and with the release image's attention build it
is 0 of 11.

Time to the first token, greedy, no drafter, prompt not in the prefix cache:

| Prompt | First build | Release image |
| --- | --- | --- |
| 2,000 tokens | 0.59 s | 0.46 s |
| 6,000 tokens, the first 2,000 in the prefix cache | 1.20 / 1.21 s | 0.93 s |
| 31,000 tokens, answer included | | 10.3 to 11.2 s |
| 130,000 tokens, answer included | | 46.3 to 48.0 s |

The first two rows are from the mirrored bench-text run (both runs of each arm gave the same figure to two digits
where one number is shown). The two long rows are the needle tasks below, greedy with the drafter on.

## Quality

The weights are the publisher's and run in the arithmetic they were trained in, so quality checks here look for
port faults, not for quantization loss.

- Greedy replies of the served model match the PyTorch reference's token ids on four fixture prompts for the first
  47, 50, 13 and 78 tokens (the full fixture length for the first and last). The image prompt matches for 97
  tokens.
- Task set, 35 tasks, greedy, drafter on, through the launcher: **34 of 35**. Arithmetic 10 of 10, code output 10 of 10,
  tool calls 9 of 10, needle in a long text 5 of 5 (two at 31,000 tokens, three at 130,000). Without the drafter
  the score is the same 34 of 35. The miss is the same task in both: the model answers a translation request
  directly and correctly without calling the offered tool.
- 8-bit output head (the default): next-token KL against the bf16 head 0.00045 on the check set, and no change in
  the task score.
- `--int8-attention` (off by default): KL about 0.003 against the default, greedy decode about 9.5 % faster without
  the drafter (one request, measured on an earlier build). Task set with the drafter on, greedy: the same 34 of 35.
  The needle prompts take longer with it (14 s against 10.5 s at 31,000 tokens, 60 s against 46.5 s at 130,000): the
  prefill kernels of this release read bf16 rows only.

## Limits of these numbers

- One host, one card, one power cap. A card at a higher cap or a faster host will differ.
- The real-text set is 11 requests. It shows the size of an effect; it is too small for a second decimal.
- Four requests at once is the most we ran.
- The quality checks are greedy. Sampled requests are checked for running and for plausible replies only.
