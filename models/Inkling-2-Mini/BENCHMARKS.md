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

| Arm | 1 request, tok/s | 4 at once, tok/s |
| --- | --- | --- |
| First build | 25.8 / 26.0 | 71.6 / 71.7 |
| Release image, `--no-drafter` | 68.4 / 68.5 | 134.4 / 134.4 |
| Release image (drafter on) | 110.4 / 110.4 | 184.4 / 185.2 |

The drafter row comes from its own mirrored run against `--no-drafter`:

| Arm | 1 request, tok/s | 4 at once, tok/s | Tokens per check step |
| --- | --- | --- | --- |
| `--no-drafter` | 68.4 / 68.5 | 134.6 / 134.1 | 1 |
| Drafter on | 110.4 / 110.4 | 184.4 / 185.2 | 3.14 at one request, 2.97 at four |

The same mirror at temperature 1.0, where the model accepts fewer draft tokens:

| Arm | 1 request, tok/s | 4 at once, tok/s | Tokens per check step |
| --- | --- | --- | --- |
| `--no-drafter` | 68.7 / 68.7 | 141.8 / 141.5 | 1 |
| Drafter on | 95.6 / 95.6 | 168.8 / 168.8 | 2.64 at one request, 2.59 at four |

The `--no-drafter` row at temperature 1.0 comes from the build before the release image, which has the same
libraries and differs only in the draft length.

In the greedy runs the token ids of all 11 requests are the same in the four cases: without the drafter alone and four at once, and
with the drafter alone and four at once. 

The launcher drafts five tokens for one request and four when requests run side by side. Five against seven for one
request, mirrored (7, 5, 5, 7):

| Draft tokens for one request | Greedy, tok/s | Temperature 1.0, tok/s |
| --- | --- | --- |
| 7 | 106.4 / 106.4 | 90.4 / 90.3 |
| 5 (default) | 110.4 / 110.5 | 95.6 / 95.6 |

A longer draft gets more tokens accepted per step (3.31 against 3.14, greedy), but every drafted token is one more
row for the model to check, and the last ones are rarely accepted.

## What each piece added, bench text, no drafter

Each row is one mirrored run against the row above it. The last row was run against the first build, which gave
27.1 / 27.5, 55.7 / 51.9 and 108.0 / 100.3 in that run.

| Step | 1 client, tok/s | 2 clients | 4 clients |
| --- | --- | --- | --- |
| First build | 26.9 / 25.8 | 52.7 / 53.6 | 104.5 / 105.4 |
| One compiled call per forward pass in place of the Python loop | 57.3 / 57.2 | 101.2 / 101.2 | 201.1 / 201.2 |
| New operator and expert builds, 8-bit output head | 61.0 | 107 | 207.6 |
| Attention decode in one call, fused operator launches | 72.6 | 125 | 240.5 |
| Attention that reads the same way at every batch size | 72.4 | 123.4 | 235.4 |
| Release image (prefill kernels, final attention and expert builds) | 72.8 / 72.9 | 124.1 / 124.1 | 236.8 / 237.3 |

The last row is a cost we chose: before it, a request could get different tokens when another request ran next to
it, on 6 of 11 check prompts. After it the count was 1 of 11, and with the release image's attention build it is 0 of 11.

Time to the first token, no drafter, prompt not in the prefix cache:

| Prompt | First build | Release image |
| --- | --- | --- |
| 2,000 tokens | 0.59 s | 0.46 s |
| 6,000 tokens, the first 2,000 in the prefix cache | 1.20 / 1.21 s | 0.93 s |
| 31,000 tokens, answer included | | 10.3 to 11.3 s |
| 130,000 tokens, answer included | | 46.4 to 48.1 s |

The first two rows are from the mirrored bench-text run (both runs of each arm gave the same figure to two digits
where one number is shown). The two long rows are the needle tasks below, with the drafter on.

## Quality

The weights are the publisher's and run in the arithmetic they were trained in, so quality checks here look for
port faults, not for quantization loss.

- Greedy replies of the served model match the PyTorch reference's token ids on four fixture prompts for the first
  47, 50, 13 and 78 tokens (the full fixture length for the first and last). The image prompt matches for 97
  tokens.
- Task set, 35 tasks, drafter on, through the launcher: **34 of 35**. Arithmetic 10 of 10, code output 10 of 10,
  tool calls 9 of 10, needle in a long text 5 of 5 (two at 31,000 tokens, three at 130,000). Without the drafter
  the score is the same 34 of 35. The miss is the same task in both: the model answers a translation request
  directly and correctly without calling the offered tool.
- 8-bit output head (the default): next-token KL against the bf16 head 0.00045 on the check set, and no change in
  the task score.
- `--int8-attention` (off by default): KL about 0.003 against the default, decode about 9.5 % faster without the
  drafter. Task set with the drafter on: the same 34 of 35. The needle prompts take longer with it (14 s against
  10.5 s at 31,000 tokens, 60 s against 46.5 s at 130,000): the prefill kernels of this release read bf16 rows only.

## Limits of these numbers

- One host, one card, one power cap. A card at a higher cap or a faster host will differ.
- The real-text set is 11 requests. It shows the size of an effect; it is too small for a second decimal.
- Four requests at once is the most we ran.
- The quality checks are greedy. Sampled requests are checked for running and for plausible replies only.
