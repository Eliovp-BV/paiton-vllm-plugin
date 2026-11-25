# Inkling 2 Mini benchmarks

Measured on 25 November 2026 on one Radeon AI PRO R9700 (gfx1201, 32 GB) at a 300 W power cap, with the image
`inkling-2-mini-rdna4-v1.0.0` (ROCm 10.2, vLLM 0.32) and the checkpoint `thinkingmachines/Inkling-2-Mini` at
revision `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`.

vLLM has no path for this model on the R9700, so there is no stock number to compare with. Every comparison here is
with Paiton's own first complete build of 24 November ("first build" below): the same weights and arithmetic on
Paiton's first compiled expert, attention and operator libraries, with the layers called one by one from Python and
without the drafter.

## How the numbers were taken

- Every arm is a fresh container and a fresh server. Comparisons run in mirrored order (old, new, new, old) and the
  tables show both runs of each arm.
- Each table says whether its requests are greedy (temperature 0) or at temperature 1.0, the model's default.
  Sampled runs use seed 1, and their replies differ from the greedy ones in length.
- "Real text" is 11 requests: chat, code and tool-call prompts with 64 to 160 new tokens each, and one 2,400-token
  prompt with 64 new tokens. The first request or group warms the server up and is not counted. Tokens per second
  is the number of generated tokens divided by the wall time of the requests, prompt processing included.
- No other job ran on this card during an arm. The second card of the host ran other work at the same time.

## Release image against the first build, real text, greedy

| Arm | 1 request, tok/s | 4 at once, tok/s |
| --- | --- | --- |
| First build | 25.8 / 24.8 | 72.1 / 66.7 |
| Release image, `--no-drafter` | 68.4 / 68.2 | 134.2 / 133.9 |
| Release image (drafter on) | 111.0 / 111.3 | 185.6 / 185.6 |

The drafter row comes from its own mirrored run against `--no-drafter`, greedy:

| Arm | 1 request, tok/s | 4 at once, tok/s | Tokens per check step |
| --- | --- | --- | --- |
| `--no-drafter` | 68.3 / 68.3 | 134.1 / 133.9 | 1 |
| Drafter on | 111.0 / 111.3 | 185.6 / 185.6 | 3.14 at one request, 2.97 at four |

The same mirror at temperature 1.0, where the model accepts fewer draft tokens:

| Arm | 1 request, tok/s | 4 at once, tok/s | Tokens per check step |
| --- | --- | --- | --- |
| `--no-drafter` | 68.7 / 68.5 | 141.9 / 141.2 | 1 |
| Drafter on | 96.5 / 96.3 | 169.7 / 169.3 | 2.64 at one request, 2.59 at four |

In the greedy runs the token ids of all 11 requests are the same in the four cases: without the drafter alone and
four at once, and with the drafter alone and four at once. The launcher drafts five tokens for one request and four
when requests run side by side.

## Long prompts, greedy

One request per length, 320 new tokens, one run each (not mirrored), prompt not in the prefix cache:

| Prompt tokens | First token | Prompt speed | Generation, no drafter | Generation, drafter on |
| ---: | ---: | ---: | ---: | ---: |
| 4,285 | 1.1 s | 3,803 tok/s | 70.1 tok/s | 105.6 tok/s |
| 17,225 | 4.4 s | 3,885 tok/s | 66.8 tok/s | 85.7 tok/s |
| 71,411 | 20.7 s | 3,449 tok/s | 56.4 tok/s | 81.7 tok/s |
| 143,840 | 47.9 s | 3,002 tok/s | 46.6 tok/s | 74.3 tok/s |
| 199,028 | 73.1 s | 2,724 tok/s | 41.1 tok/s | 63.4 tok/s |

First token and prompt speed are from the run without the drafter. At every length, and at a 43-token prompt, the
reply with the drafter is the same text as the reply without it.

## Quality

The weights are the publisher's and run in the arithmetic they were trained in, so quality checks here look for
port faults, not for quantization loss.

- Greedy replies of the served model match the PyTorch reference's token ids on four fixture prompts for the first
  47, 50, 13 and 78 tokens (the full fixture length for the first and last). The image prompt matches for 97
  tokens.
- Task set, 35 tasks, greedy, drafter on, through the launcher: **34 of 35**. Arithmetic 10 of 10, code output 10 of
  10, tool calls 9 of 10, needle in a long text 5 of 5 (two at 31,000 tokens, answered in 10.3 and 10.9 s; three at
  130,000 tokens, 46.6 to 47.6 s). The miss is a translation request that the model answers directly and correctly
  without calling the offered tool.
- 8-bit output head (the default): next-token KL against the bf16 head 0.00045 on the check set.
- `--int8-attention` (off by default): KL about 0.003 against the default. Greedy, mirrored against the default:

  | Arm | 1 request, tok/s | 4 at once, tok/s |
  | --- | --- | --- |
  | Default, drafter on | 111.9 / 111.7 | 187.1 / 186.8 |
  | `--int8-attention`, drafter on | 115.6 / 115.3 | 194.6 / 193.6 |
  | Default, `--no-drafter` | 68.4 / 68.3 | 134.1 / 134.0 |
  | `--int8-attention`, `--no-drafter` | 78.2 / 78.3 | 151.5 / 151.3 |

  The context gets smaller with it: 186,496 tokens with the drafter (206,624 without the option) and 254,432
  without the drafter (277,728). Prompt speed stays within 1.3 % at 4,285 to 143,840 tokens, and the drafter reply is
  the same text as the plain reply at those lengths. It is not run through the task set and not measured at
  temperature 1.0.

## 16 GB cards

[16 GB RX 9070 / 9060 XT: numbers after its gate. Measured on an R9700 held to 16 GiB on a PCIe Gen4 host unless a
16 GB card ran them, and named that way.]

## Limits of these numbers

- One host, one card, one power cap. A card at a higher cap or a faster host will differ.
- The real-text set is 11 requests. It shows the size of an effect; it is too small for a second decimal.
- The long-prompt table is one run per length.
- Four requests at once is the most we ran.
- The quality checks are greedy. Sampled requests are checked for running and for plausible replies only.
