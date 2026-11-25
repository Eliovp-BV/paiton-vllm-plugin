# Inkling 2 Mini on one R9700

Inkling 2 Mini is Thinking Machines' model for chat, coding, tool calls and image input. This guide serves it with
vLLM and the Paiton plugin on one Radeon AI PRO R9700 (32 GB), with the model's own drafter switched on.

One request generates at **111 tok/s** with greedy decoding and at **96 tok/s** at the model's default temperature
of 1.0; four requests at once reach 186 and 169 tok/s in total (real text, 25 November 2026, 300 W power cap,
[benchmarks](BENCHMARKS.md)). vLLM has no path for this model on this card, so
the comparison in the benchmarks is with Paiton's own first complete build, which ran at 26 tok/s for one request and
72 for four (greedy).

Start it:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
./models/Inkling-2-Mini/serve-docker.sh
```

The first start downloads the pinned weights (20.2 GB). The OpenAI-compatible API is on port 8042, model name
`inkling-2-mini`.

[Run](#run) · [Options](#options) · [Reasoning effort](#reasoning-effort) · [Images](#images) ·
[What it does not do yet](#what-it-does-not-do-yet) · [Weights](#model-weights-and-existing-downloads)

## Run

```bash
./models/Inkling-2-Mini/serve-docker.sh --download-only   # download and verify ahead of time
./models/Inkling-2-Mini/serve-docker.sh --offline         # start without network access
./models/Inkling-2-Mini/serve-docker.sh --verify          # hash every file again before the start
```

```bash
curl http://localhost:8042/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Explain binary search in two sentences."}],"max_tokens":512,"stream":true}'
```

The server listens on all interfaces (0.0.0.0). Set `PAITON_BIND=127.0.0.1` to keep it on this machine, and set
`PAITON_API_KEY` (the server then requires that key on `/v1`) or use a firewall if other machines can reach it.
Weights and caches stay in `~/.cache/paiton-inkling-2-mini`; `PAITON_CACHE` chooses another folder and `PAITON_PORT`
another port.

What the default start gives you:

- Up to four requests at once.
- A context of 206,624 tokens per request. vLLM sizes it at start from the memory left on the card (8.3 GiB for
  the cache after the model, the drafter and the vision tower). The longest prompts we checked have 130,000 tokens.
- Tool calls (`tool_choice: "auto"`) and reasoning in a separate `reasoning` field of the reply.
- Prefix caching (vLLM's default): the start of a prompt that the server has seen before is not computed again.
- The drafter: the model's own draft head proposes five tokens (four when requests run side by side) and the model
  checks them in one step. On our 11 check requests (greedy, prompts of up to 2,400 tokens) the replies are the same
  tokens with and without the drafter, alone and four at once. Longer prompts are a
  [known issue](#known-issue-long-prompts-with-the-drafter).

## Options

Options go after the script name.

| Option | What it does |
| --- | --- |
| `--no-drafter` | Serves without the drafter: 68 tok/s for one request, greedy and at temperature 1.0, instead of 111 (greedy) or 96 (temperature 1.0) with it. Frees 0.7 GiB. |
| `--int8-attention` | Stores the attention projections as 8-bit rows. Greedy decode without the drafter is about 9.5 % faster (one request, measured on an earlier build); the next-token distribution moves a little (KL 0.003 against the default build) and long prompts are read slower (60 s against 46.5 s for 130,000 tokens, greedy). Same task score (greedy), 34 of 35. Off by default. |
| `--draft-cut P` | Stops a draft at the first token the drafter gives a probability below `P`. About 6 % faster for one greedy request at `P` = 0.3 in our test, no gain at four. Off by default. |
| `--download-only`, `--offline`, `--verify` | Prepare the weights and exit; start without network access; hash every file again before the start. |
| `--host`, `--api-key` | Bind address and API key inside the container. |

Anything else is passed on to `vllm serve`.

## Reasoning effort

The model reasons before it answers, and the chat template takes an effort level: `none`, `minimal`, `low`,
`medium`, `high` (the default), `xhigh` or `max`.

```bash
curl http://localhost:8042/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is 17 * 23?"}],"max_tokens":600,"chat_template_kwargs":{"reasoning_effort":"none"}}'
```

Reasoning and the answer share `max_tokens`. With the default effort a low limit can end the reply before the answer
starts, so give reasoning requests a few thousand tokens. The model still writes a short reasoning passage at
`none`.

## Images

Send images as OpenAI-style `image_url` content in a chat message, either a URL the server can reach or a
`data:image/png;base64,...` string. The image request we check at every build is a small chart sent as a base64
string, from which the model reads a label and a value. The vision tower loads with the model; there is no switch.

## What it runs

- The weights are the publisher's files, unchanged: the experts in the 4-bit and 8-bit formats the model was trained
  in, everything else in bf16. Nothing is quantized again, with one exception: the output head is stored as 8-bit
  rows when the model loads (no measurable change in our checks, see the benchmarks).
- Experts, attention, the small operators, the vision tower and the drafter run in Paiton's compiled libraries for
  the R9700. One library call runs a whole forward pass.
- Memory on the card: 18.45 GiB for the model with the drafter, the rest for the cache.

## What it does not do yet

- More than four requests at once is not tested.
- Equal tokens alone and next to other requests are checked on 11 requests only. A longer or different mix of
  requests is not checked.
- There is no host-memory cache tier: cached prompts live on the card only.
- The drafter is the one the publisher ships. At four requests it adds less than at one (see the benchmarks).
- Time to first token on a long prompt is the model's full prefill: about 47 s for 130,000 tokens (greedy, answer
  included).
- One card only. Two cards do not make one request faster.

### Known issue: long prompts with the drafter

With the drafter on, a greedy reply to a long prompt can differ from the reply without the drafter. It differed at
every length we checked from 4,285 to 199,028 prompt tokens. It needs more than 1,024 tokens of prompt and reply
together; our 11 check requests (prompts of up to 2,400 tokens) gave the same tokens. The cause is in the attention of
the window layers: in a check step that crosses a 64-key step of the 1,024-key window, some rows add up their partial
sums in another order than a one-token step does, so the last bits differ and a greedy reply can take another path. The task scores in the [benchmarks](BENCHMARKS.md) include prompts of
31,000 and 130,000 tokens with the drafter on. A fixed attention build gives the same text with and without the
drafter at 43 to 199,028 prompt tokens in our check; it is not in this image yet. Start with `--no-drafter` if you
need the exact tokens of plain decoding on long prompts.

## Model weights and existing downloads

This release needs `thinkingmachines/Inkling-2-Mini` at revision `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`: the
model file (19.5 GB), the drafter file `mtp.safetensors` (0.6 GB), the tokenizer and the configuration. The launcher
downloads them on first use, checks every file against `checkpoint.lock.json` and reuses its cache afterwards.

If the files are already in a Hugging Face cache on the host, mount that cache instead of downloading again:

```bash
PAITON_HF_CACHE="$HOME/.cache/huggingface" ./models/Inkling-2-Mini/serve-docker.sh --offline
```

The cache must hold the exact revision above. [Cache paths and Docker mounts](../../docs/MODEL_WEIGHTS.md).

## Requirements

- One Radeon AI PRO R9700 (gfx1201, 32 GB). Other cards are not supported by this image.
- Linux with a working AMD GPU driver, and Docker.
- The image's vLLM (0.32.1.dev, ROCm 10.2 build). The plugin's Inkling code is built and tested for that vLLM only,
  not for a stock vLLM 0.32.0 install. vLLM versions after 0.32 bring their own Inkling 2 model classes under the same
  names; the plugin then leaves them in place, so its R9700 code is not used. Use the launcher and its image.
- 21 GB of disk for the weights and 64 GB for the image (a 22 GB download).

Licenses and provenance: [NOTICE.md](NOTICE.md). The model is Apache-2.0; the publisher's usage policy is in
[USAGE_POLICY](USAGE_POLICY).
