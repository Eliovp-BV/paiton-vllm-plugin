# Inkling 2 Mini on a single R9700

Inkling 2 Mini is Thinking Machines' model for chat, coding, tool calls and image input. This guide serves it with
vLLM and the Paiton plugin on a Radeon AI PRO R9700 (32 GB).

One request can hold **1,048,576 tokens**, the model's full context length. With 1,024-token prompts
and 512-token replies (`vllm bench serve`), one request generates at **151 tok/s** with greedy decoding and at
**129 tok/s** at the model's default temperature of 1.0 (the model's own sampling settings: no top-p or top-k); four
requests at once reach 285 and 295 tok/s in total. After a 522,196-token prompt one request still generates at
126 tok/s. All numbers are from 26 November 2026 at a 300 W power cap; the
tables are in [Context and speed](#context-and-speed) and the [benchmarks](BENCHMARKS.md). vLLM has no path for this
model on this card, so the comparison in the benchmarks is with Paiton's own first complete build, which ran at 25 to
26 tok/s for one request and 67 to 72 for four on our real-text set (greedy), where this release gives 143 and 241.

Start it:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
./models/Inkling-2-Mini/serve-docker.sh
```

The first start downloads the pinned weights (20.2 GB) and the drafter (1.0 GB). The OpenAI-compatible API is on port
8042, model name `inkling-2-mini`.

[Context and speed](#context-and-speed) · [Run](#run) · [Options](#options) ·
[Prompt cache in RAM or on an SSD](#prompt-cache-in-ram-or-on-an-ssd-opt-in) · [Reasoning effort](#reasoning-effort) ·
[Images](#images) ·
[What it does not do yet](#what-it-does-not-do-yet) · [Weights](#model-weights-and-existing-downloads)

## Context and speed

Measured on an R9700 at a 300 W power cap, with the launcher's defaults unless a row says otherwise: the 4-bit cache and speculative
decoding with the default drafter. "Real text" is our set of 11 chat, code and tool requests (64 to 160 new tokens
each, prompt time included); a real-text cell gives one request / four at once. Real-text speeds are the mean of two
runs in mirrored order (old, new, new, old); the long-prompt cells are one run each.

### Context per cache format

`--kv-cache` sets the format of the cache of the model's eight full-attention passes. Greedy. Each long-prompt cell
gives the first token, the facts found out of ten hidden in the text, and the generation speed after the prompt:

| `--kv-cache` | Tokens per request | Real text, tok/s | 260,049-token prompt | 522,196-token prompt | 1,038,311-token prompt |
| --- | ---: | --- | --- | --- | --- |
| `kv4` (default) | 1,048,576 | 142.7 / 240.9 | 107 s, 10 of 10, 136 tok/s | 295 s, 10 of 10, 126 tok/s | 885 s, 10 of 10, 122 tok/s |
| `fp8` | 558,496 | 140.4 / 237.6 | 109 s, 9 of 10, 158 tok/s | 305 s, 10 of 10, 121 tok/s | does not fit |
| `bf16` | 279,248 | 146.6 / 243.7 | 109 s, 10 of 10, 120 tok/s | does not fit | does not fit |

- Quality against a bf16 reference in our study: `kv4` about 0.005 KL, `fp8` no change (details under [Run](#run)).
- The fp8 miss at 260,049 tokens: the reply used its whole 4,000-token budget on reasoning and never started the
  answer; the reasoning names 9 of the 10 facts.
- On the real-text set bf16 is a little faster than the 4-bit cache (3 % for one request) and fp8 a little slower
  (2 %).
  At 260,049 tokens the three read the prompt at about the same speed: 2,436 tok/s for the 4-bit cache, 2,397 for
  fp8 and 2,380 for bf16.
- The generation speeds in the long-prompt cells depend on each reply (the fp8 cell at 260,049 tokens is 4,000
  tokens of reasoning), so they do not rank the formats.
- Every number on this page comes from a build with the same native libraries as the release image. Run again on
  the release image, the `vllm bench serve` cells came within 0.4 % of them and the 1,038,311-token prompt read at
  1,174 tok/s against 1,172.

### Speed with `vllm bench serve`

vLLM's own benchmark client from the image: 1,024-token prompts, 512 new tokens each, no prefix-cache hits; four
fresh servers in the order `--no-drafter`, default, default, `--no-drafter`. Total output tok/s, the mean of the two
runs of each setting (they agree within 1 %):

| Requests at once | Greedy, default | Greedy, `--no-drafter` | Temperature 1.0, default | Temperature 1.0, `--no-drafter` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 151.3 | 68.8 | 129.4 | 68.9 |
| 2 | 200.9 | 112.1 | 200.5 | 112.2 |
| 4 | 284.9 | 170.7 | 295.2 | 169.5 |

With one request the first token comes after 0.32 s (median), and each further token after 6.0 ms greedy and 7.2 ms
at temperature 1.0.

### Speed by prompt length

Default settings, greedy, one request, the prompt not in the prefix cache, one run per length. Up to 218,915 tokens
the reply is 320 tokens; the three longest rows are the fact checks above:

| Prompt tokens | First token | Prompt speed | Generation | Generation, `--no-drafter` |
| ---: | ---: | ---: | ---: | ---: |
| 4,285 | 1.1 s | 3,824 tok/s | 120.7 tok/s | 69.0 tok/s |
| 17,225 | 4.4 s | 3,921 tok/s | 110.8 tok/s | 68.0 tok/s |
| 71,411 | 20.3 s | 3,524 tok/s | 130.3 tok/s | 64.7 tok/s |
| 143,840 | 46.9 s | 3,068 tok/s | 106.6 tok/s | 61.2 tok/s |
| 218,915 | 80.6 s | 2,718 tok/s | 117.1 tok/s | 57.8 tok/s |
| 260,049 | 106.7 s | 2,436 tok/s | 136.1 tok/s | not run |
| 522,196 | 294.7 s | 1,772 tok/s | 125.6 tok/s | not run |
| 1,038,311 | 884.7 s | 1,174 tok/s | 122.3 tok/s | not run |

Up to 218,915 tokens, first token and prompt speed are from the run without the drafter; with it they are within
2 %, and the reply is the same text with and without the drafter. Each length has its own reply, so the generation
speed does not fall steadily with the length. Four requests of 247,027 prompt tokens each at once (988,108 tokens in
the cache together) all finish; the card's memory peaks at 29.89 of 31.86 GiB.

### Speed by drafter setting and temperature, real text

Real text (11 requests), tok/s for one request / four at once, mirrored:

| Setting | Greedy | Temperature 1.0 |
| --- | --- | --- |
| Default | 142.7 / 240.9 | 137.8 / 231.5 |
| `--drafter dspark` | 109.4 / 181.7 | 98.8 / 171.6 |
| `--no-drafter` | 66.6 / 138.6 | 66.6 / 136.7 |
| `--kv-cache fp8` | 140.4 / 237.6 | not run |
| `--kv-cache bf16` | 146.6 / 243.7 | not run |

With greedy decoding the token ids are the same with each drafter and without one, alone and four at once.

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
- A context of 1,048,576 tokens per request. The cache of the model's eight full-attention passes is stored in a
  4-bit format (`--kv-cache kv4`, about a quarter of the bytes of bf16); the sliding-window passes keep bf16. vLLM
  sizes the cache at start from the memory left on the card (9.7 GiB after the model, the drafter and the vision
  tower, room for 1,074,776 tokens). The longest prompt we ran has 1,038,311 tokens: all ten facts hidden in it
  found, first token after 885 s. See [Context and speed](#context-and-speed).
- What the 4-bit cache costs: in our quality study against a bf16 reference it moves the next-token distribution by
  about 0.005 KL (0.001 from the 4-bit values, the rest from the attention arithmetic that reads them), the same from
  2K to 512K tokens, and every needle is found. The served task set scores 34 of 35 as with bf16. Greedy, one long
  check (ten facts in a 466,944-token text) ran into a loop with the 4-bit cache where fp8 answered. fp8 loops the
  same way on a 262,144-token text, where the bf16 reference picks the same tokens, so we count it as the model's
  behaviour. At the default temperature neither text loops: two of three samples answered each, and the third ran
  out of its 4,000-token budget. `--kv-cache fp8` (558,496 tokens) showed no change in the study.
- Tool calls (`tool_choice: "auto"`) and reasoning in a separate `reasoning` field of the reply.
- Prefix caching (vLLM's default): the start of a prompt that the server has seen before is not computed again.
- Speculative decoding: the default drafter (downloaded at the first start; license in [NOTICE.md](NOTICE.md))
  proposes five tokens (four when requests run side by side) and the model checks them in one step. With greedy decoding the
  replies are the same tokens with and without the drafter: on our 11 check requests alone and four at once, and on
  prompts of 43 to 218,915 tokens. At a temperature above 0 the drafts are sampled too, and a distribution test
  against serving without a drafter passes. `--drafter dspark` uses the draft head in the model's own checkpoint
  instead.

## Options

Options go after the script name.

| Option | What it does |
| --- | --- |
| `--drafter dspark` | Drafts with the draft head in the model's checkpoint instead of the default drafter (no extra download): on the real-text set 109 tok/s greedy for one request instead of 143. Greedy replies are the same tokens. |
| `--no-drafter` | Serves without a drafter: 69 tok/s for one request and 171 for four (greedy, `vllm bench serve`), instead of 151 and 285. 1.5 GiB more for the cache; the context stays at 1,048,576. |
| `--kv-cache fp8`, `--kv-cache bf16` | Cache format of the full-attention passes. fp8 holds 558,496 tokens per request and showed no change in our quality study; bf16 holds 279,248. Speed and long prompts per format: [Context and speed](#context-and-speed). Both have the first-hit miss in [What it does not do yet](#what-it-does-not-do-yet). |
| `--int8-attention` | Stores the attention projections (input, output and tap projections) as 8-bit rows. Greedy, one request: 78 tok/s instead of 68 without the drafter, 115 instead of 112 with it; four at once 151 instead of 134 without the drafter, 194 instead of 187 with it. The next-token distribution moves a little (KL about 0.003 against the default) and the context is smaller: 235,600 tokens with the drafter, 309,120 without. Prompt speed stays within 1.3 %. Measured on the build of 25 November with the bf16 cache and the model's own draft head; not measured with the default drafter, at temperature 1.0 or through the task set. Off by default. |
| `--host-cache-gib GIB`, `--extend-cache [auto\|ram\|disk]` | Keep prompts the card's cache drops in system memory or on an SSD; see [Prompt cache in RAM or on an SSD](#prompt-cache-in-ram-or-on-an-ssd-opt-in). Off by default. |
| `--draft-cut P` | Stops a draft at the first token the drafter gives a probability below `P`. Needs vLLM's synchronous scheduling, which the launcher then sets. Not measured on this build. Off by default. |
| `--download-only`, `--offline`, `--verify` | Prepare the weights and exit; start without network access; hash every file again before the start. |
| `--host`, `--api-key` | Bind address and API key inside the container. |

Anything else is passed on to `vllm serve`.

## Prompt cache in RAM or on an SSD (opt-in)

The card keeps the prompts it has read in its prefix cache, as far as the cache holds them (1,074,776 tokens with
the default settings). Two options keep the prompts the card drops, so a long prompt sent again is loaded instead of read again:

```bash
./models/Inkling-2-Mini/serve-docker.sh --host-cache-gib 24      # 24 GiB of pinned system memory
./models/Inkling-2-Mini/serve-docker.sh --extend-cache disk      # files on the SSD, kept across restarts
```

- `--host-cache-gib GIB` keeps them in GIB of pinned system memory. The launcher refuses a size the host cannot pin
  at start.
- For a host that serves several sessions or agents, with 64 GB of RAM or more and one GPU, we suggest
  `--host-cache-gib 20`. It adds about 13 s to every start, and a hit from RAM takes 0.06 to 0.25 s longer than a hit
  on the card. When you start without a tier and the host has room for one, the launcher prints this suggestion.
- `--extend-cache ram` pins all the memory the host can spare at start. `--extend-cache disk` writes them as files
  under the cache folder (`~/.cache/paiton-inkling-2-mini/kv-disk`), behind a 12 GiB system-memory stage.
  `--extend-cache` alone picks `ram` when the host can pin 15 GiB or more, otherwise `disk`.
- The disk tier uses at most 64 GiB or a quarter of the free space, whichever is smaller. The launcher checks this at
  each start and removes whole folders until the tier fits, other configurations' folders first. The files are
  derived from your prompts, are not encrypted and stay until you delete the folder.
- `serve-docker.sh` gives the container a private 13 GB `/dev/shm` for the disk stage. A `docker run` of your own
  needs `--ipc=private --shm-size=13g`.

First token (greedy, one request at a time; source code as the prompt; measured on the build of 25
November with the bf16 cache and the model's own draft head, [tier check on the release image]):

| Prompt | Cold read | Hit on the card | Hit from RAM | From the SSD after a restart |
| --- | ---: | ---: | ---: | ---: |
| 32K tokens | 8.9 s | 0.16 s | 0.21 s | 0.56 s |
| 146K tokens | 51.2 s | 0.57 s | 0.81 s | 2.3 s |
| Image + 32K tokens | 9.2 s | 0.17 s | 0.23 s | not measured |

- A hit from RAM or from the SSD gives the same text as a hit on the card when both reuse the same tokens. That held
  for every RAM hit (also after 318K tokens of other prompts pushed the first ones off the card) and for the 146K SSD
  restore.
- The SSD tier reuses a prompt in 256-token pieces, so it can recompute up to 255 more tokens than a hit on the card.
  Its text can then differ from the card's hit for the reason in [What it does not do yet](#what-it-does-not-do-yet).
- Decode speed with a tier on stayed within 0.5 % of the speed without one.
- With the bf16 cache the SSD tier stores about 39 KB per prompt token (5.6 GB for the 146K prompt). Restores ran at about 2.6 GB/s from
  an NVMe SSD; a slower drive gives slower restores.
- Pinning system memory takes about 0.65 s per GiB at start, and while it runs the GPU driver briefly stops the GPU
  queues of every process on the host, including a server on a second card.

## Reasoning effort

The model reasons before it answers, and the chat template takes an effort level: `none`, `minimal`, `low`,
`medium`, `high` (the default), `xhigh` or `max`, or a number from 0.0 to 0.99 (the names stand for 0.0, 0.1, 0.2,
0.7, 0.9, 0.99 and 0.99). A lower effort asks the model for shorter reasoning before the answer.

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
- Experts, attention, the small operators, the vision tower and the drafter run in Paiton's native HIP libraries for
  the R9700. One library call runs a whole forward pass. vLLM's step bookkeeping (input preparation, block tables,
  slot mapping, drafter bookkeeping) runs in native HIP too.
- Memory on the card: vLLM reports 18.49 GiB for the model with the default drafter and the vision tower (18.24 with
  `--drafter dspark`, 17.03 without a drafter); the rest is cache.
  The input embedding table (0.86 GiB) stays in pinned system memory, which gives about 24K more context than on the
  card and the same tokens.
- After warm-up the launcher keeps 1 GiB of the card free of PyTorch's allocator. With it, card memory peaked at
  30.71 GiB in BetterBench's decode run and at 30.63 GiB in its concurrency run (sampled requests, prompts of
  changing length), out of 31.86 GiB. To turn it off, start the container with your own `docker run` and
  `-e PAITON_VRAM_HEADROOM_MIB=0`; `serve-docker.sh` does not pass the variable on.

## What it does not do yet

- Cards with less memory than the R9700's 32 GB are not supported yet: when the card has less than 20 GiB free at
  start, the launcher stops before the download.
- More than four requests at once is not tested.
- With a drafter on (the default), vLLM refuses `min_p` and `logit_bias` with HTTP 400 ("not yet supported with
  speculative decoding"). Start with `--no-drafter` for requests that need them.
- A seeded request at a temperature above 0 repeats exactly when it runs alone. When requests run together, the
  number of draft tokens per step depends on how many requests are running at that moment, so the same seed can give
  different text from run to run (6 of 44 seeded replies differed between two runs of four requests at once). Greedy
  output is not affected. Send seeded requests one at a time when you need the same reply again.
- Equal tokens alone and next to other requests are checked on 11 requests only. A longer or different mix of
  requests is not checked.
- A reply after a prefix-cache hit can differ in wording from the reply to the same prompt read cold. We saw it at
  16K and 146K tokens and with an image plus 32K tokens; one 32K prompt gave the same text. The difference starts in
  the first pass through the layers and follows the number of prompt rows a forward pass computes, which is not the
  same for a hit and a cold read. The card's own prefix cache does this without any tier, and both replies are
  valid outputs of the model.
- With `--kv-cache bf16` or `fp8`, the first request that shares a long start with an earlier, different prompt
  reads the shared part again; the request after it hits. The default 4-bit cache hits at once. vLLM keeps the state
  of the sliding-window passes only at the end of a prompt and where two prompts part.
- Time to first token on a long prompt is the model's full prefill: 48 s for 143,840 tokens, 82 s for 218,915 and
  295 s for 522,196.
- The first request after a start that asks for logprobs, a repetition penalty or a JSON schema waits about one
  second while vLLM compiles that option's sampling kernel (on the release image: 1.7 s instead of 0.3 s for a short
  reply with logprobs, 1.4 s with a JSON schema). Later requests with the option do not wait.
- The launcher uses a single card; a second card does not make a request faster.

## Model weights and existing downloads

This release needs `thinkingmachines/Inkling-2-Mini` at revision `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`: the
model file (19.5 GB), the draft head file `mtp.safetensors` (0.6 GB), the tokenizer and the configuration. With the
default drafter it also needs that drafter's checkpoint (1.0 GB; repository, revision and license in
[NOTICE.md](NOTICE.md)). The launcher downloads them on first use, checks every file
against `checkpoint.lock.json` and reuses its cache afterwards.

If the files are already in a Hugging Face cache on the host, mount that cache instead of downloading again:

```bash
PAITON_HF_CACHE="$HOME/.cache/huggingface" ./models/Inkling-2-Mini/serve-docker.sh --offline
```

The cache must hold the exact revision above. [Cache paths and Docker mounts](../../docs/MODEL_WEIGHTS.md).

## Requirements

- A Radeon AI PRO R9700 (gfx1201, 32 GB). Other cards are not supported by this image.
- Linux with a working AMD GPU driver, and Docker.
- The image's vLLM (0.32.1.dev, ROCm 10.2 build). The plugin's Inkling code is built and tested for that vLLM only,
  not for a stock vLLM 0.32.0 install. vLLM versions after 0.32 bring their own Inkling 2 model classes under the same
  names; the plugin then leaves them in place, so its R9700 code is not used. Use the launcher and its image.
- 22 GB of disk for the weights and the drafter, and 67 GB for the image (a 23 GB download; the base layers are the same as those of
  the Qwen3.8 27B image of 24 November, so a host that has that image downloads only the last 0.1 GB or less).

Licenses and provenance: [NOTICE.md](NOTICE.md). The model is Apache-2.0; the publisher's usage policy is in
[USAGE_POLICY](USAGE_POLICY).
