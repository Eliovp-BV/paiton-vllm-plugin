# Inkling 2 Mini on one R9700

Inkling 2 Mini is Thinking Machines' model for chat, coding, tool calls and image input. This guide serves it with
vLLM and the Paiton plugin on one Radeon AI PRO R9700 (32 GB), with the model's own drafter switched on. The same
image and launcher also serve 16 GB cards; see [16 GB cards](#16-gb-cards-rx-9070-xt-rx-9070-rx-9060-xt).

One request generates at **111 tok/s** with greedy decoding and at **96 tok/s** at the model's default temperature
of 1.0; four requests at once reach 186 and 169 tok/s in total (real text, 25 November 2026, 300 W power cap,
[benchmarks](BENCHMARKS.md)). vLLM has no path for this model on this card, so the comparison in the benchmarks is
with Paiton's own first complete build, which ran at 25 to 26 tok/s for one request and 67 to 72 for four (greedy).

Start it:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
./models/Inkling-2-Mini/serve-docker.sh
```

The first start downloads the pinned weights (20.2 GB). The OpenAI-compatible API is on port 8042, model name
`inkling-2-mini`.

[Run](#run) · [Options](#options) · [Prompt cache in RAM or on an SSD](#prompt-cache-in-ram-or-on-an-ssd-opt-in) ·
[Reasoning effort](#reasoning-effort) · [Images](#images) ·
[16 GB cards](#16-gb-cards-rx-9070-xt-rx-9070-rx-9060-xt) · [What it does not do yet](#what-it-does-not-do-yet) ·
[Weights](#model-weights-and-existing-downloads)

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
- A context of 254,224 tokens per request. vLLM sizes it at start from the memory left on the card (9.9 GiB for
  the cache after the model, the drafter and the vision tower). The longest prompt we ran has 199,028 tokens.
- Tool calls (`tool_choice: "auto"`) and reasoning in a separate `reasoning` field of the reply.
- Prefix caching (vLLM's default): the start of a prompt that the server has seen before is not computed again.
- The drafter: the model's own draft head proposes five tokens (four when requests run side by side) and the model
  checks them in one step. With greedy decoding the replies are the same tokens with and without the drafter: on our
  11 check requests alone and four at once, and on prompts of 4,285 to 199,028 tokens.

## Options

Options go after the script name.

| Option | What it does |
| --- | --- |
| `--no-drafter` | Serves without the drafter: 68 tok/s for one request, greedy and at temperature 1.0, instead of 111 (greedy) or 96 (temperature 1.0) with it. The context grows to 330,736 tokens. |
| `--int8-attention` | Stores the attention projections (input, output and tap projections) as 8-bit rows. Greedy, one request: 78 tok/s instead of 68 without the drafter, 115 instead of 112 with it; four at once 151 instead of 134 without the drafter, 194 instead of 187 with it. The next-token distribution moves a little (KL about 0.003 against the default) and the context is smaller: 235,600 tokens with the drafter, 309,120 without. Prompt speed stays within 1.3 %. Not measured at temperature 1.0 and not run through the task set. Off by default. |
| `--host-cache-gib GIB`, `--extend-cache [auto\|ram\|disk]` | Keep prompts the card's cache drops in system memory or on an SSD; see [Prompt cache in RAM or on an SSD](#prompt-cache-in-ram-or-on-an-ssd-opt-in). Off by default. |
| `--draft-cut P` | Stops a draft at the first token the drafter gives a probability below `P`. Needs vLLM's synchronous scheduling, which the launcher then sets. Not measured on this build. Off by default. |
| `--download-only`, `--offline`, `--verify` | Prepare the weights and exit; start without network access; hash every file again before the start. |
| `--host`, `--api-key` | Bind address and API key inside the container. |

Anything else is passed on to `vllm serve`.

## Prompt cache in RAM or on an SSD (opt-in)

The card keeps the prompts it has read in its prefix cache, as far as the cache holds them (254,224 tokens with the
drafter). Two options keep the prompts the card drops, so a long prompt sent again is loaded instead of read again:

```bash
./models/Inkling-2-Mini/serve-docker.sh --host-cache-gib 24      # 24 GiB of pinned system memory
./models/Inkling-2-Mini/serve-docker.sh --extend-cache disk      # files on the SSD, kept across restarts
```

- `--host-cache-gib GIB` keeps them in GIB of pinned system memory. The launcher refuses a size the host cannot pin
  at start.
- `--extend-cache ram` pins all the memory the host can spare at start. `--extend-cache disk` writes them as files
  under the cache folder (`~/.cache/paiton-inkling-2-mini/kv-disk`), behind a 12 GiB system-memory stage.
  `--extend-cache` alone picks `ram` when the host can pin 15 GiB or more, otherwise `disk`.
- The disk tier uses at most 64 GiB or a quarter of the free space, whichever is smaller. The launcher checks this at
  each start and removes whole folders until the tier fits, other configurations' folders first. The files are
  derived from your prompts, are not encrypted and stay until you delete the folder.
- `serve-docker.sh` gives the container a private 13 GB `/dev/shm` for the disk stage. A `docker run` of your own
  needs `--ipc=private --shm-size=13g`.

First token on one R9700 (greedy, one request at a time; source code as the prompt):

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
- The SSD tier stores about 39 KB per prompt token (5.6 GB for the 146K prompt). Restores ran at about 2.6 GB/s from
  an NVMe SSD; a slower drive gives slower restores.
- Pinning system memory takes about 0.65 s per GiB at start, and while it runs the GPU driver briefly stops the GPU
  queues of every process on the host, including a server on a second card.

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
- Experts, attention, the small operators, the vision tower and the drafter run in Paiton's native HIP libraries for
  the R9700. One library call runs a whole forward pass. vLLM's step bookkeeping (input preparation, block tables,
  slot mapping, drafter bookkeeping) runs in native HIP too. Only vLLM's sampling kernels still use Triton; the image
  carries them compiled, so the first request after a start does not wait for them.
- Memory on the card: vLLM reports 18.24 GiB for the model with the drafter and the vision tower; the rest is cache.
  The input embedding table (0.86 GiB) stays in pinned system memory, which gives about 24K more context than on the
  card and the same tokens.

## 16 GB cards: RX 9070 XT, RX 9070, RX 9060 XT

[16 GB RX 9070 / 9060 XT: numbers after its gate.]

The launcher picks a 16 GB profile by itself when the card has less than 20 GiB free at start. The experts that do not
fit on the card stay in host memory and are copied to the card when a token needs them; none is skipped.
[Exactness against the 32 GB card: after its gate.]

- Context: 32,768 tokens. `--max-model-len 65536` gives 64K, slower because more experts then come from host memory.
- One request at a time.
- Images are off; `--vision` turns them on and leaves less room for experts on the card.
- The drafter is on, as on the 32 GB card; `--no-drafter` turns it off.
- The host needs about 15.5 GiB of pinned memory for the expert copies and the embedding; the launcher refuses to
  start when it cannot get it. [Host RAM requirement: after its gate.]
- `--dry-run 15.9` prints the plan the launcher would use on a card with 15.9 GiB free, without downloading or
  serving.
- [RX 9060 XT (gfx1200): after its gate.]

## What it does not do yet

- More than four requests at once is not tested.
- Equal tokens alone and next to other requests are checked on 11 requests only. A longer or different mix of
  requests is not checked.
- A greedy answer from a cache hit can differ from a cold read of the same long prompt. At 146K tokens, and with an
  image plus 32K tokens, the texts differ from character 240 and 149; at 32K they agree. The hit and the cold read
  split the prompt into prefill chunks at different places, which changes the rounding. The card's own prefix cache
  does this without any tier, and both texts are valid outputs of the model.
- The drafter is the one the publisher ships. At four requests it adds less than at one (see the benchmarks).
- Time to first token on a long prompt is the model's full prefill: 48 s for 143,840 tokens and 73 s for 199,028.
- One card only. Two cards do not make one request faster.

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

- One Radeon AI PRO R9700 (gfx1201, 32 GB), or a 16 GB card as in [16 GB cards](#16-gb-cards-rx-9070-xt-rx-9070-rx-9060-xt).
  Other cards are not supported by this image.
- Linux with a working AMD GPU driver, and Docker.
- The image's vLLM (0.32.1.dev, ROCm 10.2 build). The plugin's Inkling code is built and tested for that vLLM only,
  not for a stock vLLM 0.32.0 install. vLLM versions after 0.32 bring their own Inkling 2 model classes under the same
  names; the plugin then leaves them in place, so its R9700 code is not used. Use the launcher and its image.
- 21 GB of disk for the weights and 67 GB for the image (a 23 GB download; the base layers are the same as those of
  the Qwen3.8 27B image of 24 November, so a host that has that image downloads only the last 26 MB or less).

Licenses and provenance: [NOTICE.md](NOTICE.md). The model is Apache-2.0; the publisher's usage policy is in
[USAGE_POLICY](USAGE_POLICY).
