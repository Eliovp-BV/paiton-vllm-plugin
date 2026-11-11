# MiniMax-H3-Image on a single R9700

[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) is MiniMax's 33B image model: text to image and
image editing, 4 to 16 MP output, 6 sampling steps. This package runs it on a single Radeon AI PRO R9700 (32 GB) with
Paiton's native HIP kernels. There is no PyTorch and no Triton in the container. The server takes the same requests as
MiniMax's own examples (`/v1/images/generations` and `/v1/images/edits` on port 30000).

| Request | Time |
| --- | ---: |
| text to image, 2048 x 2048 | 6.9 s |
| text to image, 1024 x 1024 (made at 2048 x 2048, then resized) | 7.1 s |
| text to image, 4096 x 4096 | 33.4 s |
| edit, 4096 x 4096, one 1024 x 1024 reference | 39.9 s |
| edit, 4736 x 3552, one 4736 x 3552 reference | 50.1 s |
| edit, 4096 x 4096, two references | 66.9 s |
| edit, 4096 x 4096, three references | 88.5 s |

Server-side time per request from HTTP request to PNG, warm, measured through this container on 11 Nov 2026 (three runs
each for the 2048 and 4096 text-to-image rows, one run for the others). It covers the text encoder, 6 steps, VAE
decode, PNG encoding and the resize. A client on the same machine sees 0.1 to 0.7 s more for the base64 transfer. The
card runs at its 300 W limit and uses up to 27.6 GiB.

## Requirements

- One AMD Radeon AI PRO R9700 (gfx1201, 32 GB) and a ROCm-capable kernel driver. The server uses up to 30.6 GiB of
  VRAM, so nothing else should run on that card.
- Host RAM: 64 GB or more. The 50 text encoder layers (24.4 GB) are copied once into pinned host memory and streamed
  to the card for every request; 12 of them stay on the card when VRAM allows.
- Disk: 48 GB under the cache volume.
- Docker, and on the first start a network connection to Hugging Face.

## Start

From a checkout of this repository:

```bash
./models/MiniMax-H3-Image/launch.sh
```

`launch.sh` runs the published image `ghcr.io/eliovp/paiton-vllm-plugin:minimax-h3-image-rdna4-v0.1.0`. The compiled
Paiton runtime (the native libraries and the runner) is only in that image, not in this repository, so the image cannot
be rebuilt from a checkout.

The first start downloads the FP8 weights from
[EliovpAI/MiniMax-H3-Image-FP8-Paiton-RDNA4](https://huggingface.co/EliovpAI/MiniMax-H3-Image-FP8-Paiton-RDNA4)
(48 GB, every file SHA-256 checked against `checkpoints.lock.json`). The first SHA-256 check of the 48 GB took 30 s on
our test host; later starts skip it and are ready in about 20 s. The server takes requests once it prints `READY http://0.0.0.0:30000`.

Settings for `launch.sh`, all optional:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PAITON_CACHE` | `paiton-minimax-h3-image-cache` | Docker volume or absolute path for the downloaded weights |
| `PAITON_PORT` | `30000` | Host port |
| `PAITON_BIND` | `127.0.0.1` | Address the port listens on; `0.0.0.0` opens it to the network |
| `PAITON_RENDER` | all cards | One render node, for example `/dev/dri/renderD128`, on a machine with several GPUs |
| `PAITON_DETACH` | `0` | `1` runs the container in the background |

`./models/MiniMax-H3-Image/launch.sh prepare` only downloads and checks the weights, then exits.

## Generate an image

With the server running:

```bash
./models/MiniMax-H3-Image/run.sh generate "A red fox in fresh snow at sunrise" --size 4096x4096 --out fox.png
```

```bash
./models/MiniMax-H3-Image/run.sh edit "Make <Picture 1> a watercolor painting" --image photo.png --size 2048x2048 --out edit.png
```

```bash
./models/MiniMax-H3-Image/run.sh edit "The cat from <Picture 1> sits on the sofa from <Picture 2>" --image cat.png --image sofa.png --size 4096x4096 --out cat-sofa.png
```

```bash
./models/MiniMax-H3-Image/run.sh benchmark --size 4096x4096 --runs 3
```

```bash
./models/MiniMax-H3-Image/run.sh stop
```

MiniMax's example clients `t2i.py` and `edit.py` (in the model repository under `examples/`) work against the same
port without changes.

## API

`POST /v1/images/generations` takes JSON; `POST /v1/images/edits` takes `multipart/form-data` with the same fields plus
one `image` file per reference, in the order `<Picture 1>`, `<Picture 2>`, ... of the prompt. The answer is `{"created": ..., "data": [{"b64_json": "<PNG>"}], "metrics": {...}}`.

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | required | Used as given: no chat template, no prompt rewriting |
| `size` | `2048x2048` | `WIDTHxHEIGHT`. Images below 4 MP are generated on a 4 MP canvas of the same shape and resized down with Lanczos, as in the diffusers pipeline. Sides round to multiples of 32. The largest canvas is 16.8 MP (4736 x 3552) |
| `seed` | `42` | Draws the same noise as `torch.Generator("cpu").manual_seed(seed)` in the diffusers pipeline, to within 1 ulp per value |
| `response_format`, `output_format`, `generator_device`, `n` | `b64_json`, `png`, `cpu`, `1` | Only these values are accepted |

`GET /health` and `GET /v1/models` report status. Requests run one at a time; others wait.

## How it runs

| Part | Precision on the card |
| --- | --- |
| Transformer (50 blocks) | FP8 weights (GPTQ rounding) and activations, per-channel weight scales; AdaLN modulation precomputed for the fixed 6-step schedule |
| Attention | int8 Q and K, FP8 P and V |
| Text encoder (Qwen3-VL, 50 layers) | FP8 weights, streamed from host memory |
| VAE decoder | int8, 256 px tiles blended as in the reference |

A 30-case quality gate (24 text-to-image prompts from 4 to 17 MP, 6 edits with 1 to 3 references) compared these
weights with the BF16 model in diffusers on the same token ids and noise. The CLIP-L text-image score is 32.18 against
32.31 for BF16, within the case-to-case spread. Mean image similarity to the BF16 image is 0.979 (CLIP-L) and 0.945
(DINOv2). The images differ from the BF16 ones mainly in layout, because a 6-step sampler follows a different path
after small numeric changes; text and faces render correctly. Edits stay closest to BF16.

## Limits of this build

- Each reference adds about 1,000 text-encoder tokens and its own image rows. Prompts are limited to 4,096 tokens in
  total, so an edit takes about three 1024 px references (the model accepts up to 16). Larger requests get an HTTP 400
  error that names the limit.
- 16.8 MP is the largest canvas that fits next to the weights on 32 GB.
- A request whose reference keeps a short edge of exactly 1024 px with a long edge that is not a multiple of 32 is
  resized for the text encoder with PIL bicubic, which differs slightly from the reference processor.

## License

The Paiton code in this directory is Apache-2.0 (`LICENSE`, `NOTICE`). The model is under the MiniMax-H3-Image
Community License, which excludes the European Union, the United Kingdom, the Republic of Korea and the United States,
asks commercial products to show "MiniMax-H3-Image" prominently, and needs MiniMax's written authorization for hosted
services and for companies above US$1 million revenue. Details in `THIRD_PARTY_NOTICES.md`; full text in
`LICENSES/MiniMax-H3-Image-Community-License.txt`.
