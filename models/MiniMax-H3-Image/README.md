# MiniMax-H3-Image on one R9700

DRAFT, not published. Numbers marked "runner" come from the native runner's own measurements (11 Nov 2026); the
container rows are filled in from the end-to-end test below.

[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) is MiniMax's 33B image model: text to image and
image editing, 4 to 16 MP output, 6 sampling steps. This package runs it on one Radeon AI PRO R9700 (32 GB) with
Paiton's native HIP kernels. There is no PyTorch and no Triton in the container. The server takes the same requests as
MiniMax's own examples (`/v1/images/generations` and `/v1/images/edits` on port 30000).

| Request | Time on one R9700 (runner) |
| --- | ---: |
| text to image, 2048 x 2048 | 7.0 s |
| text to image, 4096 x 4096 | 33.5 to 34.5 s |
| text to image, 4736 x 3552 | 33.6 s |
| edit, 4096 x 4096, one reference | 39.7 s |

Times are warm requests: text encoder, 6 steps and VAE decode. PNG encoding adds 0.04 s at 4 MP and 0.16 s at 16 MP.
The card runs at its 300 W limit.

## Requirements

- One AMD Radeon AI PRO R9700 (gfx1201, 32 GB) and a ROCm-capable kernel driver. The server uses up to 30.6 GiB of
  VRAM, so nothing else should run on that card.
- Host RAM: 64 GB or more. The 50 text encoder layers (24.4 GB) are copied once into pinned host memory and streamed
  to the card for every request; 12 of them stay on the card when VRAM allows.
- Disk: about 190 GB under the cache volume (135 GB download, about 50 GB converted).
- Docker, and on the first start a network connection to Hugging Face.

## Start

Build the image from this directory, then start the server:

```bash
docker build -t paiton-minimax-h3-image:rdna4-v0.1.0 models/MiniMax-H3-Image
```

```bash
./models/MiniMax-H3-Image/launch.sh
```

The first start downloads the pinned MiniMax files (135 GB, SHA-256 checked against `checkpoints.lock.json`) and
converts them to the native format inside the container. Both steps run once; later starts check the files and load in
about 20 s. Pass `HF_TOKEN` in the environment if your Hugging Face account needs it. The server prints `READY` when it
takes requests.

Settings for `launch.sh`, all optional:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PAITON_CACHE` | `paiton-minimax-h3-image-cache` | Docker volume or absolute path for downloads and converted weights |
| `PAITON_PORT` | `30000` | Host port |
| `PAITON_BIND` | `127.0.0.1` | Address the port listens on; `0.0.0.0` opens it to the network |
| `PAITON_RENDER` | all cards | One render node, for example `/dev/dri/renderD128`, on a machine with several GPUs |
| `PAITON_DETACH` | `0` | `1` runs the container in the background |

`./models/MiniMax-H3-Image/launch.sh prepare` only downloads and converts, then exits.

## Generate an image

With the server running:

```bash
./models/MiniMax-H3-Image/run.sh generate "A red fox in fresh snow at sunrise" --size 4096x4096 --out fox.png
```

```bash
./models/MiniMax-H3-Image/run.sh edit "Make <Picture 1> a watercolor painting" --image photo.png --size 2048x2048 --out edit.png
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
one `image` file. The answer is `{"created": ..., "data": [{"b64_json": "<PNG>"}], "metrics": {...}}`.

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | required | Used as given: no chat template, no prompt rewriting |
| `size` | `2048x2048` | `WIDTHxHEIGHT`. Images below 4 MP are generated on a 4 MP canvas of the same shape and resized down with Lanczos, as in the diffusers pipeline. Sides round to multiples of 32. The largest canvas is 16.8 MP (4736 x 3552) |
| `seed` | `42` | Draws the same noise as `torch.Generator("cpu").manual_seed(seed)` in the diffusers pipeline |
| `response_format`, `output_format`, `generator_device`, `n` | `b64_json`, `png`, `cpu`, `1` | Only these values are accepted |

`GET /health` and `GET /v1/models` report status. Requests run one at a time; others wait.

## How it runs

| Part | Precision on the card |
| --- | --- |
| Transformer (50 blocks) | FP8 weights and activations, per-channel weight scales; AdaLN modulation precomputed for the fixed 6-step schedule |
| Attention | int8 Q and K, FP8 P and V |
| Text encoder (Qwen3-VL, 50 layers) | FP8 weights, streamed from host memory |
| VAE decoder | int8, 256 px tiles blended as in the reference |

Against the BF16 reference on the same noise, the velocity cosine per step is 0.990 to 0.999 for text to image and
0.994 to 0.999 for the edit case. Images follow their own trajectory after a few steps, so they are not pixel copies of
the BF16 images; the edit case, held by its reference image, is 32.5 dB from the reference output.

## Limits of this build

- An edit takes one reference image. The model accepts up to 16; more need runner support.
- 16.8 MP is the largest canvas that fits next to the weights on 32 GB.
- A request whose reference keeps a short edge of exactly 1024 px with a long edge that is not a multiple of 32 is
  resized for the text encoder with PIL bicubic, which differs slightly from the reference processor.

## License

The Paiton code in this directory is Apache-2.0 (`LICENSE`, `NOTICE`). The model is under the MiniMax-H3-Image
Community License, which excludes the European Union, the United Kingdom, the Republic of Korea and the United States,
asks commercial products to show "MiniMax-H3-Image" prominently, and needs MiniMax's written authorization for hosted
services and for companies above US$1 million revenue. Details in `THIRD_PARTY_NOTICES.md`; full text in
`LICENSES/MiniMax-H3-Image-Community-License.txt`.
