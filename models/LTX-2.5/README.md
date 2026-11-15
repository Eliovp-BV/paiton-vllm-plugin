# LTX-2.5 on a single R9700

[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) is Lightricks' 22B video model. It makes video with a matching
soundtrack from a text prompt, or from a prompt and a first frame. This package runs the distilled two-stage pipeline
(8 steps at half size, a 2x latent upscale, 3 steps at full size) on one Radeon AI PRO R9700 (32 GB) with Paiton's
native HIP kernels. There is no PyTorch and no Triton in the container. The server answers HTTP requests on port 30000
with an mp4 (H.264 video, AAC audio).

| Request (121 frames, 5 s at 24 fps) | Paiton | Stock LTX-2.5, BF16 |
| --- | ---: | ---: |
| text to video, 1280 x 704 | 18.5 s | 128.6 s |
| image to video, 1280 x 704 | 21.3 s | - |
| text to video, 1920 x 1088 | 50.0 s | 420.5 s |
| image to video, 1920 x 1088 | 52.7 s | - |

Paiton: server-side time per request from HTTP request to mp4, warm (median of 3 runs at 1280 x 704, 2 at 1920 x 1088
text to video, 1 for image to video), measured through this container on 15 Nov 2026 (text encoder, both stages,
upscale, VAE decode, audio decode, H.264 and AAC encoding). Stock: the Lightricks `ltx-pipelines` distilled pipeline
in BF16 on the same card, warm, with `--offload cpu` because the BF16 weights do not fit in 32 GB. The card runs at
its 300 W limit; peak VRAM was 26.0 GiB at 1280 x 704 and 27.2 GiB at 1920 x 1088. The server is ready about 25 s
after `launch.sh` once the weights are on disk.

## Requirements

- One AMD Radeon AI PRO R9700 (gfx1201, 32 GB) and a ROCm-capable kernel driver. Nothing else should run on that card.
- Host RAM: the text encoder units that do not fit next to the transformer (about 12.5 GB of its 14.1 GB) are kept in
  pinned host memory and streamed to the card for every request. We tested on a host with 251 GB; 32 GB is the
  smallest size we would try.
- Disk: 38 GB under the cache volume.
- Docker, and on the first start a network connection to Hugging Face.

## Start

From a checkout of this repository:

```bash
./models/LTX-2.5/launch.sh
```

`launch.sh` runs the published image `ghcr.io/eliovp/paiton-vllm-plugin:ltx-2.5-rdna4-v0.1.0`. The compiled Paiton
runtime (the native libraries and the runner) is only in that image, not in this repository, so the image cannot be
rebuilt from a checkout.

The first start downloads the weights from
[EliovpAI/LTX-2.5-FP8-Paiton-RDNA4](https://huggingface.co/EliovpAI/LTX-2.5-FP8-Paiton-RDNA4) and checks every file
against its SHA-256 in `checkpoints.lock.json`. Later starts skip the check. The server takes requests once it prints
`READY http://0.0.0.0:30000`.

Settings for `launch.sh`, all optional:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PAITON_CACHE` | `paiton-ltx-2.5-cache` | Docker volume or absolute path for the downloaded weights |
| `PAITON_PORT` | `30000` | Host port |
| `PAITON_BIND` | `127.0.0.1` | Address the port listens on; `0.0.0.0` opens it to the network |
| `PAITON_RENDER` | all cards | One render node, for example `/dev/dri/renderD128`, on a machine with several GPUs |
| `PAITON_DETACH` | `0` | `1` runs the container in the background |

`./models/LTX-2.5/launch.sh prepare` only downloads and checks the weights, then exits.

## Make a video

With the server running:

```bash
./models/LTX-2.5/run.sh generate "A red fox trots through fresh snow in a pine forest at dawn, birds calling in the distance" --out fox.mp4
```

```bash
./models/LTX-2.5/run.sh generate "The camera pushes in slowly as she turns toward the window, soft piano music" --image first-frame.png --size 1920x1088 --out scene.mp4
```

```bash
./models/LTX-2.5/run.sh benchmark --size 1280x704 --runs 3
```

```bash
./models/LTX-2.5/run.sh stop
```

## API

`POST /v1/videos/generations` takes JSON, or `multipart/form-data` with the same fields and the first frame as an
`image` file. The answer is `{"created": ..., "data": [{"b64_json": "<mp4>"}], "metrics": {...}}`; `metrics` has the
time of each stage.

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | required | Used as given, up to 1,024 tokens. Describe the sound as well as the picture |
| `size` | `1280x704` | `WIDTHxHEIGHT`. Both sides must be multiples of 64 (stage 1 runs at half size on a 32-pixel latent grid), at least 256; width at most 1920 and height at most 1088. Portrait sizes are not supported in this build |
| `num_frames` | `121` | 8k + 1, from 9 to 121, at 24 fps |
| `seed` | `42` | Same seed, same video on this build. The noise does not match a given PyTorch seed |
| `image` | none | First frame for image to video: a PNG, JPEG or WebP file in multipart, or base64 (plain or a `data:` URL) in JSON. It is scaled and center-cropped to `size` |
| `response_format`, `n` | `b64_json`, `1` | Only these values are accepted |

`GET /health` and `GET /v1/models` report status. Requests run one at a time; others wait.

## How it runs

| Part | Precision on the card |
| --- | --- |
| Transformer (48 blocks, video and audio streams) | FP8 weights and activations, one scale per weight row; AdaLN modulation precomputed for the 9 fixed sigma values of the distilled schedule |
| Attention | Video self-attention: int8 Q and K, FP8 P and V. Text cross-attention: int8 Q and K, BF16 P and V. Audio and audio-video attention in BF16 |
| Text encoder (Gemma 4 12B, feature extractor, connectors) | FP8 weights and activations; layers that do not fit stream from host memory |
| Latent upscaler, video VAE decoder and encoder, audio VAE and vocoder | Native kernels, FP8 convolutions in the VAE decoder resnet blocks |

Against the BF16 reference pipeline run with the same text-encoder output and the same noise, the frames of a 1280 x 704 text-to-video
clip have a PSNR of 32.7 dB (SSIM 0.97) and those of a 1920 x 1088 clip 27.4 dB (SSIM 0.92). The stock pipeline with
`--quantization fp8-cast` reaches 25.3 dB at 1280 x 704. The differences are drift in background detail; the
composition follows the BF16 video.

## Limits of this build

- Two sizes are measured: 1280 x 704 and 1920 x 1088, 121 frames each. Other sizes within the limits above run, with
  no measured timings. The runner sizes its buffers once for 1920 x 1088 x 121, so portrait sizes are rejected.
- `num_frames` must be given. The reference pipeline can pick the length from the prompt with its duration head; this
  server does not do that yet.
- The prompt is used as written. The reference pipeline's optional prompt enhancer is not part of this build.
- The seed does not reproduce the noise of the PyTorch pipeline, so the same seed gives a different video there.

## License

The Paiton code in this directory is Apache-2.0 (`LICENSE`, `NOTICE`). The model is under the LTX-2.x Community
License Agreement. Companies with US$10 million or more in annual revenue need a paid license from Lightricks for
anything beyond non-commercial testing and research. Every use is subject to the use restrictions in Attachment A and
Lightricks' Acceptable Use Policy, including stating that generated content is machine generated when you publish it.
Details in `THIRD_PARTY_NOTICES.md`; full text in `LICENSES/LTX-2.x-Community-License.txt`.
