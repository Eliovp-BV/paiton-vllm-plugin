# Paiton Ornith 1.5 on Radeon AI PRO R9700

This package runs the public
[`Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4`](https://huggingface.co/Capicua25x/Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4)
checkpoint on one 32 GB Radeon AI PRO R9700.

## Start

```bash
docker run -d \
  --name paiton-ornith \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc=host \
  -p 8000:8000 \
  --mount type=volume,src=paiton-ornith-cache,dst=/models/cache \
  ghcr.io/eliovp/paiton-vllm-plugin:ornith15-mxfp4-rdna4-v1.0.0
```

The first start downloads the pinned 22.9 GB target checkpoint and its 772 MB
DFlash draft from Hugging Face. Paiton losslessly splits the target checkpoint
into smaller shards once. Allow about 48 GB of free disk space for the download
and ready-to-run copy. The Docker volume preserves both for later starts.

Follow startup:

```bash
docker logs -f paiton-ornith
```

## Chat

After the server reports that it is ready:

```bash
docker exec -it paiton-ornith paiton-chat --model ornith
```

Use `/reset` to clear the conversation and `/quit` to exit. The server also
provides an OpenAI-compatible endpoint at
`http://127.0.0.1:8000/v1/chat/completions`.

## Qualified scope

- Radeon AI PRO R9700 with `gfx1201`
- ROCm 7.14
- one GPU, tensor parallel size 1
- batch size 1 and up to 8,192 tokens
- MXFP4 target model with 16-token DFlash speculation
- public checkpoint revision `9e488f46c0f7969f84c9923ee0256311cd50316e`

The container downloads model weights directly from the public Hugging Face
repository. It contains no checkpoint weights and no Paiton compiler source.
