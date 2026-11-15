# syntax=docker/dockerfile:1.7
# Native MiniMax H3 Ref2VA v1.2 for one Radeon AI PRO R9700 (gfx1201). The image holds the compiled Paiton libraries,
# the runner binary, the host-side Python front end and ffmpeg. No PyTorch, no Triton, no ComfyUI, no compiler source.
# Weights download from Hugging Face on the first run (see ../README.md). Maintainer build only: artifacts/ (from
# stage_artifacts.sh) is never committed, so a plain checkout cannot build this image.
# Build from models/MiniMax-H3: docker build -f native/Dockerfile -t <tag> .
FROM ubuntu:24.04@sha256:1e0a86e57d247923571b75e0aaf48a1449cf8c543d51fb3e07a4a7d7bfa79316

LABEL org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.title="Paiton MiniMax H3 Ref2VA for RDNA4" \
      org.opencontainers.image.description="Native MiniMax H3 Ref2VA video and audio generation for one R9700; weights download on first run" \
      org.opencontainers.image.base.name="ubuntu:24.04" \
      dev.paiton.model="MiniMaxAI/MiniMax-H3" \
      dev.paiton.model.revision="42ed227ee7df40d41602854ae760620d6eb651fe" \
      dev.paiton.gpu.arch="gfx1201"

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY native/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --no-deps --require-hashes --break-system-packages -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

WORKDIR /app
COPY native/artifacts/lib/ ./lib/
COPY native/artifacts/manifest.json native/verify_artifacts.py native/checkpoints.lock.json LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
COPY native/paiton_h3v/ ./paiton_h3v/
COPY LICENSES/MiniMax-H3-Community.txt ./LICENSES/
RUN python3 verify_artifacts.py /app \
    && ! python3 -c 'import torch' 2>/dev/null && ! python3 -c 'import triton' 2>/dev/null

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib \
    PAITON_CACHE_DIR=/cache
WORKDIR /work
ENTRYPOINT ["python3", "-m", "paiton_h3v"]
CMD ["serve"]

LABEL org.opencontainers.image.version="minimax-h3-rdna4-v1.2.0"
