# syntax=docker/dockerfile:1.7
# Native LTX-2.5 (two-stage distilled, video + audio) for one Radeon AI PRO R9700 (gfx1201). The image holds the
# compiled Paiton libraries, the runner binary, a small Python server and ffmpeg. No PyTorch, no Triton, no compiler
# source. The converted weights download from Hugging Face on the first start (see README.md). Maintainer build only:
# artifacts/ (from stage_artifacts.sh) is never committed, so a plain checkout cannot build this image.
FROM ubuntu:24.04@sha256:1e0a86e57d247923571b75e0aaf48a1449cf8c543d51fb3e07a4a7d7bfa79316

LABEL org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.title="Paiton LTX-2.5 for RDNA4" \
      org.opencontainers.image.description="Native LTX-2.5 text-to-video and image-to-video server with audio for one R9700; weights download on first start" \
      org.opencontainers.image.base.name="ubuntu:24.04" \
      dev.paiton.model="Lightricks/LTX-2.5" \
      dev.paiton.gpu.arch="gfx1201"

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --no-deps --require-hashes --break-system-packages -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

WORKDIR /app
COPY artifacts/lib/ ./lib/
COPY artifacts/manifest.json verify_artifacts.py checkpoints.lock.json client.py LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
COPY paiton_ltx25/ ./paiton_ltx25/
COPY LICENSES/ ./LICENSES/
RUN python3 verify_artifacts.py /app \
    && ! python3 -c 'import torch' 2>/dev/null && ! python3 -c 'import triton' 2>/dev/null

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib:/usr/local/lib/python3.12/dist-packages/_rocm_sdk_libraries/lib \
    PAITON_CACHE_DIR=/cache
EXPOSE 30000
HEALTHCHECK --interval=30s --timeout=5s --start-period=7200s \
  CMD python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=3).read()'
ENTRYPOINT ["python3", "-m", "paiton_ltx25"]
CMD ["serve"]

LABEL org.opencontainers.image.version="ltx-2.5-rdna4-v0.1.0"
