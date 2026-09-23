FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

LABEL org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.title="Paiton Qwen-Image-2.1 MXFP4 for RDNA4" \
      org.opencontainers.image.description="Pinned image runtime; model weights download from Hugging Face on first launch"

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git libdrm2 libdrm-amdgpu1 libnuma1 libelf1 libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-python.txt requirements-torch.txt requirements.txt ./
RUN python -m pip install --no-cache-dir --no-deps -r requirements-python.txt \
    && python -m pip install --no-cache-dir --no-deps -r requirements-torch.txt \
    && python -m pip install --no-cache-dir --no-deps -r requirements.txt

COPY paiton_image21/ ./paiton_image21/
COPY artifacts/ ./artifacts/
COPY verify_artifacts.py ./
COPY run.py checkpoint.lock.json checkpoint.uncensored.lock.json model.json LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
COPY LICENSES/ ./LICENSES/
RUN python verify_artifacts.py \
    && PYTHONDONTWRITEBYTECODE=1 python -c 'from paiton_image21.native_regions import qualified_upstream; assert qualified_upstream(), "Unqualified framework/source build"'

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    XDG_CACHE_HOME=/cache/runtime \
    MIOPEN_USER_DB_PATH=/cache/miopen-db \
    MIOPEN_CUSTOM_CACHE_DIR=/cache/miopen-kernels \
    PAITON_GPU_LOCK=/cache/paiton-image21-gpu.lock
EXPOSE 8191
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s \
  CMD python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8191/health", timeout=3).read()'
ENTRYPOINT ["python", "/app/run.py"]

LABEL org.opencontainers.image.version="qwen-image21-mxfp4-rdna4-v1.0.3"
