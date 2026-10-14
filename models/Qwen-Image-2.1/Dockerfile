# The Python 3.12 / ROCm 10.0 / PyTorch runtime layers are inherited unchanged from the published
# qwen-image21-mxfp4-rdna4-v1.0.3 image, pinned by digest. requirements-*.txt document that runtime (the same files are
# in its /app); one of the pinned nightly wheel URLs now serves a file with a different hash, so the runtime is reused
# instead of being reinstalled from those URLs.
FROM ghcr.io/eliovp/paiton-vllm-plugin@sha256:64f6a6154b93f39b34f272fc66110ce17f72cca3ec31284ab57056ca5cdc2677

LABEL org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.title="Paiton Qwen-Image-2.1 MXFP4 and Turbo FP8 for RDNA4" \
      org.opencontainers.image.description="Pinned image runtime; model weights download from Hugging Face on first launch"

WORKDIR /app
COPY paiton_image21/ ./paiton_image21/
COPY artifacts/ ./artifacts/
COPY verify_artifacts.py ./
COPY run.py checkpoint.lock.json checkpoint.uncensored.lock.json checkpoint.turbo.lock.json model.json LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
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

LABEL org.opencontainers.image.version="qwen-image21-mxfp4-rdna4-v1.1.0"
