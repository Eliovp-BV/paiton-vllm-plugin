# Kimi-K3.1-Pro community build for AMD Instinct CDNA4 (gfx950): plain PyTorch, about 600 lines of Python.
# Nothing proprietary: no Paiton engine, no Paiton kernels, no weights. Mount the checkpoint at /model.
# Build from this folder: docker build -t paiton-kimi31-community .
FROM rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.13.0

ARG PLUGIN_REVISION=unknown
LABEL org.opencontainers.image.title="Kimi-K3.1-Pro community build (plain PyTorch, ROCm 10.1, CDNA4)" \
      org.opencontainers.image.description="Unoptimized reference server for moonshotai/Kimi-K3.1-Pro on 4+ AMD Instinct GPUs" \
      org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.revision="${PLUGIN_REVISION}" \
      org.opencontainers.image.version="kimi31pro-rocm10.1.0-pytorch2.13.0-cdna4-v1" \
      org.opencontainers.image.licenses="Apache-2.0" \
      dev.paiton.gpu.arch="gfx950" \
      dev.paiton.model="moonshotai/Kimi-K3.1-Pro"

RUN pip install --no-cache-dir "transformers==5.20.2" "tiktoken==0.14.0" "safetensors==0.8.0"

WORKDIR /app
COPY kimi31_community /app/kimi31_community
COPY tests /app/tests
COPY MODEL_LICENSE NOTICE.md /app/

ENV PYTHONUNBUFFERED=1 MODEL_DIR=/model PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXPOSE 8000
ENTRYPOINT ["python3", "-m", "kimi31_community.server"]
