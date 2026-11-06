# Python 3.12 / ROCm 10.0 / PyTorch runtime layers inherited unchanged from the published Qwen-Image-2.1 image
# (qwen-image21-mxfp4-rdna4-v1.0.3), pinned by digest.
FROM ghcr.io/eliovp/paiton-vllm-plugin@sha256:64f6a6154b93f39b34f272fc66110ce17f72cca3ec31284ab57056ca5cdc2677
HEALTHCHECK NONE

LABEL org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.title="Paiton Idol (image to 3D Gaussians) for Radeon AI PRO R9700" \
      org.opencontainers.image.version="idol-rdna4-v1.0.0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="Idol with the native Paiton denoiser; stock weights download from Hugging Face on first run" \
      org.opencontainers.image.base.name="ghcr.io/eliovp/paiton-vllm-plugin" \
      org.opencontainers.image.base.digest="sha256:64f6a6154b93f39b34f272fc66110ce17f72cca3ec31284ab57056ca5cdc2677" \
      dev.paiton.model="speridlabs/idol" \
      dev.paiton.model.revision="2731c2291fc65a797c5cb627a5e5526fc1490902" \
      dev.paiton.gpu.arch="gfx1201"

ENV PYTHONPATH=/opt/paiton/idol:/opt/paiton/idol-deps \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PAITON_IDOL_HOME=/models/idol HF_HOME=/models/cache HF_HUB_CACHE=/models/cache/hub \
    XDG_CACHE_HOME=/models/cache/xdg MIOPEN_USER_DB_PATH=/models/cache/miopen MIOPEN_CUSTOM_CACHE_DIR=/models/cache/miopen

COPY requirements.txt /opt/paiton/idol/requirements.txt
RUN python3 -m pip install --no-cache-dir --no-deps --target /opt/paiton/idol-deps -r /opt/paiton/idol/requirements.txt
COPY paiton_idol /opt/paiton/idol/paiton_idol
COPY artifacts /opt/paiton/idol/artifacts
COPY LICENSE NOTICE README.md THIRD_PARTY_NOTICES.md /opt/paiton/idol/
RUN cd /opt/paiton/idol/artifacts && sha256sum -c SHA256SUMS \
    && python3 -c 'import idol, tensordict, plyfile, timm; from idol.pipeline import IdolPipeline; print("ok")'
WORKDIR /outputs
ENTRYPOINT ["python3", "-m", "paiton_idol"]
CMD ["--help"]
