# syntax=docker/dockerfile:1.7
FROM ghcr.io/eliovp/paiton-vllm-plugin@sha256:c56baf54aca1ad229829c1de26e8792806608e65ee9d06ee210b79cd49f70bc9
HEALTHCHECK NONE

LABEL org.opencontainers.image.title="Paiton FLUX.2 klein 4B for Radeon AI PRO R9700" \
      org.opencontainers.image.version="flux2-klein-rdna4-v1.0.1" \
      org.opencontainers.image.licenses="Apache-2.0" \
      dev.paiton.model="Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic" \
      dev.paiton.model.revision="45e9cc76cb70f84473ce5c6c2e2282d0ef3c6ecd" \
      dev.paiton.gpu.arch="gfx1201"

LABEL org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.revision="paiton-flux2-klein-gfx1201-v1.0.1" \
      org.opencontainers.image.version="flux2-klein-rdna4-v1.0.1" \
      org.opencontainers.image.description="Local FLUX.2 klein image generation with Paiton on Radeon" \
      org.opencontainers.image.base.name="ghcr.io/eliovp/paiton-vllm-plugin" \
      org.opencontainers.image.base.digest="sha256:c56baf54aca1ad229829c1de26e8792806608e65ee9d06ee210b79cd49f70bc9" \
      dev.paiton.model="Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic" \
      dev.paiton.model.revision="45e9cc76cb70f84473ce5c6c2e2282d0ef3c6ecd" \
      dev.paiton.artifact.sha256="" \
      dev.paiton.artifact-manifest.sha256="" \
      dev.paiton.config.sha256="" \
      dev.paiton.lm-head-artifact.sha256="" \
      dev.paiton.lm-head-artifact-manifest.sha256="" \
      dev.paiton.release-metadata.sha256=""

ENV PYTHONPATH=/opt/paiton/image:/opt/paiton/image-deps:/opt/paiton/venv-site-packages:/opt/paiton/site-packages \
    HF_HOME=/models/cache HF_HUB_CACHE=/models/cache/hub \
    TORCHINDUCTOR_CACHE_DIR=/models/cache/inductor TRITON_CACHE_DIR=/models/cache/triton \
    XDG_CACHE_HOME=/models/cache/xdg \
    VLLM_USE_PAITON_PLATFORM=0 VLLM_DISABLE_PAITON_PLATFORM=1

COPY requirements.txt /opt/paiton/image/requirements.txt
RUN python3 -m pip install --no-cache-dir --no-deps --target /opt/paiton/image-deps \
      -r /opt/paiton/image/requirements.txt
COPY paiton_image /opt/paiton/image/paiton_image
COPY artifacts /opt/paiton/image/artifacts
COPY service /opt/flux2-service
COPY LICENSE NOTICE README.md THIRD_PARTY_NOTICES.md /opt/paiton/image/
COPY LICENSES /opt/paiton/image/LICENSES
COPY assets/pipeline-memory.* /opt/paiton/image/assets/
RUN python3 -c 'import torch, diffusers, importlib.util; from paiton_image.pipeline import REVISION; assert importlib.util.find_spec("sdnq") is None; print(torch.__version__, diffusers.__version__, REVISION)'
WORKDIR /outputs
EXPOSE 7860
ENTRYPOINT ["python3", "-m", "paiton_image"]
CMD ["--help"]
