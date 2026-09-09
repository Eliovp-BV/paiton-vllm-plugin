# syntax=docker/dockerfile:1.7
# Publishable artifact/dependency image. ComfyUI is installed from its
# pinned upstream source only in the end user's local image (Dockerfile.local).
FROM ghcr.io/eliovp/paiton-vllm-plugin@sha256:c56baf54aca1ad229829c1de26e8792806608e65ee9d06ee210b79cd49f70bc9
HEALTHCHECK NONE
LABEL org.opencontainers.image.title="Paiton MiniMax H3 RDNA4 artifacts and dependencies" \
      dev.paiton.gpu.arch="gfx1201" \
      org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.version="minimax-h3-rdna4-v1.0.0"
ENV PYTHONPATH=/opt/h3:/opt/h3-deps:/opt/paiton/venv-site-packages:/opt/paiton/site-packages \
    PAITON_H3_DATA=/data PAITON_COMFY_ROOT=/opt/comfyui \
    HF_HOME=/data/cache HF_HUB_CACHE=/data/cache/hub \
    TRITON_CACHE_DIR=/data/cache/triton TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor \
    XDG_CACHE_HOME=/data/cache/xdg \
    VLLM_USE_PAITON_PLATFORM=0 VLLM_DISABLE_PAITON_PLATFORM=1 \
    MIOPEN_FIND_MODE=FAST HF_HUB_DISABLE_TELEMETRY=1
COPY requirements.lock.txt /opt/h3/requirements.lock.txt
COPY paiton_video /opt/h3/paiton_video
COPY artifacts /opt/h3/artifacts
RUN test -s /opt/h3/artifacts/minimax_h3_projections_gfx1201.so
COPY checkpoints.lock.json extra_model_paths.yaml LICENSE NOTICE THIRD_PARTY_NOTICES.md /opt/h3/
COPY LICENSES /opt/h3/LICENSES

RUN python3 -c 'import torch; assert torch.version.hip; print(torch.__version__)'
RUN python3 -c 'import importlib.util; assert importlib.util.find_spec("comfy") is None; assert importlib.util.find_spec("comfyui_frontend_package") is None'
WORKDIR /outputs
ENTRYPOINT ["python3", "-m", "paiton_video"]
CMD ["--help"]
