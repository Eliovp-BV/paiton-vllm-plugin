# syntax=docker/dockerfile:1.7
FROM ghcr.io/eliovp/paiton-vllm-plugin@sha256:c56baf54aca1ad229829c1de26e8792806608e65ee9d06ee210b79cd49f70bc9

ARG PLUGIN_REVISION=local
LABEL org.opencontainers.image.title="Paiton Qwen3-Coder 30B R9700 local candidate" \
      org.opencontainers.image.revision="${PLUGIN_REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      dev.paiton.model.revision="4bd30395b72ea6045edd04806c4fea448d4467b3" \
      dev.paiton.artifact.sha256="74fb7015e285d3b7863fc8f54341204a252a2ab305921edd5b30f5724915958f" \
      dev.paiton.gpu.arch="gfx1201"

# The separate local build context contains only the compiled artifact,
# manifest and AMD SMI runtime package with its license. No compiler source.
COPY --from=qwen3_coder_overlay . /opt/paiton/qwen3-coder-overlay/
COPY paiton_vllm_plugin /opt/paiton/release-plugin/paiton_vllm_plugin/
COPY models/Qwen3-Coder-30B /opt/paiton/qwen3-coder/
COPY LICENSE LICENSES THIRD_PARTY_NOTICES.md /opt/paiton/qwen3-coder/notices/
RUN cp -a /opt/paiton/qwen3-coder-overlay/amdsmi /opt/paiton/venv-site-packages/amdsmi

ENV HF_HOME=/models/cache/huggingface \
    HF_HUB_CACHE=/models/cache/huggingface/hub \
    XDG_CACHE_HOME=/models/cache \
    TRITON_CACHE_DIR=/models/cache/triton \
    VLLM_CACHE_ROOT=/models/cache/vllm \
    VLLM_PLUGINS=register_paiton_models \
    VLLM_ROCM_USE_AITER=0 \
    OMP_NUM_THREADS=1 \
    PAITON_QWEN3_CODER_ARTIFACT=/opt/paiton/qwen3-coder-overlay/qwen3_coder_moe_w4a16_g32_gfx1201.so
VOLUME ["/models/cache"]
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=3s --start-period=30m --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=2)"
ENTRYPOINT ["python3", "/opt/paiton/qwen3-coder/server.py"]
