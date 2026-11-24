# syntax=docker/dockerfile:1.19@sha256:b6afd42430b15f2d2a4c5a02b919e98a525b785b1aaff16747d2f623364e39b6
# Inkling 2 Mini on one R9700 (gfx1201): AMD's ROCm 10.2 / vLLM 0.32 image plus the plugin and Paiton's native
# libraries. Build from the repository root with the libraries as a named context:
#   docker build -f models/Inkling-2-Mini/Dockerfile --build-context inkling2_libs=<folder with the .so files> .
FROM rocm/vllm@sha256:73a6637cd69ecac529e6008a668fd03c686007997630039d18fb6d45f161cc42
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md /opt/paiton/inkling2-release-source/
COPY LICENSES /opt/paiton/inkling2-release-source/LICENSES
COPY paiton_vllm_plugin /opt/paiton/inkling2-release-source/paiton_vllm_plugin
RUN python3 -m pip install --no-index --no-build-isolation --no-deps --target /opt/paiton/inkling2-plugin /opt/paiton/inkling2-release-source
COPY models/Inkling-2-Mini/server.py models/Inkling-2-Mini/checkpoint.lock.json /opt/paiton/inkling2/
COPY --from=inkling2_libs . /opt/paiton/inkling2/lib/
# The base image keeps amdsmi on its PYTHONPATH and its own LD_LIBRARY_PATH: prepend, never replace.
# GPU_MAX_HW_QUEUES=1: a fourth working hardware queue slows decode on RDNA4.
ENV PYTHONPATH=/opt/paiton/inkling2-plugin:${PYTHONPATH} GPU_MAX_HW_QUEUES=1 \
    HF_HOME=/models/cache/huggingface HF_HUB_CACHE=/models/cache/huggingface/hub \
    XDG_CACHE_HOME=/models/cache VLLM_CACHE_ROOT=/models/cache/vllm VLLM_DO_NOT_TRACK=1 \
    PAITON_INK2_MOE_SO=/opt/paiton/inkling2/lib/inkling2_moe_gfx1201.so \
    PAITON_INK2_ATTN_SO=/opt/paiton/inkling2/lib/inkling2_attn_gfx1201.so \
    PAITON_INK2_OPS_SO=/opt/paiton/inkling2/lib/inkling2_ops_gfx1201.so \
    PAITON_INK2_STEP_SO=/opt/paiton/inkling2/lib/inkling2_step_gfx1201.so \
    PAITON_INK2_VISION_SO=/opt/paiton/inkling2/lib/inkling2_vision_gfx1201.so
ARG PLUGIN_REVISION=local
LABEL org.opencontainers.image.revision="${PLUGIN_REVISION}" \
      org.opencontainers.image.title="Paiton Inkling 2 Mini R9700" \
      org.opencontainers.image.version="inkling-2-mini-rdna4-v1.0.0" \
      org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.licenses="Apache-2.0" \
      dev.paiton.model.revision="7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6" \
      dev.paiton.gpu.arch="gfx1201"
VOLUME ["/models/cache"]
EXPOSE 8042
HEALTHCHECK --interval=30s --timeout=3s --start-period=30m --retries=3 CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8042/health', timeout=2)"
ENTRYPOINT ["python3", "/opt/paiton/inkling2/server.py"]
