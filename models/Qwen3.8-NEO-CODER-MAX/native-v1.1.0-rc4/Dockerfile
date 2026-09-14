# Build only from an allowlisted, externally staged runtime context.
FROM ghcr.io/eliovp/paiton-vllm-plugin@sha256:c56baf54aca1ad229829c1de26e8792806608e65ee9d06ee210b79cd49f70bc9 AS qualified
# Drop upstream test-only credentials and the unrelated Qronos payload.
# Copy the resulting filesystem into a fresh image so removed bytes are absent
# from every published layer, rather than hidden beneath a whiteout.
RUN rm -f /opt/paiton/site-packages/transformers/testing_utils.py \
          /opt/paiton/site-packages/transformers/__pycache__/testing_utils.*.pyc \
          /usr/local/bin/paiton-chat && \
    rm -rf /opt/paiton/qwen38-overlay

FROM scratch AS runtime
COPY --from=qualified / /
ENV PATH=/opt/rocm/core-7.14/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    ROCM_PATH=/opt/rocm/core-7.14 HIP_PATH=/opt/rocm/core-7.14 \
    LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:/opt/rocm/core-7.14/lib64 \
    PYTHONPATH=/opt/paiton/neo-plugin:/opt/paiton/release-plugin:/opt/paiton/venv-site-packages:/opt/paiton/site-packages:/opt/paiton/vllm \
    VLLM_USE_PAITON_PLATFORM=1 VLLM_DISABLE_PAITON_PLATFORM=0 \
    VLLM_PLUGINS=paiton_platform,register_paiton_models \
    PAITON_GPU_ARCH=gfx1201 PYTORCH_ROCM_ARCH=gfx1201 \
    AITER_ROCM_ARCH=gfx1201 ROCM_SDK_TARGET_FAMILY=gfx1201 \
    HF_HOME=/models/cache HF_HUB_CACHE=/models/cache/hub
WORKDIR /opt/paiton

FROM runtime AS release
# Scope queue sharing and larger prefill to this candidate profile.
ENV GPU_MAX_HW_QUEUES=1 PAITON_GGUF_PREFILL_TOKENS=2048
ARG PLUGIN_REVISION
ARG COMPILER_REVISION
LABEL org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.title="Paiton Qwen3.8 NEO CODER MAX Q4_K_M RDNA4" \
      org.opencontainers.image.version="q4km-native-v1.1.0-rc4" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      org.opencontainers.image.revision="${PLUGIN_REVISION}" \
      dev.paiton.compiler.revision="${COMPILER_REVISION}" \
      dev.paiton.model.revision="89230607b3708bc1efe174e8fd34d4b164476c88" \
      dev.paiton.runtime.base.digest="sha256:c56baf54aca1ad229829c1de26e8792806608e65ee9d06ee210b79cd49f70bc9"
COPY plugin/ /opt/paiton/neo-plugin/paiton_vllm_plugin/
COPY model/ /opt/paiton/neo-model/
EXPOSE 8000
VOLUME ["/models/cache"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=30m \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
ENTRYPOINT ["python3", "-m", "paiton_vllm_plugin.gguf_release_server"]
CMD []
