FROM docker.io/vllm/vllm-openai-rocm@sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032
COPY plugin /opt/paiton-native-plugin-source
RUN python3 -m pip install --no-deps --no-build-isolation --no-cache-dir /opt/paiton-native-plugin-source
COPY native /opt/paiton-native
ENV PAITON_EXPERIMENTAL_MXFP4_W4A8=1 \
    PAITON_MXFP4_RUNTIME_MANIFEST=/opt/paiton-native/runtime.json \
    VLLM_DISABLED_KERNELS=AiterMxfp4LinearKernel,EmulationMxfp4LinearKernel \
    VLLM_DISABLE_PAITON_PLATFORM=1
LABEL dev.paiton.scope="Qualified MXFP4 DFlash2 native runtime"

ENV PAITON_EXPERIMENTAL_DFLASH_CONTEXT=1 \
    PAITON_DFLASH_CONTEXT_RUNTIME_MANIFEST=/opt/paiton-native/dflash_context.json

ENV PAITON_EXPERIMENTAL_MXFP4_FORWARD_ERROR_SCOPE=1

ENV PAITON_EXPERIMENTAL_GDN_REPLAY=1 PAITON_GDN_RUNTIME_MANIFEST=/opt/paiton-native/gdn_runtime.json

ENV PAITON_EXPERIMENTAL_ATTENTION=1 PAITON_ATTENTION_RUNTIME_MANIFEST=/opt/paiton-native/attention.json

ENV PAITON_EXPERIMENTAL_DFLASH_HEAD=1 PAITON_DFLASH_HEAD_RUNTIME_MANIFEST=/opt/paiton-native/draft_head.json
ENV PAITON_EXPERIMENTAL_TARGET_HEAD=1 PAITON_TARGET_HEAD_RUNTIME_MANIFEST=/opt/paiton-native/target_logits.json

ENV PAITON_EXPERIMENTAL_GDN_PREFILL=1 PAITON_GDN_PREFILL_RUNTIME_MANIFEST=/opt/paiton-native/gdn_prefill.json

ENV PAITON_EXPERIMENTAL_GDN_CONV_PREFILL=1 PAITON_GDN_CONV_PREFILL_MANIFEST=/opt/paiton-native/gdn_conv_prefill.json

ENV PAITON_EXPERIMENTAL_SILU_FP8=1 PAITON_SILU_FP8_MANIFEST=/opt/paiton-native/silu_fp8.json

ENV PAITON_EXPERIMENTAL_NORM_FP8=1 PAITON_NORM_FP8_MANIFEST=/opt/paiton-native/norm_fp8.json

COPY release/ /opt/paiton-release/
LABEL org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.title="Paiton Qwen3.8 MXFP4 + DFlash2 on RDNA4" \
      org.opencontainers.image.version="qwen38-mxfp4-dflash2-rdna4-v1.0.0" \
      org.opencontainers.image.licenses="NOASSERTION"
EXPOSE 8000
VOLUME ["/models/cache"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=30m CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2)"
ENTRYPOINT ["python3", "/opt/paiton-release/container_entrypoint.py"]
CMD []
