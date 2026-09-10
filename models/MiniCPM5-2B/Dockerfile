# syntax=docker/dockerfile:1.19@sha256:b6afd42430b15f2d2a4c5a02b919e98a525b785b1aaff16747d2f623364e39b6
# Reuse the qualified public ROCm 7.14 runtime and its GHCR-sized layers.
FROM ghcr.io/eliovp/paiton-vllm-plugin@sha256:0cf9c11304ade58e91d97db876a4462781fa2c18d2d4846f26d56fe5d207669d
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md /opt/paiton/minicpm-release-source/
COPY LICENSES /opt/paiton/minicpm-release-source/LICENSES
COPY paiton_vllm_plugin /opt/paiton/minicpm-release-source/paiton_vllm_plugin
ARG SOURCE_DATE_EPOCH=1789031647
RUN python3 -m pip install --no-index --no-build-isolation --no-deps --target /opt/paiton/minicpm-plugin /opt/paiton/minicpm-release-source
ENV PYTHONPATH=/opt/paiton/minicpm-plugin:/opt/python/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi \
    VLLM_DISABLE_PAITON_PLATFORM=1 VLLM_ROCM_USE_AITER=0 OMP_NUM_THREADS=1 \
    VLLM_WORKER_MULTIPROC_METHOD=fork VLLM_USE_V2_MODEL_RUNNER=1
COPY --exclude=__pycache__ --exclude=**/__pycache__ --exclude=*.pyc --exclude=**/*.pyc \
     --exclude=README.md --exclude=BENCHMARKS.md --exclude=SELECTION.md \
     --exclude=REPRODUCE.md --exclude=results.json --exclude=container-images.json \
     --exclude=benchmark/results --exclude=Dockerfile --exclude=serve-docker.sh \
     models/MiniCPM5-2B /opt/paiton/minicpm5/
RUN python3 /opt/paiton/minicpm5/patches/apply.py
RUN python3 /opt/paiton/minicpm5/runtime_link_check.py check --manifest /opt/paiton/runtime-links.json
COPY --from=minicpm_overlay . /opt/paiton/minicpm5-artifacts/
ENV HF_HOME=/models/cache/huggingface HF_HUB_CACHE=/models/cache/huggingface/hub \
    XDG_CACHE_HOME=/models/cache TRITON_CACHE_DIR=/models/cache/triton VLLM_CACHE_ROOT=/models/cache/vllm \
    PAITON_MINICPM5_ARTIFACT=/opt/paiton/minicpm5-artifacts/minicpm5_awq_decode_gfx1201.so
ARG PLUGIN_REVISION=local
LABEL org.opencontainers.image.revision="${PLUGIN_REVISION}" \
      org.opencontainers.image.title="Paiton MiniCPM5-2B W4A16 R9700" \
      org.opencontainers.image.version="minicpm5-2b-w4a16-rdna4-v1.0.0" \
      org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      dev.paiton.model.revision="6c1ee6fa521aa53f47cfb32696e6d8ef5b0db805" \
      dev.paiton.gpu.arch="gfx1201"
VOLUME ["/models/cache"]
EXPOSE 8036
HEALTHCHECK --interval=30s --timeout=3s --start-period=30m --retries=3 CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8036/health', timeout=2)"
ENTRYPOINT ["python3", "/opt/paiton/minicpm5/server.py"]
