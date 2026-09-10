# syntax=docker/dockerfile:1.19@sha256:b6afd42430b15f2d2a4c5a02b919e98a525b785b1aaff16747d2f623364e39b6
FROM rocm/vllm-dev@sha256:4fa5bc9c24d25ef7e0fe38b23fc5f2b27fc986177bd2f9e496997bcf5af03871 AS qualified
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md /opt/paiton/release-source/
COPY LICENSES /opt/paiton/release-source/LICENSES
COPY paiton_vllm_plugin /opt/paiton/release-source/paiton_vllm_plugin
RUN python3 -m pip install --no-build-isolation --no-deps --target /opt/paiton/release-plugin /opt/paiton/release-source
COPY models/GPT-OSS-20B/runtime_link_check.py /tmp/runtime_link_check.py
RUN python3 /tmp/runtime_link_check.py snapshot --manifest /opt/paiton/runtime-links.json
FROM ubuntu:24.04@sha256:1e0a86e57d247923571b75e0aaf48a1449cf8c543d51fb3e07a4a7d7bfa79316
COPY --from=qualified /usr /usr
COPY --from=qualified /var/lib/dpkg /var/lib/dpkg
COPY --from=qualified /etc/ssl/certs /etc/ssl/certs
# Keep SDK hard-link groups together: separate copies can load duplicate ROCm DSOs.
# Torch is a separate layer; each compressed layer stays below GHCR's 10 GB limit.
COPY --from=qualified --exclude=lib/python3.14/site-packages/torch /opt/python /opt/python
COPY --from=qualified /opt/python/lib/python3.14/site-packages/torch /opt/python/lib/python3.14/site-packages/torch
COPY --from=qualified /root/.local/share/uv/python /root/.local/share/uv/python
# The standalone Python runtime must also work for the unprivileged code sandbox.
RUN chmod 0755 /root
COPY --from=qualified /app /app
COPY --from=qualified /opt/paiton/runtime-links.json /opt/paiton/runtime-links.json
COPY --from=qualified /opt/paiton/release-plugin /opt/paiton/release-plugin
COPY --from=qualified /opt/paiton/release-source/LICENSE /licenses/paiton/LICENSE
COPY --from=qualified /opt/paiton/release-source/LICENSES /licenses/third-party
COPY --from=qualified /opt/paiton/release-source/THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
ENV PATH=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/bin:/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin:/opt/python/lib/python3.14/site-packages/_rocm_sdk_core/bin:/opt/python/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    ROCM_PATH=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel \
    ROCM_HOME=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel \
    HIP_PATH=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel \
    HIP_CLANG_PATH=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin \
    HIP_DEVICE_LIB_PATH=/opt/python/lib/python3.14/site-packages/_rocm_sdk_core/lib/llvm/amdgcn/bitcode \
    LD_LIBRARY_PATH=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib:/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib/rocm_sysdeps/lib:/opt/python/lib/python3.14/site-packages/_rocm_sdk_core/lib:/opt/python/lib/python3.14/site-packages/_rocm_sdk_core/lib/rocm_sysdeps/lib \
    PYTHONPATH=/opt/paiton/release-plugin:/opt/python/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi \
    HSA_NO_SCRATCH_RECLAIM=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 \
    ROCM_VERSION=7.14.0 TOKENIZERS_PARALLELISM=false SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1 \
    KINETO_CONFIG=/app/libkineto.conf TORCH_BLAS_PREFER_HIPBLASLT=1 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_DO_NOT_TRACK=1 VLLM_ROCM_ENABLE_CUDAGRAPH=TRUE \
    VLLM_DISABLE_PAITON_PLATFORM=1 VLLM_ROCM_USE_AITER=0 OMP_NUM_THREADS=1
COPY --exclude=__pycache__ --exclude=**/__pycache__ --exclude=*.pyc --exclude=**/*.pyc \
     --exclude=README.md --exclude=BENCHMARKS.md --exclude=SELECTION.md \
     --exclude=REPRODUCE.md --exclude=results.json --exclude=container-images.json \
     --exclude=benchmark/results --exclude=Dockerfile --exclude=serve-docker.sh \
     models/GPT-OSS-20B /opt/paiton/gpt-oss/
RUN python3 /opt/paiton/gpt-oss/runtime_link_check.py check --manifest /opt/paiton/runtime-links.json
COPY --from=gptoss_overlay . /opt/paiton/gpt-oss-overlay/
ENV HF_HOME=/models/cache/huggingface HF_HUB_CACHE=/models/cache/huggingface/hub \
    XDG_CACHE_HOME=/models/cache TRITON_CACHE_DIR=/models/cache/triton VLLM_CACHE_ROOT=/models/cache/vllm \
    PAITON_GPTOSS_ARTIFACT=/opt/paiton/gpt-oss-overlay/gptoss_mxfp4_decode_gfx1201.so
# Harmony loads its vocabulary separately from the Hugging Face tokenizer.
ADD --chmod=0644 --checksum=sha256:446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken /opt/paiton/harmony-cache/fb374d419588a4632f3f557e76b4b70aebbca790
ENV TIKTOKEN_RS_CACHE_DIR=/opt/paiton/harmony-cache
ARG PLUGIN_REVISION=local
LABEL org.opencontainers.image.revision="${PLUGIN_REVISION}" \
      org.opencontainers.image.title="Paiton GPT-OSS-20B R9700 community candidate" \
      org.opencontainers.image.version="gpt-oss-20b-mxfp4-rdna4-v1.0.0" \
      org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      dev.paiton.model.revision="6cee5e81ee83917806bbde320786a8fb61efebee" \
      dev.paiton.gpu.arch="gfx1201"
VOLUME ["/models/cache"]
EXPOSE 8020
HEALTHCHECK --interval=30s --timeout=3s --start-period=30m --retries=3 CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8020/health', timeout=2)"
ENTRYPOINT ["python3", "/opt/paiton/gpt-oss/server.py"]
