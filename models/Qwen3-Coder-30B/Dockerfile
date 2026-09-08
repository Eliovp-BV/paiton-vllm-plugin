# syntax=docker/dockerfile:1.7
FROM ghcr.io/eliovp/paiton-vllm-plugin@sha256:c56baf54aca1ad229829c1de26e8792806608e65ee9d06ee210b79cd49f70bc9

ARG PLUGIN_REVISION=local
LABEL org.opencontainers.image.title="Paiton Qwen3-Coder 30B R9700 coding server" \
      org.opencontainers.image.description="Qwen3-Coder coding chat and API on one Radeon AI PRO R9700" \
      org.opencontainers.image.version="qwen3-coder-30b-awq-rdna4-v1.0.0" \
      org.opencontainers.image.revision="${PLUGIN_REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      org.opencontainers.image.base.name="ghcr.io/eliovp/paiton-vllm-plugin" \
      org.opencontainers.image.base.digest="sha256:c56baf54aca1ad229829c1de26e8792806608e65ee9d06ee210b79cd49f70bc9" \
      dev.paiton.qualified-base.digest="sha256:c56baf54aca1ad229829c1de26e8792806608e65ee9d06ee210b79cd49f70bc9" \
      dev.paiton.model="cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit" \
      dev.paiton.model.revision="4bd30395b72ea6045edd04806c4fea448d4467b3" \
      dev.paiton.artifact.sha256="74fb7015e285d3b7863fc8f54341204a252a2ab305921edd5b30f5724915958f" \
      dev.paiton.config.sha256="42981b44892a1940c8f93bf5fb40972bc62f2c9a6c8f0d10115175e68b11396d" \
      dev.paiton.artifact-manifest.sha256="83284e476ce918726e0e216d8c29b9b7326061306380b7c875d6625b16872769" \
      dev.paiton.lm-head-artifact.sha256="" \
      dev.paiton.lm-head-artifact-manifest.sha256="" \
      dev.paiton.release-metadata.sha256="" \
      dev.paiton.gpu.arch="gfx1201"

# The separate build context contains only the compiled artifact,
# manifest and AMD SMI runtime package with its license. No compiler source.
COPY --from=qwen3_coder_overlay . /opt/paiton/qwen3-coder-overlay/
COPY paiton_vllm_plugin /opt/paiton/release-plugin/paiton_vllm_plugin/
COPY models/Qwen3-Coder-30B /opt/paiton/qwen3-coder/
COPY LICENSE LICENSES THIRD_PARTY_NOTICES.md /opt/paiton/qwen3-coder/notices/
RUN cp -a /opt/paiton/qwen3-coder-overlay/amdsmi /opt/paiton/venv-site-packages/amdsmi
RUN python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('/opt/paiton/qwen3-coder-overlay')
artifact = root / 'qwen3_coder_moe_w4a16_g32_gfx1201.so'
manifest = json.loads(artifact.with_suffix('.json').read_text())
assert hashlib.sha256(artifact.read_bytes()).hexdigest() == manifest['sha256'] == '74fb7015e285d3b7863fc8f54341204a252a2ab305921edd5b30f5724915958f'
assert artifact.with_suffix('.json').read_bytes() == Path('/opt/paiton/qwen3-coder/artifact-manifest.json').read_bytes()
PY

ENV HF_HOME=/models/cache/huggingface \
    HF_HUB_CACHE=/models/cache/huggingface/hub \
    XDG_CACHE_HOME=/models/cache \
    TRITON_CACHE_DIR=/models/cache/triton \
    VLLM_CACHE_ROOT=/models/cache/vllm \
    VLLM_PLUGINS=register_paiton_models \
    VLLM_USE_PAITON_PLATFORM=0 \
    VLLM_DISABLE_PAITON_PLATFORM=1 \
    VLLM_ROCM_USE_AITER=0 \
    OMP_NUM_THREADS=1 \
    PAITON_QWEN3_CODER_ARTIFACT=/opt/paiton/qwen3-coder-overlay/qwen3_coder_moe_w4a16_g32_gfx1201.so
VOLUME ["/models/cache"]
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=3s --start-period=30m --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=2)"
ENTRYPOINT ["python3", "/opt/paiton/qwen3-coder/server.py"]
