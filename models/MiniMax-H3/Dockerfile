# syntax=docker/dockerfile:1.7
# Publishable artifact/dependency image for MiniMax H3 Ref2VA v1.1.0.
# ComfyUI, the frontend and AIMDO are NOT in this image. The end user builds
# Dockerfile.local from pinned upstream sources for the GUI path.
ARG BASE_IMAGE=ghcr.io/eliovp/paiton-vllm-plugin@sha256:c56baf54aca1ad229829c1de26e8792806608e65ee9d06ee210b79cd49f70bc9
FROM ${BASE_IMAGE}

# Build args carry the frozen hashes from release.lock.json; do not hard-code them.
ARG ARTIFACT_SHA256=""
ARG ARTIFACT_MANIFEST_SHA256=""

HEALTHCHECK NONE
LABEL org.opencontainers.image.title="Paiton MiniMax H3 Ref2VA RDNA4 artifacts" \
      org.opencontainers.image.description="Artifact-only gfx1201 runtime; weights download from pinned publishers" \
      org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.version="minimax-h3-rdna4-v1.1.0" \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-MiniMax-H3-Community" \
      org.opencontainers.image.base.name="ghcr.io/eliovp/paiton-vllm-plugin" \
      org.opencontainers.image.base.digest="sha256:c56baf54aca1ad229829c1de26e8792806608e65ee9d06ee210b79cd49f70bc9" \
      dev.paiton.gpu.arch="gfx1201" \
      dev.paiton.contract="ref2va-v1" \
      dev.paiton.distribution="artifact-image" \
      dev.paiton.model="EliovpAI/MiniMax-H3-W4A8-Paiton-RDNA4" \
      dev.paiton.model.revision="paiton-rdna4-v1.1.0" \
      dev.paiton.artifact.filename="minimax_h3_projections_gfx1201.so" \
      dev.paiton.artifact.sha256="${ARTIFACT_SHA256}" \
      dev.paiton.artifact-manifest.sha256="${ARTIFACT_MANIFEST_SHA256}" \
      dev.paiton.lm-head-artifact.sha256="" \
      dev.paiton.lm-head-artifact-manifest.sha256="" \
      dev.paiton.config.sha256="" \
      dev.paiton.release-metadata.sha256=""
# The empty overrides above clear stale labels inherited from the vLLM base
# image (Qwen3.8 model/artifact/LM-head hashes). Do not remove them.


ENV PYTHONPATH=/opt/h3:/opt/h3-deps:/opt/paiton/venv-site-packages:/opt/paiton/site-packages \
    PAITON_H3_DATA=/data PAITON_COMFY_ROOT=/opt/comfyui \
    HF_HOME=/data/cache HF_HUB_CACHE=/data/cache/hub \
    TRITON_CACHE_DIR=/data/cache/triton TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor \
    XDG_CACHE_HOME=/data/cache/xdg \
    VLLM_USE_PAITON_PLATFORM=0 VLLM_DISABLE_PAITON_PLATFORM=1 \
    MIOPEN_FIND_MODE=FAST HF_HUB_DISABLE_TELEMETRY=1

COPY requirements.lock.txt /opt/h3/requirements.lock.txt
COPY paiton_video /opt/h3/paiton_video
# Native runner (P1). Uncomment only when paiton_h3/ is part of the RC and the
# equality gate has passed; then switch ENTRYPOINT to python3 -m paiton_h3.
# COPY paiton_h3 /opt/h3/paiton_h3
COPY artifacts /opt/h3/artifacts
COPY checkpoints.lock.json extra_model_paths.yaml LICENSE NOTICE MODIFICATIONS.md \
     THIRD_PARTY_NOTICES.md verify.py /opt/h3/
# release.lock.json is deliberately NOT baked into the image: its image digest
# cannot be known before the build, and a zero/placeholder value would be
# misleading. The lock ships in the release bundle and is verified out of tree.
COPY LICENSES /opt/h3/LICENSES

# Build-time gate: artifact and manifest agree, contract is Ref2VA, no compiler
# paths leak into the image. Runtime re-verifies before loading.
RUN test -s /opt/h3/artifacts/minimax_h3_projections_gfx1201.so \
 && python3 - <<'PY'
import hashlib, json, pathlib
artifact = pathlib.Path("/opt/h3/artifacts/minimax_h3_projections_gfx1201.so")
manifest = json.loads(artifact.with_suffix(".manifest.json").read_text())
digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
assert manifest["artifact"]["sha256"] == digest, "artifact hash mismatch"
assert manifest["artifact"]["size_bytes"] == artifact.stat().st_size, "artifact size mismatch"
assert manifest["target"]["arch"] == "gfx1201"
assert manifest["parallelism"]["tp_size"] == 1
assert manifest["h3_max_tokens"] == 49152
assert manifest["h3_contract_version"] == 1
assert manifest["h3_packed_layout"] == ["text", "refs", "audio", "video"]
assert manifest["h3_ref_limits"] == {"images": 9, "videos": 3, "audios": 3}
assert "ref2va" in manifest["h3_capabilities"]
print("artifact contract ok", digest)
PY

RUN python3 -c 'import torch; assert torch.version.hip; print(torch.__version__)'
RUN python3 -c 'import importlib.util as u; assert u.find_spec("comfy") is None; assert u.find_spec("comfyui_frontend_package") is None; assert u.find_spec("comfy_aimdo") is None'

WORKDIR /outputs
ENTRYPOINT ["python3", "-m", "paiton_video"]
CMD ["--help"]
