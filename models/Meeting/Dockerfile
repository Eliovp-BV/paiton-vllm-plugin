FROM ghcr.io/eliovp/paiton-vllm-plugin@sha256:b8d6bf8e6c1934a82d8c99156061e0a44aed956ea021f73276fc07bf74c88e5e AS media_builder
USER root
COPY --from=meeting_media_sources . /sources
COPY media-sources.lock.json /sources/sources.json
COPY scripts/build_media_libraries.sh /build-media-libraries.sh
RUN python3 -c 'import json,pathlib,hashlib; p=pathlib.Path("/sources"); assert all(hashlib.sha256((p/x["filename"]).read_bytes()).hexdigest()==x["sha256"] for x in json.loads((p/"sources.json").read_text()))'
RUN sh /build-media-libraries.sh
COPY scripts/build_media_python.sh /build-media-python.sh
RUN sh /build-media-python.sh

FROM ghcr.io/eliovp/paiton-vllm-plugin@sha256:b8d6bf8e6c1934a82d8c99156061e0a44aed956ea021f73276fc07bf74c88e5e
USER root
COPY --from=media_builder /opt/meeting-media /opt/meeting-media
COPY --from=media_builder /wheels /tmp/meeting-media-wheels
RUN python3 -m pip install --no-index --no-deps /tmp/meeting-media-wheels/av-*.whl
COPY requirements.lock /opt/paiton/meeting-requirements.lock
RUN python3 -m pip install --no-cache-dir -r /opt/paiton/meeting-requirements.lock
# The inherited interpreter is under /root/.local. Move that runtime to a public
# executable location so Studio can run as the caller's unprivileged UID.
RUN set -eu; runtime="$(readlink -f /opt/python)"; \
    if [ "$runtime" != /opt/python ]; then mv "$runtime" /opt/meeting-python; rm /opt/python; ln -s /opt/meeting-python /opt/python; fi
COPY scripts/patch_vllm_optional_codec.py /opt/paiton/patch_vllm_optional_codec.py
RUN python3 /opt/paiton/patch_vllm_optional_codec.py
COPY pyproject.toml models.lock.json /opt/paiton/meeting/
COPY --from=meeting_overlay . /opt/paiton/meeting-artifacts/
COPY paiton_meeting /opt/paiton/meeting/paiton_meeting
ENV PYTHONPATH=/opt/paiton/meeting \
    HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYANNOTE_METRICS_ENABLED=0 \
    PAITON_MEETING_SUMMARY_BACKEND=vllm \
    XDG_CACHE_HOME=/models/cache/xdg VLLM_PLUGINS="" VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV PAITON_MEETING_ARTIFACT=/opt/paiton/meeting-artifacts/meeting_lstm_float16_gfx1201.so
ARG PLUGIN_REVISION=local
LABEL org.opencontainers.image.revision="${PLUGIN_REVISION}" \
      org.opencontainers.image.title="Paiton local meeting notes for R9700" \
      org.opencontainers.image.version="meeting-rdna4-v1.0.0-rc1" \
      org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      dev.paiton.gpu.arch="gfx1201"
HEALTHCHECK NONE
WORKDIR /opt/paiton/meeting
ENTRYPOINT ["python3", "-m", "paiton_meeting"]
