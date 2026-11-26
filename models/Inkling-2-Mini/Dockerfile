# syntax=docker/dockerfile:1.19@sha256:b6afd42430b15f2d2a4c5a02b919e98a525b785b1aaff16747d2f623364e39b6
# Inkling 2 Mini on one R9700 (gfx1201): AMD's ROCm 10.2 / vLLM 0.32 image plus the plugin and Paiton's native
# libraries. Build from the repository root with the libraries as a named context:
#   docker build -f models/Inkling-2-Mini/Dockerfile --build-context inkling2_libs=<folder with the .so files> \
#     --build-context runner_glue=<folder with the runner glue files> --build-context triton_seed=<Triton cache folder> .
FROM rocm/vllm@sha256:380efeed0d5fa65fc044253adc1ce53a04907d1202b30fb36e175370ee39a003
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md /opt/paiton/inkling2-release-source/
# Only the parts of the plugin that Inkling 2 loads (import trace of the launcher in every mode, 25 Nov 2026): the entry
# point, its two helpers and the inkling2 package. The console scripts are removed: they point at modules left out here.
COPY paiton_vllm_plugin/__init__.py paiton_vllm_plugin/activation.py paiton_vllm_plugin/artifact_manifest.py /opt/paiton/inkling2-release-source/paiton_vllm_plugin/
COPY paiton_vllm_plugin/inkling2 /opt/paiton/inkling2-release-source/paiton_vllm_plugin/inkling2
RUN python3 -m pip install --no-index --no-build-isolation --no-deps --target /opt/paiton/inkling2-plugin /opt/paiton/inkling2-release-source \
    && rm -rf /opt/paiton/inkling2-plugin/bin
COPY models/Inkling-2-Mini/server.py models/Inkling-2-Mini/checkpoint.lock.json models/Inkling-2-Mini/w3x.lock.json /opt/paiton/inkling2/
# vLLM's offload worker as in the 27B release image, used only by the prompt-cache tiers (--host-cache-gib,
# --extend-cache): the RAM tier as one exact-size hipHostMalloc allocation per tensor (PAITON_HOST_KV_PRIVATE=1), and no
# silent fall back to pageable copies, which corrupt data on gfx1201 (ROCm clr #291). With it, vLLM PR #52838 on the
# offloading connector: a failed cache copy no longer stops the server (a failed store drops those blocks, a failed load
# fails only its request). Same files as the 27B backport; the scheduler is vLLM 0.32's own plus the PR. The files they
# replace are checked.
RUN printf '%s\n' \
    "d074628708ad0378e9b73502c1f3945a647c531d54102a1db9f90ba1c7f98dff  /usr/local/lib/python3.14/dist-packages/vllm/v1/kv_offload/cpu/gpu_worker.py" \
    "fa748f6f40a76e83f7ece683e90d6cdede9a069c45438cfcdcdb70ff2dd901fe  /usr/local/lib/python3.14/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py" \
    "9b3f042c6fa1581d10e8f688d2d9bc927bb41b780a9ea01f514d57228e20c978  /usr/local/lib/python3.14/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py" \
    "f9e2f9ad8a637b54c88589ebc58353b236fafe6688def4b22b68f275decbbc6b  /usr/local/lib/python3.14/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py" \
    "ca61db86f61b4b2638102ebb04d8a6cf20947203926a31a9f0cccf284ce2bcf8  /usr/local/lib/python3.14/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py" | sha256sum -c
COPY models/Inkling-2-Mini/vllm-overlays/ /usr/local/lib/python3.14/dist-packages/
# The folder holds the seven libraries and MANIFEST.json (sha256 and source commit of each).
COPY --from=inkling2_libs . /opt/paiton/inkling2/lib/
# vLLM 0.32's model-runner bookkeeping kernels (input preparation, block tables, slot mapping, drafter bookkeeping) in
# native HIP: the runner glue of the Qwen3.8 27B r6 image, unchanged. The adapter checks the library's sha256 against
# manifest.json; PAITON_RUNNER_GLUE=1 swaps the kernels in.
COPY --from=runner_glue libpaiton_runner_glue.so manifest.json /opt/paiton/runtime/runner-glue/
COPY --from=runner_glue paiton_runner_glue_adapter.py /opt/paiton/runner-glue-py/
COPY --from=runner_glue zzz_paiton_runner_glue.pth /usr/local/lib/python3.14/dist-packages/
# vLLM's sampling kernels still compile with Triton on first use (7 to 14 s on the first sampled request after a start).
# The seed holds them, compiled by a serving run of this image with the cache at this same path, so the cache's
# absolute paths resolve; kernels not in it compile into the container as before.
COPY --from=triton_seed . /opt/paiton/cache-seed/triton/
RUN chmod -R a+rwX /opt/paiton/cache-seed
# The base image keeps amdsmi on its PYTHONPATH and its own LD_LIBRARY_PATH: prepend, never replace.
# GPU_MAX_HW_QUEUES=1: a fourth working hardware queue slows decode on RDNA4.
ENV PYTHONPATH=/opt/paiton/inkling2-plugin:${PYTHONPATH} GPU_MAX_HW_QUEUES=1 \
    HF_HOME=/models/cache/huggingface HF_HUB_CACHE=/models/cache/huggingface/hub \
    XDG_CACHE_HOME=/models/cache VLLM_CACHE_ROOT=/models/cache/vllm VLLM_DO_NOT_TRACK=1 \
    PAITON_RUNNER_GLUE=1 TRITON_CACHE_DIR=/opt/paiton/cache-seed/triton \
    PAITON_INK2_MOE_SO=/opt/paiton/inkling2/lib/inkling2_moe_gfx1201.so \
    PAITON_INK2_ATTN_SO=/opt/paiton/inkling2/lib/inkling2_attn_gfx1201.so \
    PAITON_INK2_OPS_SO=/opt/paiton/inkling2/lib/inkling2_ops_gfx1201.so \
    PAITON_INK2_STEP_SO=/opt/paiton/inkling2/lib/inkling2_step_gfx1201.so \
    PAITON_INK2_VISION_SO=/opt/paiton/inkling2/lib/inkling2_vision_gfx1201.so \
    PAITON_INK2_DSPARK_SO=/opt/paiton/inkling2/lib/inkling2_dspark_gfx1201.so \
    PAITON_INK2_DFLASH2_SO=/opt/paiton/inkling2/lib/inkling2_dflash2_gfx1201.so
# INK2_IMAGE=w3 builds the 3-bit image (inkling-2-mini-w3-rdna4-...): the launcher then starts with its settings
# (server.py, W3_FAST) and needs a W3X folder. Empty = the main image.
ARG INK2_IMAGE=
ENV PAITON_INK2_IMAGE=${INK2_IMAGE}
ARG PLUGIN_REVISION=local
ARG IMAGE_TITLE="Paiton Inkling 2 Mini R9700"
ARG IMAGE_VERSION=inkling-2-mini-rdna4-v1.0.0
LABEL org.opencontainers.image.revision="${PLUGIN_REVISION}" \
      org.opencontainers.image.title="${IMAGE_TITLE}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.source="https://github.com/Eliovp-BV/paiton-vllm-plugin" \
      org.opencontainers.image.licenses="Apache-2.0" \
      dev.paiton.model.revision="7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6" \
      dev.paiton.gpu.arch="gfx1201"
VOLUME ["/models/cache"]
EXPOSE 8042
HEALTHCHECK --interval=30s --timeout=3s --start-period=30m --retries=3 CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8042/health', timeout=2)"
ENTRYPOINT ["python3", "/opt/paiton/inkling2/server.py"]
