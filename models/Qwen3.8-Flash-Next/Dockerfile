FROM paiton-qwen38-local:qwen38-rocm10-vllm029-20261005-r1@sha256:245c71d54f046f89b74dbfbd4c894003754561e55c9d215a8d3ce65364fb0f80
RUN rm -rf /usr/local/lib/python3.12/dist-packages/paiton_vllm_plugin && cd /usr/local/lib/python3.12/dist-packages && rm -f 000_paiton_runtime_compat.pth zzz_paiton_dflash_prefill_skip.pth zzz_paiton_draft_fp8_gemm.pth zzz_paiton_gdn_fixed.pth zzz_paiton_gdn_merged_prefill.pth
COPY staging/ /
ENV PAITON_PLUGIN_MODE=flashnext VLLM_PLUGINS=register_paiton_models VLLM_DISABLE_PAITON_PLATFORM=1 PAITON_RUNTIME_COMPAT_FLOW=0 PAITON_TP_COLLECTIVE_PATH=/opt/paiton/runtime/tp-collective PAITON_MOE_GROUPED_PATH=/opt/paiton/runtime/moe-grouped
LABEL org.opencontainers.image.title="Paiton Qwen3.8 Flash Next (ROCm 10, vLLM 0.29, TP2)" paiton.plugin_commit="8582563" paiton.bundles="tp-collective-m27h-981f701e,moe-grouped-71dad0d2,fn-mixers-9f92291e"
