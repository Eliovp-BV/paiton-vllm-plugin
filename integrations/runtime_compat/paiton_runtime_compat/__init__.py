"""external runtime external serving compatibility; no Paiton compiler/model substitution."""
import os
import sys

_installed = False


def register():
    global _installed
    if _installed:
        return
    provider = os.environ.get("PAITON_RUNTIME_COMPAT_NATIVE_PROVIDER", "off")
    if provider not in ("off", "qk_norm", "gdn_norm_small"):
        raise RuntimeError("Native provider is not qualified in this compatibility lane")
    if os.environ.get("PAITON_RUNTIME_COMPAT_FLOW", "0") == "1":
        from . import bootstrap
        if not bootstrap._installed:
            raise RuntimeError("external runtime compatibility bootstrap did not run before vLLM imports")
    # Invoke the original hook; fail closed if its outer initialization raises.
    # The deployment pins the complete qualified external runtime and providers.
    try:
        import radiance_kernels
        radiance_kernels.install_all()
    except Exception as exc:
        sys.stderr.write(f"[radiance] install_all failed: {exc!r}\n")
        raise RuntimeError("external runtime compatibility initialization failed") from exc
    if provider == "qk_norm":
        from paiton_vllm_plugin.runtime_native_qk import install
        install()
    if provider == "gdn_norm_small":
        from paiton_vllm_plugin.runtime_native_gdn_norm import install
        install()
    _installed = True
    sys.stderr.write(f"[paiton.runtime_flow] original external runtime hooks installed; native_provider={provider}\n")
