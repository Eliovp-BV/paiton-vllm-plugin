"""Opt-in external DFlash integration. Native arithmetic belongs to Paiton HIP artifacts."""
import json
import os
from .configuration import Settings

_installed = None

def install():
    """Install once per worker, before draft imports; OFF imports no GPU framework."""
    global _installed
    settings = Settings.from_environment(os.environ)
    if settings.rerank or settings.sample_method != 'inherit' or settings.verify_cap:
        expected = json.dumps([settings.sample_method, settings.rerank, settings.block_candidates, settings.verify_cap])
        if os.environ.get('_PAITON_DFLASH_SERVER_OPTIONS_V3') != expected:
            raise RuntimeError('Use paiton-dflash-serve to apply DFlash proposal options to vLLM')
    if _installed is not None:
        if settings != _installed:
            raise RuntimeError('Paiton DFlash settings changed after startup')
        return settings
    if settings.uniform_graphs:
        from .uniform_graph import install as install_uniform_graphs
        install_uniform_graphs()
    # Source transformations must load before post-import context wrappers.
    # Finders are prepended: registering native first leaves context outermost.
    if settings.native:
        from .native_hooks import install as install_native
        install_native()
    if settings.context_graph:
        from . import context_graph  # external-runtime graph scheduling only
    if settings.context_norm_rope:
        from . import context  # installs the version-pinned ownership/dispatch boundary
    if settings.rerank:
        from .rerank import install as install_rerank
        install_rerank()
    if settings.verify_cap or settings.verify_trace:
        from .verification import install as install_verification
        install_verification(settings.verify_cap, settings.verify_trace)
    _installed = settings
    return settings
