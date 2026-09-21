"""Separate saved graphs for external runtime-hook selections before imports."""
import hashlib
import json
import os
from pathlib import Path


def configure(root, manifest):
    root = Path(root)
    provider = os.environ.get('PAITON_RUNTIME_COMPAT_NATIVE_PROVIDER', 'off')
    selected = {k: v for k, v in os.environ.items() if k.startswith('RADIANCE_')}
    value = dict(compatibility=manifest, provider=provider, environment=selected,
                 build=os.environ.get('PAITON_RUNTIME_COMPAT_BUILD_ID', 'development'),
                 bootstrap=hashlib.sha256((root/'bootstrap.py').read_bytes()).hexdigest())
    if provider != 'off':
        manifest_key, adapter = {
            'qk_norm': ('PAITON_RUNTIME_COMPAT_QK_MANIFEST', 'runtime_native_qk.py'),
            'gdn_norm_small': ('PAITON_RUNTIME_COMPAT_GDN_NORM_MANIFEST', 'runtime_native_gdn_norm.py'),
        }[provider]
        path = Path(os.environ[manifest_key])
        value['native_manifest'] = json.loads(path.read_text())
        value['native_adapter'] = hashlib.sha256(
            (root.parent/'paiton_vllm_plugin'/adapter).read_bytes()).hexdigest()
    namespace = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]
    # Preserve base paths across spawned processes and provider switches.
    bases = json.loads(os.environ.get('PAITON_RUNTIME_COMPAT_CACHE_BASES', '{}'))
    cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache')))
    defaults = {'VLLM_CACHE_ROOT': str(cache/'vllm'),
                'TORCHINDUCTOR_CACHE_DIR': str(cache/'inductor'),
                'TRITON_CACHE_DIR': str(cache/'triton')}
    for key, default in defaults.items():
        base = bases.setdefault(key, os.environ.get(key, default))
        os.environ[key] = str(Path(base)/'paiton-runtime'/namespace)
    os.environ['PAITON_RUNTIME_COMPAT_CACHE_BASES'] = json.dumps(bases, sort_keys=True)
    os.environ['PAITON_RUNTIME_COMPAT_CACHE_NAMESPACE'] = namespace
    return namespace
