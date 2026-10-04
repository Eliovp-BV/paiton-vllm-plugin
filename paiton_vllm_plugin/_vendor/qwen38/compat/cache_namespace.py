"""Separate saved graphs for external runtime-hook selections before imports."""
import hashlib
import json
import os
import shutil
import sys
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
    if os.environ.get('PAITON_COMPILE_CACHE', '0') == '1':
        # PAITON opt-in persistent compile cache: the W3 decode adapter disables vLLM's cache because a graph compiled
        # for one weight path could be reloaded into another. Key the namespace on every PAITON_ setting and on the
        # installed runtime pieces (adapters, radiance modules, native libraries), so such a reload cannot happen.
        value['paiton_env'] = {k: v for k, v in os.environ.items()
                               if k.startswith('PAITON_') and not k.startswith('PAITON_RUNTIME_COMPAT_CACHE')}
        listing = []
        for base in (root.parent, Path('/opt/paiton')):
            if base.is_dir():
                for f in sorted(base.rglob('*') if base.name == 'paiton' else base.glob('*')):
                    if 'cache-seed' in f.parts:   # the shipped compile-cache seed is not a runtime piece
                        continue
                    if f.is_file() and (base.name == 'paiton' or f.name.startswith(('paiton', 'radiance', 'r4d',
                                                                                        'libr4d', 'zzz'))):
                        st = f.stat()
                        listing.append((str(f), st.st_size, int(st.st_mtime)))
        value['paiton_runtime'] = hashlib.sha256(json.dumps(listing).encode()).hexdigest()
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
    seed_caches(cache)
    return namespace


def seed_caches(xdg_cache):
    """Fill an empty compile cache from the image's seed (start-time item A4).

    A first start with an empty runtime cache spends minutes compiling Triton and inductor kernels and comgr code objects
    that are the same on every host with this image: their cache entries are keyed by content hashes (kernel source,
    constants, target), not by paths or times, and this namespace only partitions them. The image may ship those entries
    under PAITON_CACHE_SEED (default /opt/paiton/cache-seed/{triton,inductor,comgr}, a Paiton path the release scan covers); when a target directory does
    not exist yet, it is created from the seed with a copy-and-rename, so a concurrent process sees either nothing or the
    complete seed. A directory that exists (a warm cache) is never touched. PAITON_CACHE_SEED=off disables it.
    """
    seed = os.environ.get('PAITON_CACHE_SEED', '/opt/paiton/cache-seed')
    if seed == 'off' or not Path(seed).is_dir():
        return {}
    targets = {'triton': Path(os.environ['TRITON_CACHE_DIR']), 'inductor': Path(os.environ['TORCHINDUCTOR_CACHE_DIR']),
               'comgr': Path(os.environ.get('AMD_COMGR_CACHE_DIR', str(Path(xdg_cache)/'comgr')))}
    done = {}
    for kind, target in targets.items():
        source = Path(seed)/kind
        if not source.is_dir() or target.exists():
            continue
        tmp = target.with_name(f'{target.name}.seed.tmp.{os.getpid()}')
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, tmp, symlinks=True)
            os.rename(tmp, target)
            done[kind] = sum(1 for _ in target.rglob('*') if _.is_file())
        except FileExistsError:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001  (a seed must never stop a start)
            shutil.rmtree(tmp, ignore_errors=True)
            done[kind] = f'failed: {exc!r}'
    if done:
        print('[paiton.cache] ' + json.dumps({'seeded': done, 'seed': seed, 'pid': os.getpid()}, sort_keys=True), file=sys.stderr, flush=True)
    return done
