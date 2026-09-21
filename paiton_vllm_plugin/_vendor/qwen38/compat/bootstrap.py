"""Pinned compatibility overlays for an unchanged upstream serving installation.

This external integration contains no Paiton compiler or native execution path.
The finder runs at interpreter startup; regular vLLM plugin registration installs
external runtime's runtime hooks later, at their original phase.
"""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import linecache
import json
import os
from pathlib import Path
import sys

_installed = False


class CompatibilityLoader(importlib.machinery.SourceFileLoader):
    """Keep canonical code filenames for framework tracing and resource lookup."""
    def __init__(self, fullname, overlay, original):
        super().__init__(fullname, overlay)
        self.original = original

    def get_code(self, fullname):
        data = self.get_data(self.path)
        source = importlib.util.decode_source(data)
        # mtime=None keeps inspect/linecache on the actual loaded source without
        # overwriting the upstream file or changing framework filename rules.
        linecache.cache[self.original] = (
            len(data), None, source.splitlines(keepends=True), self.original)
        return self.source_to_code(data, self.original)


class OverlayFinder(importlib.abc.MetaPathFinder):
    def __init__(self, manifest, root):
        self.root = Path(root).resolve()
        self.entries = manifest['files']
        self.loaded = set()

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.origin not in self.entries:
            return None
        entry = self.entries[spec.origin]
        original = Path(spec.origin)
        overlay = (self.root / entry['overlay']).resolve()
        if not overlay.is_relative_to(self.root):
            raise ImportError('external runtime overlay escapes its package')
        if hashlib.sha256(original.read_bytes()).hexdigest() != entry['original_sha256']:
            raise ImportError('external runtime overlay original-source mismatch: ' + fullname)
        if hashlib.sha256(overlay.read_bytes()).hexdigest() != entry['overlay_sha256']:
            raise ImportError('external runtime overlay payload mismatch: ' + fullname)
        # Framework tracing uses code filenames as well as module identity.
        # All three remain canonical; only the loader's source payload differs.
        spec.loader = CompatibilityLoader(fullname, str(overlay), str(original))
        spec.cached = None
        self.loaded.add(fullname)
        return spec


def install():
    global _installed
    if _installed or os.environ.get('PAITON_RUNTIME_COMPAT_FLOW', '0') != '1':
        return
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'compat_manifest.json').read_text())
    from .cache_namespace import configure
    configure(root, manifest)
    origins = set(manifest['files'])
    early = [name for name, module in sys.modules.items()
             if getattr(module, '__file__', None) in origins]
    if early:
        raise RuntimeError('external runtime bootstrap ran after affected imports: ' + ','.join(early))
    sys.meta_path.insert(0, OverlayFinder(manifest, root))
    _installed = True
