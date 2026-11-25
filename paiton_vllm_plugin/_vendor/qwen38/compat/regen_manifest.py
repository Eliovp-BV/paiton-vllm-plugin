#!/usr/bin/env python3
"""regen_manifest.py: rebuild paiton_runtime_compat/compat_manifest.json for this interpreter's site-packages.
Runs inside the image build. Every file under compat/overlays/ must have an original at the same relative path in
site-packages; the manifest pins both sha256 values (bootstrap.py refuses a mismatch at import)."""
import hashlib, json, sys, sysconfig
from pathlib import Path

import vllm

site = Path(sysconfig.get_paths()['purelib'])
root = site / 'paiton_runtime_compat'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
old = json.loads((root / 'compat_manifest.json').read_text())
files, missing = {}, []
for ov in sorted((root / 'overlays').rglob('*.py')):
    rel = ov.relative_to(root / 'overlays')
    orig = site / rel
    if not orig.is_file():
        missing.append(str(rel)); continue
    files[str(orig)] = {'overlay': f'overlays/{rel}', 'original_sha256': sha(orig), 'overlay_sha256': sha(ov)}
if missing:
    sys.exit('overlay without an original in site-packages: ' + ', '.join(missing))
man = {'vllm': vllm.__version__, 'native_provider': old.get('native_provider', 'off'), 'files': files}
(root / 'compat_manifest.json').write_text(json.dumps(man, indent=1, sort_keys=True) + '\n')
print(f'compat manifest: vllm {vllm.__version__}, {len(files)} overlays, site {site}')
