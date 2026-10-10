#!/usr/bin/env python3
"""Assemble the hash-locked build context of the Qwen3.8-Flash-Next image without downloads or builds.

Inputs (never compiler sources or build tools):
  --plugin-src DIR   a checkout of paiton_vllm_plugin at the locked plugin commit (build-context.lock.json: plugin_package)
  --artifacts DIR    the staged release artefacts: runtime/{tp-collective,moe-grouped,fn-mixers} (compiled bundles) and
                     cache-seed/{a8,a8-nopf,MANIFEST.sha256} (compile-cache seeds), as distributed with the release
  --out DIR          the context directory to create (must not exist); it receives Dockerfile + staging/ and is verified file by file
Every file is checked against build-context.lock.json before `docker build`; a missing or differing file stops the run.
"""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
SITE = 'staging/usr/local/lib/python3.12/dist-packages/paiton_vllm_plugin/'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plugin-src', required=True); ap.add_argument('--artifacts', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); lock = json.loads((HERE / 'build-context.lock.json').read_text()); out = Path(a.out)
    if out.exists(): sys.exit(f'{out} exists; choose a new directory')
    plugin = Path(a.plugin_src); art = Path(a.artifacts)
    if (plugin / 'paiton_vllm_plugin').is_dir(): plugin = plugin / 'paiton_vllm_plugin'
    out.mkdir(parents=True); shutil.copy(HERE / 'Dockerfile', out / 'Dockerfile')
    missing, differ = [], []
    for rel, want in lock['files'].items():
        if rel == 'Dockerfile': src = HERE / 'Dockerfile'
        elif rel.startswith(SITE): src = plugin / rel[len(SITE):]
        elif rel.startswith('staging/opt/paiton/'): src = art / rel[len('staging/opt/paiton/'):]
        else: sys.exit(f'unexpected locked path {rel}')
        if not src.is_file(): missing.append(rel); continue
        if sha(src) != want: differ.append(rel); continue
        dst = out / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, dst)
    if missing or differ:
        print('context NOT assembled:', len(missing), 'missing,', len(differ), 'differing files'); [print('  missing', m) for m in missing[:20]]; [print('  differs', d) for d in differ[:20]]; sys.exit(2)
    print(f'context ready: {len(lock["files"])} files verified -> {out}'); print(f'build: docker build -t <tag> {out}'); print(f'base image: {lock["base_image"]}')

if __name__ == '__main__':
    main()
