"""Pinned model download and verification before ordinary vLLM serving."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def verify_snapshot(folder, spec):
    for name, expected in spec['files'].items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != expected['bytes']:
            raise ValueError(f'Missing or wrong-size model file: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected['sha256']:
            raise ValueError(f'Model hash mismatch: {path}')


def resolve_snapshot(spec, folder, cache, offline):
    if folder is None:
        from huggingface_hub import snapshot_download
        folder = Path(snapshot_download(repo_id=spec['repository'], revision=spec['revision'],
            allow_patterns=list(spec['files']), cache_dir=str(cache / 'hub'),
            local_files_only=offline, max_workers=2))
    folder = folder.resolve()
    print(f"Verifying {spec['repository']} at {spec['revision']}...", flush=True)
    verify_snapshot(folder, spec)
    return folder


def engine_command(profile, target, draft):
    args = list(profile['arguments'])
    for flag in ('--model', '--tokenizer'):
        args[args.index(flag) + 1] = str(target)
    index = args.index('--speculative-config') + 1
    spec = json.loads(args[index]); spec['model'] = str(draft)
    args[index] = json.dumps(spec)
    return [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', *args]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, help='Existing pinned target snapshot')
    parser.add_argument('--draft', type=Path, help='Existing pinned DFlash2 snapshot')
    parser.add_argument('--cache', type=Path, default=Path('/models/cache'))
    parser.add_argument('--offline', action='store_true', help='Require cached or mounted snapshots')
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    locks = json.loads((base / 'checkpoint.lock.json').read_text())['models']
    profile = json.loads((base / 'engine-profile.json').read_text())
    args.cache.mkdir(parents=True, exist_ok=True)
    target = resolve_snapshot(locks['target'], args.target, args.cache, args.offline)
    draft = resolve_snapshot(locks['draft'], args.draft, args.cache, args.offline)
    if args.download_only:
        print('Pinned target and DFlash2 snapshots are downloaded and verified.', flush=True)
        return
    env = os.environ.copy()
    env.update(profile['environment'])
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HOME=str(args.cache / 'hf'), XDG_CACHE_HOME=str(args.cache / 'runtime'),
               VLLM_CACHE_ROOT=str(args.cache / 'runtime/vllm'),
               TRITON_CACHE_DIR=str(args.cache / 'runtime/triton'))
    command = engine_command(profile, target, draft)
    print('Starting the ordinary vLLM API server with the qualified Paiton profile.', flush=True)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
