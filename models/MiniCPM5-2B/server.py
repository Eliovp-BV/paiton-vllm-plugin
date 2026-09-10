"""Pinned MiniCPM5-2B W4A16 server and verified local checkpoint preparation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / 'checkpoint.lock.json').read_text())


def prepare(offline=False, verify=False):
    from huggingface_hub import snapshot_download
    started = time.perf_counter()
    snapshot = Path(snapshot_download(
        LOCK['model'], revision=LOCK['revision'],
        allow_patterns=[item['name'] for item in LOCK['files']],
        max_workers=2, local_files_only=offline))
    downloaded = time.perf_counter()
    receipt = Path(os.environ.get('XDG_CACHE_HOME', '/models/cache')) / 'verified' / (LOCK['revision'] + '.json')
    try:
        previous = json.loads(receipt.read_text())
    except (OSError, ValueError):
        previous = {}
    current = {}
    for item in LOCK['files']:
        path = snapshot / item['name']
        stat = path.stat()
        if stat.st_size != item['bytes']:
            raise ValueError('Checkpoint size mismatch: ' + item['name'])
        signature = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                     stat.st_dev, stat.st_ino, item['sha256']]
        if verify or previous.get(item['name']) != signature:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != item['sha256']:
                raise ValueError('Checkpoint SHA256 mismatch: ' + item['name'])
        current[item['name']] = signature
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(current))
    temporary.replace(receipt)
    print(json.dumps(dict(model=LOCK['model'], revision=LOCK['revision'],
        preparation_seconds=downloaded-started,
        verification_seconds=time.perf_counter()-downloaded)), flush=True)
    return snapshot


def command(snapshot, stock=False):
    result = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(snapshot), '--served-model-name', os.environ.get('PAITON_SERVED_MODEL', 'minicpm5-2b'),
        '--host', '0.0.0.0', '--port', '8036', '--dtype', 'float16',
        '--max-model-len', '8192', '--max-num-seqs', '2',
        '--max-num-batched-tokens', '512', '--kv-cache-memory-bytes', '1G',
        '--no-enable-prefix-caching', '--generation-config', 'vllm',
        '--reasoning-parser', 'qwen3', '--enable-auto-tool-choice',
        '--tool-call-parser', 'minicpm5', '--safetensors-load-strategy', 'lazy',
        '--seed', '1201', '--attention-backend', 'TRITON_ATTN',
        '--default-chat-template-kwargs', '{"enable_thinking":false}', '--compilation-config',
        json.dumps({'mode':0, 'cudagraph_mode':'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': [1, 2], 'max_cudagraph_capture_size': 2})]
    if not stock:
        result += ['--hf-overrides', json.dumps({'architectures': ['PaitonMiniCPM5AWQForCausalLM']})]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stock', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Rehash all checkpoint files, ignoring verification receipts')
    args = parser.parse_args()
    snapshot = prepare(args.offline, args.verify)
    if args.download_only:
        return
    os.environ.update(HF_HUB_OFFLINE='1', VLLM_ROCM_USE_AITER='0', OMP_NUM_THREADS='1',
                      VLLM_WORKER_MULTIPROC_METHOD='fork',
                      VLLM_PLUGINS='' if args.stock else 'register_paiton_models')
    argv = command(snapshot, args.stock)
    print(json.dumps({'command': argv}), flush=True)
    os.execv(sys.executable, argv)


if __name__ == '__main__':
    main()
