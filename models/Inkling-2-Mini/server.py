"""Pinned Inkling 2 Mini server for one R9700: verified checkpoint, then vLLM on Paiton's native libraries."""
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


# The model's own drafter (DSpark, mtp.safetensors): 7 draft tokens while one request runs, 4 from two on.
DRAFTER = {'method': 'dspark', 'num_speculative_tokens': 7,
           'num_speculative_tokens_per_batch_size': [[1, 1, 7], [2, 4, 4]]}


def command(snapshot, host='0.0.0.0', extra=(), drafter=True):
    # Later arguments win in vLLM's parser, so `extra` (what the user adds after the image name) overrides these.
    return [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(snapshot), '--served-model-name', os.environ.get('PAITON_SERVED_MODEL', 'inkling-2-mini'),
        '--host', host, '--port', '8042', '--quantization', 'inkling2', '--enforce-eager',
        '--max-model-len', MAX_MODEL_LEN, '--max-num-seqs', '4', '--max-num-batched-tokens', '512',
        '--reasoning-parser', 'inkling2', '--enable-auto-tool-choice', '--tool-call-parser', 'inkling2',
        *(['--speculative-config', json.dumps(DRAFTER)] if drafter else []), *extra]


# auto: vLLM sizes the context to the cache memory the card has left (322,736 tokens on a 32 GB R9700).
MAX_MODEL_LEN = 'auto'


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Other arguments go to vLLM unchanged.')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Rehash all checkpoint files, ignoring verification receipts')
    parser.add_argument('--host', default='0.0.0.0',
                        help='address the API listens on (default 0.0.0.0, every interface; 127.0.0.1 keeps it local)')
    parser.add_argument('--api-key', default=None,
                        help='opt-in: require this key on /v1 requests (passed to vLLM as VLLM_API_KEY, never printed)')
    parser.add_argument('--int8-attention', action='store_true',
                        help='opt-in: attention projection weights as 8-bit rows instead of bf16; about 9 %% faster '
                             'decode, a small measurable change of the output distribution (KL +0.003)')
    parser.add_argument('--no-drafter', action='store_true',
                        help='serve without the drafter: one token per step, 0.7 GiB more memory for the cache')
    parser.add_argument('--draft-cut', type=float, metavar='P', default=None,
                        help='opt-in: verify only the drafts whose running probability stays above P (0.4 measured '
                             '6 %% faster for one request); the server then runs without async scheduling')
    args, extra = parser.parse_known_args()
    from paiton_vllm_plugin.activation import activate_model_launcher
    activate_model_launcher(False, os.environ)
    snapshot = prepare(args.offline, args.verify)
    if args.download_only:
        return
    os.environ['HF_HUB_OFFLINE'] = '1'
    if args.draft_cut is not None:
        os.environ['PAITON_INK2_DRAFT_CUT'] = str(args.draft_cut)
        extra = ['--no-async-scheduling', *extra]
    if args.int8_attention:
        os.environ['PAITON_INK2_PROJ'] = 'int8'
    # Defaults of the release; `docker run -e NAME=value` overrides one for a comparison.
    for name, value in dict(PAITON_INKLING2='1', PAITON_INK2_EXPERTS='native', PAITON_INK2_OPS='native',
                            PAITON_INK2_STEP='native', PAITON_INK2_STEP_FLAGS='7', PAITON_INK2_LMHEAD='int8', PAITON_INK2_DRAFT_LOGITS='int8').items():
        os.environ.setdefault(name, value)
    if args.api_key:
        os.environ['VLLM_API_KEY'] = args.api_key
    argv = command(snapshot, args.host, extra, not args.no_drafter)
    print(json.dumps({'command': argv}), flush=True)
    os.execv(sys.executable, argv)


if __name__ == '__main__':
    main()
