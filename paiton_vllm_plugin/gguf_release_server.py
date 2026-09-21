"""Start the pinned native NEO GGUF model through vLLM's serving process."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath

from .qwen38_release_server import (
    ReleaseModelError,
    _strict_overlay_file,
    _validate_runtime_environment,
    _verify_declared_file,
)


def verify_payload(directory: Path) -> dict:
    """Check the exact allowlisted payload before loading any native artifact."""
    manifest_path = directory / 'paiton-release.manifest.json'
    _strict_overlay_file(manifest_path, 'GGUF release manifest')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('bundle_version') != 1:
        raise ReleaseModelError('Unsupported GGUF release bundle version')
    seen = set()
    for record in manifest['files']:
        relative = PurePosixPath(record['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:
            raise ReleaseModelError('Invalid or duplicate GGUF release path')
        seen.add(str(relative))
        path = directory / relative
        for parent in [path, *path.parents]:
            if parent == directory:
                break
            if parent.is_symlink():
                raise ReleaseModelError('GGUF release payload must not contain symlinks')
        _strict_overlay_file(path, 'GGUF release payload')
        _verify_declared_file(path, dict(file=relative.name,
            size_bytes=record['size_bytes'], sha256=record['sha256']))
    expected = seen | {'paiton-release.manifest.json', 'paiton-release.spdx.json', 'SHA256SUMS'}
    actual = {str(path.relative_to(directory)) for path in directory.rglob('*') if path.is_file()}
    if actual != expected:
        raise ReleaseModelError('GGUF release payload inventory mismatch')
    config = json.loads((directory/'config.json').read_text())
    architecture = 'PaitonQwen38GGUFForConditionalGeneration' if config.get('paiton_qwen38_contract', {}).get('multimodal') else 'PaitonQwen38GGUFForCausalLM'
    if config.get('architectures') != [architecture]:
        raise ReleaseModelError('Release does not declare the native Paiton GGUF model')
    return config['paiton_qwen38_contract']


def serving_command(directory: Path, host: str, port: int, *, multimodal: bool = False,
                    prefill_chunk_tokens: int = 512) -> list[str]:
    if prefill_chunk_tokens not in (512, 1024, 2048):
        raise ReleaseModelError('Unsupported GGUF prefill chunk size')
    command = ['vllm', 'serve', str(directory), '--served-model-name', 'qwen38-neo',
        '--dtype', 'bfloat16', '--tensor-parallel-size', '1',
        '--max-model-len', '8192', '--max-num-batched-tokens', str(prefill_chunk_tokens),
        '--max-num-seqs', '1', '--block-size', '16', '--kv-cache-memory-bytes', '2G',
        '--load-format', 'paiton_gguf', '--enforce-eager', '--no-enable-prefix-caching',
        '--reasoning-parser', 'qwen3', '--enable-auto-tool-choice',
        '--tool-call-parser', 'qwen3_coder', '--host', host, '--port', str(port)]
    if multimodal:
        command += ['--worker-cls', 'paiton_vllm_plugin.gguf_worker.PaitonGGUFWorker',
                    '--limit-mm-per-prompt', json.dumps({'image': 1, 'video': 0}),
                    '--mm-processor-kwargs', json.dumps({'max_pixels': 1048576})]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('/opt/paiton/neo-model'))
    parser.add_argument('--cache-dir', type=Path, default=Path('/models/cache'))
    parser.add_argument('--checkpoint', type=Path, default=os.getenv('PAITON_GGUF_CHECKPOINT'))
    parser.add_argument('--projector', type=Path, default=os.getenv('PAITON_GGUF_PROJECTOR'))
    parser.add_argument('--prefill-chunk-tokens', type=int, choices=(512, 1024, 2048),
                        default=int(os.getenv('PAITON_GGUF_PREFILL_TOKENS', '512')))
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be in [1,65535]')
    directory = args.model_dir.resolve(strict=True)
    contract = verify_payload(directory)
    if args.prefill_chunk_tokens > int(contract['max_num_batched_tokens']):
        raise ReleaseModelError('Requested prefill chunk exceeds the native artifact capacity')
    _validate_runtime_environment()
    if args.checkpoint is None:
        from huggingface_hub import hf_hub_download
        source = contract['checkpoint']
        args.checkpoint = Path(hf_hub_download(repo_id=source['repo_id'],
            revision=source['revision'], filename=source['selected']['file'],
            cache_dir=str(args.cache_dir)))
    if contract.get('multimodal'):
        vision = json.loads((directory/'vision-runtime.json').read_text())
        if args.projector is None:
            from huggingface_hub import hf_hub_download
            source = contract['checkpoint']
            args.projector = Path(hf_hub_download(repo_id=source['repo_id'],
                revision=source['revision'], filename=vision['checkpoint']['file'],
                cache_dir=str(args.cache_dir)))
        os.environ['PAITON_GGUF_PROJECTOR'] = str(args.projector.resolve(strict=True))
    checkpoint = args.checkpoint.resolve(strict=True)
    # The bounded GGUF loader checks the full checkpoint SHA256 before copying
    # any tensor to the device. Resolving Hub links lets it reuse the blob itself.
    os.environ.update(PAITON_GGUF_CHECKPOINT=str(checkpoint),
        VLLM_USE_PAITON_PLATFORM='1', PAITON_GGUF_SAFE_DETOKENIZER='1',
        PAITON_QWEN38_W4_LM_HEAD='0', PAITON_PREFILL_ATTENTION_AOT='0',
        PAITON_DECODE_ATTENTION_AOT='0',
        PAITON_QWEN38_SERIALIZED_EXTERNAL_GRAPH_CAPTURE='0')
    from .activation import activate
    activate('legacy', os.environ)
    command = serving_command(directory, args.host, args.port, multimodal=contract.get('multimodal', False),
                              prefill_chunk_tokens=args.prefill_chunk_tokens)
    print('Starting native Paiton GGUF execution through vLLM; checkpoint:', checkpoint, flush=True)
    import sys
    os.execv(sys.executable, [sys.executable, '-m', 'vllm.entrypoints.cli.main', *command[1:]])


if __name__ == '__main__':
    main()
