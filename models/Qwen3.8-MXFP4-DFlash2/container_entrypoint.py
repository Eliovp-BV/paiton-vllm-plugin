"""Pinned model download and verification before ordinary vLLM serving."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


INTEGER_ENGINE_OVERRIDE_FLAGS = (
    ('max_model_len', '--max-model-len'),
    ('max_num_seqs', '--max-num-seqs'),
    ('kv_cache_memory_bytes', '--kv-cache-memory-bytes'),
)
ENGINE_OVERRIDE_FLAGS = INTEGER_ENGINE_OVERRIDE_FLAGS + (
    ('tool_call_parser', '--tool-call-parser'),
    ('reasoning_parser', '--reasoning-parser'),
)
PARSER_CHOICES = {'tool_call_parser': ('hermes', 'qwen3_xml'),
                  'reasoning_parser': ('qwen3',)}


def positive_integer(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('must be a positive integer') from error
    if result <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def supported_sequence_count(value):
    result = positive_integer(value)
    if result > 8:
        raise argparse.ArgumentTypeError('must be between 1 and 8 for the native replay runtime')
    return result


def add_engine_override_arguments(parser):
    parser.add_argument('--max-model-len', type=positive_integer,
                        help='Experimental context limit for both target and drafter; default is the image profile')
    parser.add_argument('--max-num-seqs', type=supported_sequence_count,
                        help='Experimental concurrent sequence limit, 1–8; default is the image profile')
    parser.add_argument('--kv-cache-memory-bytes', type=positive_integer,
                        help='Experimental KV cache budget in positive integer bytes; default is the image profile')
    parser.add_argument('--tool-call-parser', choices=PARSER_CHOICES['tool_call_parser'],
                        help='Experimental tool parser override; default is the release profile')
    parser.add_argument('--reasoning-parser', choices=PARSER_CHOICES['reasoning_parser'],
                        help='Experimental reasoning parser override; default is the image profile')
    parser.add_argument('--disable-thinking', action='store_true',
                        help='Default enable_thinking to false; individual requests may override it')


def engine_overrides(args):
    overrides = {name: getattr(args, name) for name, _ in ENGINE_OVERRIDE_FLAGS
                 if getattr(args, name, None) is not None}
    if getattr(args, 'disable_thinking', False):
        overrides['disable_thinking'] = True
    return overrides


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


def engine_command(profile, target, draft, *, max_model_len=None,
                   max_num_seqs=None, kv_cache_memory_bytes=None,
                   tool_call_parser=None, reasoning_parser=None, disable_thinking=False):
    args = list(profile['arguments'])
    for flag in ('--model', '--tokenizer'):
        args[args.index(flag) + 1] = str(target)
    index = args.index('--speculative-config') + 1
    spec = json.loads(args[index]); spec['model'] = str(draft)
    overrides = {'max_model_len': max_model_len, 'max_num_seqs': max_num_seqs,
                 'kv_cache_memory_bytes': kv_cache_memory_bytes,
                 'tool_call_parser': tool_call_parser, 'reasoning_parser': reasoning_parser}
    for name, flag in ENGINE_OVERRIDE_FLAGS:
        value = overrides[name]
        if value is not None:
            if name in PARSER_CHOICES:
                if value not in PARSER_CHOICES[name]:
                    raise ValueError(f'{flag} must be one of {PARSER_CHOICES[name]}')
            else:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f'{flag} must be a positive integer')
                if name == 'max_num_seqs' and value > 8:
                    raise ValueError('--max-num-seqs must be between 1 and 8 for the native replay runtime')
            if flag in args:
                args[args.index(flag) + 1] = str(value)
            else:
                args += [flag, str(value)]
    if max_model_len is not None:
        spec['max_model_len'] = max_model_len
    args[index] = json.dumps(spec)
    if not isinstance(disable_thinking, bool):
        raise ValueError('--disable-thinking must be boolean')
    if disable_thinking:
        flag = '--default-chat-template-kwargs'
        defaults = json.loads(args[args.index(flag) + 1]) if flag in args else {}
        if not isinstance(defaults, dict):
            raise ValueError('Default chat template kwargs must be a JSON object')
        defaults['enable_thinking'] = False
        if flag in args:
            args[args.index(flag) + 1] = json.dumps(defaults)
        else:
            args += [flag, json.dumps(defaults)]
    return [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', *args]


def effective_engine_settings(command):
    spec = json.loads(command[command.index('--speculative-config') + 1])
    effective = {name: int(command[command.index(flag) + 1])
                 for name, flag in INTEGER_ENGINE_OVERRIDE_FLAGS}
    for name in PARSER_CHOICES:
        flag = '--' + name.replace('_', '-')
        effective[name] = command[command.index(flag) + 1] if flag in command else None
    flag = '--default-chat-template-kwargs'
    effective['default_chat_template_kwargs'] = (
        json.loads(command[command.index(flag) + 1]) if flag in command else None)
    effective['draft_max_model_len'] = spec['max_model_len']
    return effective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, help='Existing pinned target snapshot')
    parser.add_argument('--draft', type=Path, help='Existing pinned DFlash2 snapshot')
    parser.add_argument('--cache', type=Path, default=Path('/models/cache'))
    parser.add_argument('--offline', action='store_true', help='Require cached or mounted snapshots')
    parser.add_argument('--download-only', action='store_true')
    add_engine_override_arguments(parser)
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
    overrides = engine_overrides(args)
    command = engine_command(profile, target, draft, **overrides)
    settings = effective_engine_settings(command)
    settings['profile'] = profile.get('profile', 'engine-profile.json')
    if profile.get('validation_scope'):
        settings['validation_scope'] = profile['validation_scope']
    print('Starting the ordinary vLLM API server with Paiton profile settings: '
          + json.dumps(settings, sort_keys=True), flush=True)
    if overrides:
        print('EXPERIMENTAL Paiton engine overrides (not a qualified profile): '
              + json.dumps(overrides, sort_keys=True), flush=True)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
