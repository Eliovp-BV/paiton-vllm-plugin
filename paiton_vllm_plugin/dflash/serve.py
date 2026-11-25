"""Launch the installed vLLM API server with explicit Paiton DFlash options."""
import json
import os
import runpy
import sys
from .configuration import Settings


def prepare_server_args(argv, env):
    args = list(argv)
    settings = Settings.from_environment(env)
    if (settings.verify_cap or settings.verify_trace) and ('--no-async-scheduling' not in args or any(
            x == '--async-scheduling' or x.startswith('--async-scheduling=') for x in args)):
        raise ValueError('Verifier caps require explicit --no-async-scheduling')
    requested = settings.native or settings.rerank or settings.sample_method != 'inherit' or settings.verify_cap or settings.context_norm_rope or settings.context_graph or settings.uniform_graphs
    if not requested:
        return args
    positions = [i for i, value in enumerate(args) if value == '--speculative-config' or value.startswith('--speculative-config=')]
    if len(positions) != 1:
        raise ValueError('Exactly one --speculative-config is required for Paiton DFlash enhancements')
    index = positions[0]
    inline = args[index].startswith('--speculative-config=')
    if not inline and index + 1 == len(args):
        raise ValueError('Missing speculative configuration')
    raw = args[index].split('=', 1)[1] if inline else args[index + 1]
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Speculative configuration must be an object')
    configured = settings.configure_speculation(value)
    if configured == value:
        return args
    updated = json.dumps(configured)
    args[index if inline else index + 1] = '--speculative-config=' + updated if inline else updated
    return args


def _spawn_warmup_sidecar(args):
    """A11 (opt-in, PAITON_START_WARMUP=1): the image serves through this entry point, so the self-request warm-up sidecar is
    spawned here: a detached stdlib-only helper that waits for /health, sends the first-request shapes (C1, C8, a 2K prompt
    and its prefix hit) so their Triton kernels compile before a user arrives, and logs when it is done. Advisory: a
    failure to start it only prints a line; outputs are unchanged either way."""
    import subprocess
    try:
        port = args[args.index('--port') + 1] if '--port' in args else '18982'
        env = dict(os.environ, PAITON_PORT=str(port))
        subprocess.Popen([sys.executable, '-m', 'paiton_vllm_plugin.start_warmup'], env=env, stdin=subprocess.DEVNULL,
                         stdout=sys.stderr, stderr=sys.stderr, start_new_session=True)
        print('[paiton.warmup] self-warm-up sidecar started for port ' + str(port), file=sys.stderr, flush=True)
    except Exception as exc:  # noqa: BLE001
        print('[paiton.warmup] self-warm-up sidecar not started: %r' % (exc,), file=sys.stderr, flush=True)


def _text_only_without_mm_embeds():
    """vLLM 0.32 accepts pre-computed embedding inputs on every KV consumer (VllmConfig._resolve_mm_embedding_inputs),
    and the host and disk KV tiers run the OffloadingConnector as a kv_both consumer. With --language-model-only that
    builds the image and video processor and runs a dummy-input warm-up (about 15 s) in a text-only server; vLLM 0.29
    did neither. Skip the resolver there. The engine core gets the resolved config, so patching this process is enough."""
    try:
        from vllm.config.vllm import VllmConfig
    except Exception:  # noqa: BLE001
        return
    resolve = getattr(VllmConfig, '_resolve_mm_embedding_inputs', None)
    if resolve is None or getattr(resolve, '_paiton_text_only', False):
        return

    def guarded(self):
        model = self.model_config
        mm = model.multimodal_config if model is not None else None
        if mm is not None and mm.language_model_only:
            return None
        return resolve(self)

    guarded._paiton_text_only = True
    VllmConfig._resolve_mm_embedding_inputs = guarded


def main():
    from ..activation import activate
    activate('dflash', os.environ)
    args = prepare_server_args(sys.argv[1:], os.environ)
    settings = Settings.from_environment(os.environ)
    os.environ['_PAITON_DFLASH_SERVER_OPTIONS_V3'] = json.dumps([settings.sample_method, settings.rerank, settings.block_candidates, settings.verify_cap])
    from . import install
    install()
    if os.environ.get('PAITON_START_WARMUP', '0') == '1':
        _spawn_warmup_sidecar(args)
    _text_only_without_mm_embeds()
    sys.argv = ['vllm.entrypoints.openai.api_server', *args]
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


if __name__ == '__main__':
    main()
