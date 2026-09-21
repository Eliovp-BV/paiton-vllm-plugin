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


def main():
    from ..activation import activate
    activate('dflash', os.environ)
    args = prepare_server_args(sys.argv[1:], os.environ)
    settings = Settings.from_environment(os.environ)
    os.environ['_PAITON_DFLASH_SERVER_OPTIONS_V3'] = json.dumps([settings.sample_method, settings.rerank, settings.block_candidates, settings.verify_cap])
    from . import install
    install()
    sys.argv = ['vllm.entrypoints.openai.api_server', *args]
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


if __name__ == '__main__':
    main()
