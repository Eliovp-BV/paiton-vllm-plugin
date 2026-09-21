"""Use the current Python environment without installing a serving runtime."""
import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
import runpy
import sys


def inspect_environment():
    versions = {}
    for name in ('vllm', 'torch', 'transformers', 'huggingface-hub', 'safetensors'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    missing = [name for name, version in versions.items() if version is None]
    return {
        'python': sys.executable,
        'adapter': __file__,
        'versions': versions,
        'missing': missing,
        'runtime_present': not missing,
        'performance_qualified': False,
        'note': 'Package presence is not model, artifact, GPU or runtime compatibility validation. '
                'Published image speeds are not guaranteed in an existing environment. '
                'Native artifacts and model-specific runtime adapters must be supplied separately.',
    }


from .activation import activate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='Inspect installed package versions without importing GPU frameworks')
    serve = sub.add_parser('serve', help='Run the installed vLLM CLI; pass its normal serve arguments')
    serve.add_argument('--mode', choices=['models', 'dflash', 'legacy'], default='models')
    serve.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == 'doctor':
        report = inspect_environment()
        print(json.dumps(report, indent=2))
        return 0 if report['runtime_present'] else 1
    if importlib.util.find_spec('vllm') is None:
        parser.error('vLLM is absent from this Python environment. Activate your existing vLLM environment first.')
    forwarded = args.args[1:] if args.args[:1] == ['--'] else args.args
    activate(args.mode, os.environ)
    print('Paiton: using the installed vLLM; published image performance is not guaranteed.', file=sys.stderr)
    if args.mode == 'dflash':
        # DFlash proposal configuration must be prepared before worker startup.
        from .dflash.serve import main as dflash_main
        sys.argv = ['paiton-dflash-serve', *forwarded]
        dflash_main()
    else:
        sys.argv = ['vllm', 'serve', *forwarded]
        runpy.run_module('vllm.entrypoints.cli.main', run_name='__main__')
    return 0


if __name__ == '__main__':
    sys.exit(main())
