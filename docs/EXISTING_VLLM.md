# Use your existing vLLM environment

For verified checkpoint resolution, native bundles, offline preparation and
lockfiles, use the newer [native CLI](NATIVE_EXECUTION.md). This page describes
the retained explicit compatibility launcher; it does not provide that resolver.

Activate the supported environment in which vLLM already works, then install:

```bash
python -m pip install https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/v0.3.4/paiton_vllm_plugin-0.3.4-py3-none-any.whl
paiton-vllm doctor
```

The adapter does not install, upgrade or downgrade vLLM, Torch, Transformers or
ROCm. For offline installation with setuptools and wheel already available, use
`python -m pip install --no-build-isolation --no-deps .`.

Installing the adapter leaves ordinary vLLM execution unchanged. Activate Paiton
for a specific invocation using the installed vLLM CLI and its normal arguments:

```bash
paiton-vllm serve -- /path/to/prepared-model [your usual vllm serve options]
```

Choose a prepared Paiton model and its native artifacts using that model's guide.
Installing Python code does not download model weights or native libraries, nor
does it turn an arbitrary upstream checkpoint into a compiled Paiton model.
Upstream architectures continue to use upstream implementations.

The default `models` mode registers Paiton architectures and its GGUF loader while
keeping the upstream platform. Models that require Paiton's platform use
`paiton-vllm serve --mode legacy -- ...`. Existing explicit release launchers keep
their activation behavior. For direct vLLM invocations, set
`PAITON_PLUGIN_MODE=models` (or `legacy`) and, if you use a `VLLM_PLUGINS` allowlist,
include `register_paiton_models` (and `paiton_platform` for legacy mode).
`PAITON_PLUGIN_MODE=off` disables Paiton's entry points.

The optional `dflash` mode delegates to `paiton-dflash-serve`; that interface takes
API-server arguments, including `--model`, rather than a positional model:

```bash
paiton-vllm serve --mode dflash -- --model /path/to/target [API-server options]
```

## Compatibility and performance

The goal is version-independent installation, with compatibility checked per
model and feature. There is no package-wide vLLM version pin. This does **not**
promise support for every old release, GPU or combination of libraries. A model
adapter can depend on APIs absent from an older release. Native libraries still
require matching GPU architecture, ABI, layouts and arithmetic contracts.

`doctor` reports installed versions without importing GPU frameworks. Successful
package discovery is not proof that a model can load. Run an actual model smoke
test before relying on an existing environment. We do not guarantee the published
container's speeds in a different environment; use the supplied image to reproduce
its benchmark configuration.

In particular, the **154.42 tok/s Qwen3.8 MXFP4 + DFlash2 image is not reproduced by
`pip install .` alone**. It includes external runtime adapters, native libraries,
and source-guarded DFlash hooks. Those checks remain enforced. Installing the adapter does
not broaden their qualification or silently apply them to another vLLM source.

## Initial validation

- Existing vLLM `0.26.1.dev1+g396cd1a43.rocm714`, Torch `2.11.0+rocm7.14.0`,
  Python 3.14: installation and model registration passed without replacing
  dependencies. Native MiniCPM5 inference passed six requests in both the released
  adapter and this candidate. Messages, finish reasons and token counts matched
  exactly, including arithmetic, factual and structured JSON cases repeated twice.
- Existing vLLM `0.29.0`, Torch `2.12.0+rocm10.0.0`, Python 3.12: installation
  and model registration passed without replacing dependencies.

These used disposable copies of existing prepared environments, which retain
model-specific dependencies and fixes. They are not pristine upstream installs,
a general quality evaluation, or a cross-version performance benchmark. No new
Qwen3.8 performance result is claimed.
