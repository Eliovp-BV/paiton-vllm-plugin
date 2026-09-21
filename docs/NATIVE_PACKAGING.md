# Native execution packaging and qualification

The native CLI reuses the qualified releases' runtime payloads. It does not
rebuild kernels or replace their inference stacks. All seven execution bundles are
published with the v0.3.4 wheel on GitHub Releases. The
[native guide](NATIVE_EXECUTION.md) is the user entry point; managed container
builds remain separate.

## Reused components and boundaries

| Existing component | Responsibility / reuse | Change and compatibility check |
| --- | --- | --- |
| `artifact_manifest.py` | Native ABI, architecture, wave size, TP and file validation | Execution inventory schema 1 composes the existing validator; expected digests and inode/ctime identity enter verification-cache keys. Legacy manifest 1 remains accepted. |
| Platform/general entry points | vLLM discovery and adapter registration | One activation gate; inactive discovery imports no runtime frameworks and does not probe a device. Explicit legacy launchers retain their paths. |
| Current Qwen release payload | Native GDN, RMS, SiLU, prefill and packed matrix execution | Allowlisted native files are copied unchanged into a distinct bundle. No authoritative library is modified. |
| Released Qwen external adapters | Loader integration and compatibility overlays | Vendored with exact file receipts; process-local import hooks relocate paths and verify original installed source hashes. They do not edit vLLM files. |
| MiniCPM model adapter | AWQ loading, lossless repack and native decode dispatch | Existing adapter and artifact reused; shared resolver selects its independent runtime/profile without Qwen overlays. |
| Qwen3-Coder / GPT-OSS model adapters | Existing compiled expert decode, with upstream prefill and attention | Scoped native registration; verified binaries and the separate GPT-OSS Harmony vocabulary are packaged. |
| Qronos / Ornith compiled model adapters | Native model execution and the released Paiton cache contract | Shared preparation produces a separate configuration directory; Qronos links source weights and Ornith reuses the existing lossless resharder. |
| NEO GGUF and vision adapters | Original mixed GGUF, native projector/vision, text and one-image chat | Exact original GGUF/projector identity, released configuration/tokenizer bundle and scoped GGUF worker/loader activation. |
| Existing vLLM/Torch/ROCm environment | Scheduling, remaining operators, upstream compilation, warmup and graphs | Version/ABI checks replace default pip inference-stack pins. The package does not install or rewrite this environment. |
| Existing model checkpoint/cache | Original weights, tokenizer and configuration | Exact pinned identity plus local file fingerprints; weights stay in their existing directory or Hugging Face cache. |

The private compiler and generated implementation source remain outside the
plugin and execution bundles. Approved external adapter source already crosses
the release boundary and is retained with its original notices. Existing
external Torch/Triton integrations remain isolated in the plugin/runtime; no
new compiler dependency or kernel conversion is introduced by native packaging.

## Payload inventory

| Location | Supplied components |
| --- | --- |
| Compatible user environment | Python, vLLM, Torch, HIP/ROCm and the exact framework dependencies listed by the catalogue; standard upstream compilation and graph preparation |
| Lightweight wheel | CLI, typed API, activation/startup hooks, model adapters, pinned catalogue/manifests, Qwen external adapter/overlay sources and receipts, retained notices |
| Qwen native bundle | Released HIP libraries and manifests under `native/`, `r4d.so`, `radiance_mxfp4_fp8.so`, sanitized provenance and prebuilt offline guard |
| MiniCPM native bundle | Existing AWQ decode library and manifest, prebuilt offline guard |
| Qronos native bundle | Released compiled model and W4 LM-head libraries/manifests, compiled serving configuration, offline guard |
| Ornith native bundle | Released compiled model/manifest and serving configuration, offline guard |
| Qwen3-Coder native bundle | Released expert library/manifest and offline guard |
| GPT-OSS native bundle | Released expert library/manifest, pinned Harmony vocabulary and offline guard |
| NEO native bundle | Complete allowlisted release payload: model, GGUF helper and vision libraries; manifests; serving configuration, tokenizer/processor metadata and offline guard |
| Source checkpoint | Original safetensors or selected GGUF/projector and required metadata; source files remain unchanged |
| Paiton runtime cache | Preset preferences, plans, stat-bound verified-weight receipts, native packages, prepared configurations, required Ornith reshards and upstream compilation caches |

Each native preset names its complete pinned payload and existing environment.
The [validation status](NATIVE_EXECUTION.md#validation-status) records which
installed-wheel hardware checks have run. Speculative Qwen/DFlash and Ornith
DFlash serving retain their existing launchers; they are not selected by these
non-speculative native presets. New checkpoint variants and other environments
remain outside this release. Ordinary upstream runtime initialization still
compiles/warmups operations where required; this is not a promise that all
runtime compilation disappears.

## Build and validate locally

Run these commands in a checkout using an isolated packaging environment:

```bash
python -m pip wheel --no-deps --no-build-isolation . --wheel-dir /path/to/wheels
python tools/package_native_execution.py /path/to/qwen38-rocm10-native-20260921 \
  --spec paiton_vllm_plugin/execution/data/qwen38-execution.json --verify
python tools/package_native_execution.py /path/to/minicpm5-awq-native-20260921 \
  --spec paiton_vllm_plugin/execution/data/minicpm5-execution.json --verify
for model in qronos ornith coder gptoss neo; do
  python tools/package_native_execution.py /path/to/${model}-native-20260921 \
    --spec paiton_vllm_plugin/execution/data/${model}-execution.json --verify
done
```

The packaging tool reads an explicit reviewed allowlist, verifies every source
size/hash, generates or checks `execution.json`, rejects extra/unsafe files,
and uses the shared native ABI validator. Omitting `--verify` may create a
missing identical manifest; it never replaces a different existing manifest.
The manifest digest is SHA256 of canonical JSON, not SHA256 of its formatted
file bytes. Do not infer release trust from a manifest downloaded alongside an
arbitrary library: the installed catalogue is the trust root.

Only allowlisted runtime artifacts may be staged from the existing private
release process. Do not copy a compiler directory, entire environment or
container filesystem. `tools/native/offline_guard.c` is ordinary public C
support code, compiled ahead of time for the qualified glibc environment:

```bash
cc -shared -fPIC -O2 -Wl,-z,relro,-z,now tools/native/offline_guard.c \
  -o /path/to/staging/support/offline_guard.so -ldl
```

A rebuilt guard is a new reviewed input: record its actual bytes and source
digest in the spec/catalogue rather than overwriting an already qualified
bundle. End users receive the prebuilt file. No runtime compilation or compiler
checkout is needed for this support component.

Published archives include only regular files named by
the manifest plus `execution.json`; directories and symlinks are not accepted
archive members. The installed catalogue pins each archive's HTTPS URL, size
and SHA256. Keep those pins tied to the exact published bytes.

## Test levels

CPU tests use small local fixtures and require no GPU or checkpoint downloads:

```bash
python -m pytest -q \
  tests/test_native_cli.py tests/test_execution_api.py tests/test_offline.py \
  tests/test_platform_registration.py tests/test_existing_env.py \
  tests/test_qwen38_release.py tests/test_ornith_release.py \
  tests/test_minicpm5_release.py tests/test_qwen38_rocm10_launcher.py \
  tests/test_qwen38_mxfp4_launchers.py tests/test_serve_compatibility.py \
  tests/test_gguf_release_server.py tests/test_gguf_source.py \
  tests/test_gguf_inventory.py tests/test_gguf_detokenizer.py tests/test_chat.py
```

Also run `python -m pytest -q tests/test_serve_compatibility.py` for the short
command, all documented presets, local guide links, retained console entry points
and original per-model Python server flags/activation. The shared API tests cover
remembered checkpoint paths, immutable plans, unchanged source weights, derived
configuration integrity and the original GGUF identification contract.

In the existing qualified Torch environment, also run the CPU model-contract
tests (no GPU or checkpoint is needed):

```bash
python -m unittest tests.test_ornith_model tests.test_qwen38_model \
  tests.test_ornith_mxfp4_loader
```

These check the Ornith state allocation required by its released binary for
both ordinary and DFlash requests, alongside the existing model/weight contracts.

The offline guard fixture compiles only its small public C support file and
tests real libc/Python networking and subprocess inheritance. Archive tests
cover a pinned download fixture, interruption/recovery, corruption, offline
reuse and extraction safety. Worker tests cover plan inheritance, repeated
initialization, late activation, hot-switch refusal and environment mismatch.

Hardware qualification is separate: install the wheel with **normal pip** in
an isolated copy of each existing qualified environment, compare distribution
inventories before/after, remove access to prior Paiton adapters/payloads, and
run from outside the checkout with no compiler mounted. Record the wheel hash,
bundle digest, exact environment, plan/lock, source identities and command
vector. Use both fresh and prepared caches, normal/streaming requests, bounded
concurrency, shutdown and request-time native execution evidence. Compare Qwen
outputs, memory, startup and a bounded latency/throughput sample against the
existing release at identical non-speculative settings.

GPU-free tests do not establish hardware compatibility. Native execution checks
must establish dispatch during a request, including graph replay where used;
loading a library alone is insufficient.
