# Reproduction

The Dockerfile inherits the existing qualified public ROCm runtime by immutable digest, sharing its already-split GHCR layers. The parent recipe pins ROCm vLLM and Ubuntu inputs. The runtime contains
Python 3.14.6, Torch 2.11 / ROCm 7.14, vLLM 0.26.1.dev1+g396cd1a43 and
Transformers 5.14. The container installs the plugin with `--no-deps`; repository-wide pip dependency defaults target a different stack and are not a standalone qualification for this model. Use the pinned image. The only vLLM source patch defers an unused ROCm FlashAttention
import; both stock and Paiton receive it. `patches/apply.py` refuses a different
source SHA. No initialized CUDA/HIP process is forced to fork.

From the repository root, with the reviewed artifact directory containing
`minicpm5_awq_decode_gfx1201.so` and its SHA-256 manifest:

```bash
docker buildx build --load \
  --build-context minicpm_overlay=/absolute/path/to/reviewed-artifacts \
  -f models/MiniCPM5-2B/Dockerfile \
  -t ghcr.io/eliovp/paiton-vllm-plugin:minicpm5-2b-w4a16-rdna4-v1.0.0 .
```

The compiled artifact comes from Paiton's compiler IR, not a runtime Torch
extension. It exports a checked C ABI and dynamically links ROCm libraries;
there are no ATen, libtorch or libc10 dependencies. Python integration owns tensor
allocation and registration. Lossless GPU preparation creates a decode layout
while retaining original AWQ tensors for stock prefill. The kernel unpacks and
dequantizes to FP16, then accumulates FP32; this is not native INT4 matrix arithmetic.
The manifest pins geometry, dtype, group size, architecture and artifact checksum.
Artifacts are specific to gfx1201 and this ROCm ABI; do not reuse on untested GPUs.

## API and quality checks

From this model directory, against an already running server:

```bash
python3 benchmark/chat_benchmark.py --model minicpm5-2b --out chat-results.json
python3 benchmark/quality.py --url http://127.0.0.1:8036 --model minicpm5-2b \
  --cases benchmark/quality_cases.json --out quality-results.json \
  --code-image ghcr.io/eliovp/paiton-vllm-plugin:minicpm5-2b-w4a16-rdna4-v1.0.0 \
  --disable-thinking --reasoning-effort ''
```

Generated Python runs in an unprivileged, network-disabled, read-only Docker
sandbox with bounded CPU time and memory. No hosted judge is used. Retain raw
responses and timings, and repeat with `serve-docker.sh --stock`. Do not compare
models by tokens/s alone: the natural-stop suite retains complete answers and
completion times. Fixed-token throughput measures compute separately from quality.

Startup begins before container launch and ends after the first completed useful
response. `/health` timing is reported separately. Download verification,
container build, runtime-cache population and page-cache preparation are separate
costs. Never drop system-wide caches on a shared machine. For isolated cold-weight
trials copy the checkpoint to a task-owned directory and inspect page residency;
`benchmark/weight_cache.py` confines eviction to that private copy. Warm filesystem
results are explicitly labeled. Repeated startup trials do not justify tail
percentiles.

## Protocol checks

```bash
python3 benchmark/protocol.py --model minicpm5-2b --out protocol-results.json
python3 benchmark/chat_benchmark.py --model minicpm5-2b --warmup-repeats 4 --out steady-chat.json
```

The compressed raw records use portable placeholders for local paths. Complete
local commands, snapshots, logs and telemetry remain in the qualification archive.
The public artifact manifest and container metadata pin the exact binary/runtime.
