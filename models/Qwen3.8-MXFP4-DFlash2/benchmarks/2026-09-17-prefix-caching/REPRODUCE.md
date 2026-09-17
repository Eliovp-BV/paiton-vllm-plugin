# Reproduce the published-image APC comparison

This recipe enables the **stock GDN + automatic prefix caching path (arm C)** using the existing, immutable 64K image. It mounts a configuration profile; it does not replace the installed plugin or native libraries. Arms A and B are available as controls.

**The published images do not implement `PAITON_PREFIX_CACHING=1`, and their launchers do not accept `--enable-prefix-caching` as a container argument.** Those conveniences were implemented locally but are not part of this documentation publication. Use the complete profile override below with the published image. Setting only the environment variable on an old image does not enable APC.

The native recurrent-prefill candidate (**arm D**) also needs an unreleased adapter. Its measurements are reported transparently, but it cannot be reproduced from this image and profile alone. No new runtime image is being released with this report.

## 1. Generate a bounded profile

Run these commands from this benchmark directory in a checkout of `Eliovp-BV/paiton-vllm-plugin`:

```bash
python3 reproduction/make_profile.py --arm C --out apc-arm-c.json
```

The standard-library script derives the configuration from the checked-in [64K profile](../../engine-profile-agentic-64k-v1.json). It selects:

| Setting | Value |
|---|---|
| Target and compatible DFlash2 | The image's pinned, hash-verified checkpoints |
| Total context, including output | 65,536 tokens |
| Active sequence limit | 1 |
| Explicit FP8 cache budget | 5 GiB |
| Chunked-prefill token budget | 4,096 |
| Prefix caching | Enabled |
| Mamba cache mode | `align` |
| Native GDN replay, prefill and convolution-prefill registrations | Disabled |
| Other Paiton optimizations and DFlash2 | Retained |
| Default thinking mode | Disabled |

The generated profile sets `PAITON_EXPERIMENTAL_GDN_REPLAY=0`, `PAITON_EXPERIMENTAL_GDN_PREFILL=0` and `PAITON_EXPERIMENTAL_GDN_CONV_PREFILL=0` inside the profile's environment. This is deliberate: the image entrypoint merges profile environment values after Docker's environment. The resulting vLLM arguments contain `--enable-prefix-caching` and `--mamba-cache-mode align` together.

Keep this bounded resource profile for a matched comparison. The report's approximately 150K tests used arm D, an 8 GiB cache and a 160,000-token context setting. They are separate evidence, **not a qualification of this arm C recipe at 150K or of APC at 200K**. The released 200K compact profile's 8 GiB cache budget is insufficient for the tested stock-state APC configuration at a 200K limit.

## 2. Start one server

Use an otherwise idle R9700. Stop the server you own before starting another GPU server. The following command uses the same persistent cache volume as the [published launch instructions](../../LAUNCH-agentic-v1.1.0.md), and exposes only a localhost endpoint:

```bash
docker run -d --name paiton-qwen38-apc-c \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --ipc host \
  -p 127.0.0.1:18981:8000 \
  -v paiton-qwen38-mxfp4-cache:/models/cache \
  -v "$PWD/apc-arm-c.json:/opt/paiton-release/engine-profile.json:ro" \
  ghcr.io/eliovp/paiton-vllm-plugin@sha256:c3ec2528285b484b2e0af7f571f80f1da23c1970210e4d3d09f5d2a909414186
```

Append `--offline` after the image reference only if both pinned snapshots are already cached. Initial startup verifies model files and can compile the stock vLLM fallback's kernels; allow startup and warmup to finish before timing requests. Model revisions and file hashes are in the shared [checkpoint lock](../../checkpoint.lock.json).

Check readiness and retain the startup log:

```bash
curl --fail http://127.0.0.1:18981/health
docker logs paiton-qwen38-apc-c > apc-c-startup.log 2>&1
```

Use OpenAI-compatible base URL `http://127.0.0.1:18981/v1`, model `Qwen3.8-27B-Quark-AWQ-MXFP4`. The profile requests `FULL` graphs; the tested runtime selected `FULL_DECODE_ONLY`. Do not infer full-prefill graph execution from the requested setting.

## 3. Reuse the frozen benchmark inputs

The [evidence archive](raw-evidence.tar.gz) includes the exact synthetic corpora, HTTP clients and archived measurements. Extract it to a new directory:

```bash
mkdir evidence
tar -xzf raw-evidence.tar.gz -C evidence
cd evidence
python3 scripts/apc_benchmark.py check
```

The clients use Python's standard library and a local HTTP endpoint. They do not load GPU libraries or control the server. Run one excluded warmup with its distinct prefix, then the matched measured cases in this order:

```bash
python3 scripts/apc_benchmark.py run --arm C \
  --corpus corpus-warmup-40k.json --cases cold --out new-results/c-warmup-40k
python3 scripts/apc_benchmark.py run --arm C \
  --corpus corpus-40k.json --out new-results/c-40k
python3 scripts/apc_benchmark.py run --arm C \
  --corpus corpus-8k.json --out new-results/c-8k
python3 scripts/decode_benchmark.py --arm C --out new-results/c-decode
python3 scripts/apc_benchmark.py run --arm C \
  --corpus corpus-repeat-40k.json --cases cold,repeat --out new-results/c-repeat-40k
python3 scripts/probe_cached_tools.py --arm C \
  --corpus corpus-40k.json --out new-results/c-cached-tools
```

All commands default to `http://127.0.0.1:18981`; pass `--url` explicitly if using another local port. Each output directory must be new. **Restart the server before repeating a cold run with the same corpus**, or prepare a new prefix using the harness's `prepare` command. A case named `cold` is not proof of a cache miss: inspect the saved prefix-hit counter deltas.

The main seven-case sequence covers cold input, an identical repeat, two growing follow-ups, an edit near the start of the prefix, a branch and return to the original branch. Growing histories contain fixed correct assistant text so all arms receive identical request payloads. The separate tool probe reuses an actual returned tool call and checks a changed tool-result fixture; it is functional evidence rather than a matched performance comparison.

Main requests use `temperature=0`, `top_p=1`, `seed=42`, thinking disabled, streaming and a 192-token output budget. The separate code-generation probe uses a 512-token cap, one excluded warmup and two measured requests. It does not score completed coding tasks. Timing from streamed output chunks is a serving estimate, not a token-level kernel measurement.

The client preserves requests, timestamped stream events, responses, usage and `/metrics` snapshots. Retain the server log over the same interval:

```bash
docker logs paiton-qwen38-apc-c > new-results/c-server.log 2>&1
docker stop --timeout 30 paiton-qwen38-apc-c
```

## 4. Compare with APC off

From the benchmark directory, generate either control profile:

```bash
python3 reproduction/make_profile.py --arm A --out apc-arm-a.json
python3 reproduction/make_profile.py --arm B --out apc-arm-b.json
```

- **A:** compact native GDN replay, APC off, Mamba cache mode `none`.
- **B:** stock GDN fallback, APC off, Mamba cache mode `none`.
- **C:** stock GDN fallback, APC on, Mamba cache mode `align`.

Repeat the Docker launch with the corresponding profile filename and a fresh container name, then run the same frozen corpora with the corresponding `--arm` label and fresh output directories. Run only one server at a time. The client arm label records your intended setup; it does not detect or change backend dispatch. Keep the generated profile, immutable image reference and server logs as configuration evidence.

All three use the same checkpoint revisions, 5 GiB cache budget, context limit, sequence limit and request settings. Equal cache **bytes** do not imply equal usable capacity: the reported stock-state APC capacity is lower than the compact native path. Compare cold and warm TTFT, end-to-end sequence time, observed decode rate and capacity separately. Prefix reuse can greatly improve an interactive session even when the fallback reduces decode performance; it does not improve every workload.
