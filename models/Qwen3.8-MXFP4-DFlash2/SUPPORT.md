# Context limits and coding-agent support

**Updated 17 September 2026.**
The parser correction, 64K context smoke, and OpenCode 1.18.31 read/write smoke
pass. These are functional checks, not comprehensive model-quality or performance
qualification. The published v1.0.0 image and historical benchmark profile remain
unchanged. Separate 64K and 200K images are now published and their immutable
pulls are verified; see the [launch instructions](LAUNCH-agentic-v1.1.0.md).

## Tool calls appearing as ordinary text

The pinned Qwen3.8 chat template emits XML tool calls containing `<function=...>`
and `<parameter=...>` fields. The original v1.0.0 image selects the `hermes` parser,
which expects a different tool-call format. In four API reproductions covering
read/write tools and ordinary/streaming responses, the model emitted valid XML
as content while the API returned no structured `tool_calls` and finished with
`stop`.

This is a server profile mismatch. Those reproductions do not establish a fault
in OpenCode. Select `qwen3_xml` for tool parsing and `qwen3` for reasoning parsing.
The examples below supply those settings through regular vLLM and the existing
Paiton plugin; another inference engine is unnecessary.

With `qwen3_xml`, `qwen3`, and default thinking disabled, the same four read/write
API cases return structured tool calls in ordinary and streaming responses.
Two tool-result roundtrips also pass. A separate streaming request with thinking
explicitly enabled at `low` effort returns reasoning separately and then a valid
tool call. The parser correction therefore does not require permanently disabling
thinking.

A successful structured tool call still requires a complete model response.
An output stopped by `max_tokens` may contain incomplete XML; that is a separate
case from the reproduced parser mismatch. Inspect the API response and finish
reason when diagnosing client behavior.

## Why the original v1.0.0 server stops at 8K

[engine-profile.json](engine-profile.json) explicitly sets both the target's
`--max-model-len` and the DFlash2 `max_model_len` to `8192`. This is a per-request
limit on the combined prompt and generation budget. The cache-capacity figure in
the benchmark is a different measurement; it does not raise that request limit.

The original image entrypoint does not accept arbitrary vLLM arguments. Passing
`--max-model-len` directly after the existing image name therefore does not update
its profile. Both target and drafter limits must change together.

The updated [serve.py](serve.py) supports explicit overrides:

| Option | Behavior |
|---|---|
| `--max-model-len N` | Sets the same positive context limit for target and drafter |
| `--max-num-seqs N` | Sets concurrent sequences from 1 through 8, the native replay limit |
| `--kv-cache-memory-bytes N` | Sets an explicit positive integer byte budget |
| `--tool-call-parser qwen3_xml` | Parses the checkpoint's XML tool calls |
| `--reasoning-parser qwen3` | Separates reasoning from ordinary response content |
| `--disable-thinking` | Sets the default request template option `enable_thinking=false` |

The host launcher's existing image lock still selects the older image. With no
overrides, that launcher retains the historical 8K context, eight
sequences, 5 GiB cache and original parser settings. With any override, it mounts
the updated public `container_entrypoint.py` read-only over the entrypoint in
the existing immutable image. The image's model locks and native runtime remain
in place; no new image, model download, compiler, or build is required when the
locked models are already cached. Startup identifies these overrides as
experimental and prints the effective limits and parser settings.

## Tested 64K coding-agent configuration

From the repository root:

```bash
python3 models/Qwen3.8-MXFP4-DFlash2/serve.py \
  --max-model-len 65536 \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --disable-thinking \
  --detach
```

This command retains eight concurrent sequences and the 5 GiB cache budget.
The configuration passes the functional checks below; it has not received the
historical benchmark suite's performance qualification. A maximum context limit
does not reserve that many tokens for every concurrent request. The 5 GiB run
reported 119,088 cache tokens, or approximately 1.82 times the full 65,536-token
request limit. It does not provide eight full 64K windows simultaneously.
`--max-num-seqs 1` lowers the concurrency ceiling when testing long individual
requests; larger cache budgets require separate memory checks.

At concurrency one, all three synthetic prompts returned the three embedded
markers correctly:

| Actual prompt tokens | Time to first token | Marker retrieval |
|---|---:|---|
| 14,987 | 4.509 s | All three correct |
| 30,977 | 10.277 s | All three correct |
| 62,983 | 24.766 s | All three correct |

Each response contained 57 generated tokens. This is a three-marker synthetic
retrieval and API-latency smoke, not a long-context coding benchmark. The longest
tested prompt was 62,983 tokens; 65,536 remains the combined prompt/output limit.
**64K is the tested deployment setting here, not a claim about the model's or
hardware's maximum context.** Higher limits need their own cache and concurrency
validation.

Add `--dry-run` to inspect the exact Docker arguments. Existing `--cache`,
`--target`, `--draft`, `--offline`, `--name`, and `--port` options still apply.
Run one server on the GPU at a time and choose the container name deliberately.

The separate image uses the versioned
[engine-profile-agentic-64k-v1.json](engine-profile-agentic-64k-v1.json) as its
default `engine-profile.json`. It sets 64K for both target and drafter, eight
sequences, 5 GiB cache, `qwen3_xml`, `qwen3`, and default thinking disabled.
Its profile metadata explicitly records the limited functional validation scope.
The historical [engine profile](engine-profile.json) is retained separately, and
its benchmark figures are not reassigned to the new profile.

## Optional 200K profile: one active request

[engine-profile-agentic-200k-v1.json](engine-profile-agentic-200k-v1.json) sets
200,000 tokens for both target and drafter, an 8 GiB cache and **one active
request**. Additional requests queue. The limit includes the prompt and generated
response together; a 200,000-token prompt leaves no output budget.

The actual single-R9700 test processed a 195,999-token prompt and returned all
three embedded markers in a 57-token response. Server prefill took 125.82 seconds
and client time to first token was 126.48 seconds. The server reported 202,385
cache tokens, approximately 1.01 full 200K requests. Sampled device memory usage
left only about 129 MiB outside allocated memory on this test host. This is a
tight-memory, single-request configuration, not an eight-way 200K server.

Both pinned checkpoints declare 262,144 positions. The practical restriction
here is memory for the model, drafter, working allocations and cache on one
32 GB GPU. The 64K profile keeps a smaller cache budget and more memory headroom;
it can serve several shorter requests, but it cannot keep eight full 64K windows
resident simultaneously. Longer attention also increases prefill time.

This synthetic retrieval test establishes that the measured request works. It
does not establish comprehensive coding/reasoning quality at 200K or guarantee
that another workload will fit with other GPU allocations present. Native
libraries, model weights, checkpoint revisions and quantization are unchanged.
Both new profiles default to thinking disabled, which is an explicit model-mode
choice; clients can enable thinking per request independently of the parser fix.

## Existing image workaround without an entrypoint update

[agentic-profile.json](agentic-profile.json) is an 8K configuration overlay. It
keeps the historical target/draft limits, eight sequences, 5 GiB cache, native
runtime settings, and model verification. It changes only the tool parser,
reasoning parser, and default thinking policy. The original image entrypoint
already reads its profile from the mounted path, so no Python update is needed:

```bash
docker run -d --name paiton-qwen38-agentic \
  --device /dev/kfd --device /dev/dri --group-add video --shm-size 2g \
  -p 127.0.0.1:8000:8000 \
  -v paiton-qwen38-mxfp4-cache:/models/cache \
  -v "$PWD/models/Qwen3.8-MXFP4-DFlash2/agentic-profile.json:/opt/paiton-release/engine-profile.json:ro" \
  ghcr.io/eliovp/paiton-vllm-plugin@sha256:9b2dae214076d35de785e073b31294b033a376b16e6bc1ec1fdada4e54d96c59
```

The original entrypoint retains its historical startup wording; that wording
does not qualify a modified profile. This overlay has separate validation from
the published benchmarks. To experiment with another context limit in the JSON,
change **both** the target argument and the nested speculative configuration's
`max_model_len`. The updated host launcher does this synchronization automatically.

## Thinking, latency, and client expectations

The pinned template enables thinking by default and uses its `xhigh` reasoning
default when no request override is provided. Long reasoning can consume the
output budget before an answer or tool call appears. It should not be confused
with prefill latency or an absence of GPU activity.

`--disable-thinking` changes the server's default template options; it does not
disable reasoning permanently. It can reduce latency for interactive tool use,
with a possible reasoning-quality tradeoff. A request can opt back in:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true,
    "reasoning_effort": "low"
  }
}
```

The preset's thinking policy differs from requests that rely on the checkpoint's
default. Historical benchmark results should not be presented as measurements
of this new coding-agent profile.

## OpenCode and validation limits

The pinned OpenCode **1.18.31** client passed an isolated real-client smoke against
the corrected 64K configuration. It read a marker that was not supplied in its
prompt, used the write tool to create `result.json`, and produced exactly the
expected JSON. This verifies that read/write tools work through that client and
server combination. It is not a broad qualification of every OpenCode tool,
workflow, or long-context repository task.

- CPU launcher checks pass for default-profile parity, synchronized target/draft
  limits, argument validation, parser settings, thinking defaults, and the
  read-only compatibility mount.
- The historical parser mismatch is reproduced in four of four API cases.
- The corrected parser passes four of four ordinary/streaming read/write API
  cases, two tool-result roundtrips, and an explicit-thinking streaming tool call.
- The 64K server starts with the recorded 5 GiB cache, and all three synthetic
  single-request retrieval cases pass. This is not eight-request 64K validation.
- The actual OpenCode 1.18.31 read/write smoke passes with exact output checks.
- Community reports describe different clients, parser settings, thinking
  policies, context budgets, and builds. They do not isolate which component
  failed in this release; use the controlled API/client reproductions instead.

This draft does not change the published image, benchmark evidence, or the
historical [engine profile](engine-profile.json).
