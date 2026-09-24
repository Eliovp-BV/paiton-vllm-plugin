# Qwen3.8 prefill update and n-gram co-drafting — 2026-09-24

R9700, 300 W; vLLM 0.29 / ROCm 10; 65,536 context; maximum eight sequences; APC off; thinking off; n-gram co-drafting off. Temperature 0.7, top-p 0.95, top-k 20, seed 42. BetterBench 0.6.0 quick. Control: a fresh pull of the 20 September image. Candidate: the 24 September image contents. Each arm ran twice in fresh processes, interleaved (control, candidate, control, candidate); the tables show the mean of the two runs, which agree within 0.2 tok/s except where noted.

Images: control `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-65k-20260920-r2` (`sha256:791c09ec9662…`); measured candidate, the local build `paiton-qwen38-local:65k-20260924-fu-r1`; published package **`ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-65k-20260924-r3`** (`sha256:c2511888b76a…`). The published image differs from the measured build only in the two n-gram co-drafting adapter files, which are inactive unless `PAITON_NGRAM_CODRAFT=1`; a no-op adapter arm was hash-identical to the control at 4K and 197K. The gated n-gram measurements used the published adapter version.

**Prefill, input tok/s.** The headline of this release: faster at every depth.

| Nominal prefill depth | 20 September image (fresh run) | 24 September package | Change |
|---:|---:|---:|---:|
| 2,000 | 3,499 | **3,687** | +5.4% |
| 8,000 | 3,699 | **3,829** | +3.5% |
| 16,000 | 3,704 | **3,871** | +4.5% |
| 32,000 | 3,591 | **3,753** | +4.5% |
| 64,000 | 3,291 | **3,457** | +5.0% |

**Decode, single stream.**

| Category | Update p50 ms, 20 Sept | Update p50 ms, 24 Sept | TTFT p50 ms, 20 Sept | TTFT p50 ms, 24 Sept | tok/s, 20 Sept | tok/s, 24 Sept | Change |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat | 33.3 | 33.1 | 91.9 | 91.7 | 121.3 | **121.9** | +0.5% |
| code | 33.5 | 33.3 | 89.5 | 88.9 | 180.0 | **181.1** | +0.6% |
| file edit | 33.4 | 33.3 | 92.0 | 91.6 | 180.1 | **181.0** | +0.5% |
| json | 33.5 | 33.3 | 89.7 | 89.0 | 217.6 | **218.8** | +0.5% |
| math | 33.5 | 33.4 | 89.3 | 88.7 | 183.9 | **184.8** | +0.5% |
| prose | 33.4 | 33.3 | 53.4 | 53.4 | 77.0 | **79.4** | +3.2% |
| reasoning | 33.5 | 33.3 | 89.4 | 89.0 | 117.9 | **118.5** | +0.5% |
| summarization | 33.4 | 33.2 | 92.6 | 92.4 | 138.6 | **139.3** | +0.6% |

Weighted decode: **153.64 → 154.78 tok/s (+0.7%)**.

**Concurrency, aggregate generated tok/s over each complete 48-request workload.**

| Concurrent requests | 20 September image (fresh run) | 24 September package | Change |
|---:|---:|---:|---:|
| 1 | 122.1 | **122.8** | +0.6% |
| 2 | 205.7 | **207.0** | +0.7% |
| 4 | 311.0 | **307.1** | −1.2% |
| 8 | 421.6 | **422.9** | +0.3% |

The control's own two runs at concurrency four read 313.9 and 308.1 tok/s; the candidate's −1.2% sits inside that spread.

**Real use, 200K chat profile with prefix caching.** A 35-turn agentic coding session growing from 32.7K to 197.2K tokens plus a 123.6K-token cache miss, A/B/A/B in fresh processes: session time to first token 211.6 → 190.8 s and 211.7 → 188.5 s (−9.8% / −11.0%); per-request TTFT ratio 0.897; peak VRAM equal at 31.77 GiB. (This session run had n-gram co-drafting on; it does not affect TTFT.)

**What changed in the image.** Three exact changes to the prefill path, each bitwise-equal to the previous kernel standalone and showing zero differing elements in in-model shadow audits: long-prefill attention with 16-key tiles (attention 6.6% faster in serving), the GDN gate read in place instead of copied, and a GDN chunk-scan kernel that fills the GPU in one round (together the GDN core is about 30% faster). The prefill GEMM, two thirds to three quarters of prefill time, runs at the card's 300 W limit, which is why the gains are 3.5–5.4% rather than more.

Update p50 is the median streamed-update gap, not per-token latency or TTFT.
Sampled output content and accepted-token work can differ; these are serving-throughput measurements, not identical-output timing.
Nominal prefill depths correspond to median actual prompt lengths 1516.5, 5894.5, 11802, 23549.5 and 47016.5.

Every run: 40/40 decode, 192/192 concurrency and 40/40 prefill scored requests (plus the fixed warmups). All twelve greedy control prompts match the control before timing, and each arm repeats its twelve greedy outputs after the benchmark. No unhandled serving errors. Every run was in the same decode timing mode (C1 forward-time median 28.3–28.5 ms), so the arms are comparable.

Decode has five scored requests per category after one warmup. Prefill has eight scored requests per depth after two warmups. Category values are means of two complete runs, not selected across repeats.

## N-gram co-drafting (opt-in)

The 24 September image includes an optional second drafter in front of DFlash2.
When the last few generated tokens already occurred earlier in the prompt or the
output, the matcher proposes the tokens that followed last time, so copy-heavy
generations such as file rewrites, code echoed back into an edit, or repeated
structure accept more tokens per step. The proposals enter the existing rejection
sampler as one-hot draft rows, so the target distribution is unchanged; when every
request in a batch has a match, the DFlash2 draft forward is skipped.

It is **off by default**. Enable it per launch with `PAITON_NGRAM_CODRAFT=1`; the
launcher forwards the variable when it is set on the host:

```bash
PAITON_NGRAM_CODRAFT=1 bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10-65k.sh --profile chat
```

Measured with the published adapter, A/B/A/B in fresh processes:

| Workload | Off | `PAITON_NGRAM_CODRAFT=1` | Change |
|---|---:|---:|---:|
| 35-turn agentic coding session, 200K chat profile, decode tok/s | 90.8 | 115.4 / 115.7 | **+27%** |
| Accepted tokens per step in that session | 3.0 | 3.6 | |
| 64K-context file rewrite, tok/s¹ | 129 | 167 | +29% |
| 128K-context file rewrite, tok/s¹ | 125 | 158 | +26% |
| BetterBench weighted decode, 65K profile | 153.9 | 153.3 | −0.4% |
| BetterBench file edit / prose / json decode tok/s | 180.1 / 78.3 / 217.7 | 176.2 / 77.4 / 217.5 | −2.2% / −1.2% / −0.1% |
| BetterBench C1 / C2 / C4 / C8 aggregate tok/s | 121.8 / 206.1 / 319.0 / 424.7 | 122.0 / 206.1 / 307.8 / 421.6 | +0.2% / 0.0% / −3.5% / −0.7% |

¹ Measured with an earlier gate version of the adapter; the shipped version was not
re-measured on this workload.

Short prompts with little to copy gain nothing and pay a small bookkeeping cost.
That is why the mode ships off and why the benchmark tables below were measured
with it off. The control's own concurrency-four value ranged 306–319 tok/s across
runs. Session time to first token is unchanged by the mode.

What to expect from the output: sampling still draws from the target distribution,
and greedy decoding still returns the argmax chain in exact arithmetic. A different
acceptance pattern changes which verify row computes a position, so output bits can
differ at near-ties. The two divergences found were word choices inside generated
comments with the top two candidates 0.25 nats apart, and the released image shows
the same class of divergence between its own fresh processes. The twelve greedy
control prompts matched with the mode on, and all 35 session turns produced valid
tool calls with the same tool names. Structured-output workloads that must not change
wording should leave the mode off. `PAITON_NGRAM_CODRAFT_HOT_MATCH` (default 16, the
match length that keeps a request on the drafting path) trades session gain against
the small cost on non-copying traffic.

[Machine-readable results](numbers.json) · [Model page](../../README.md)
