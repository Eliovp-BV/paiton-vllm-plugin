# Model and quantization selection

Selected: **GPT-OSS-20B, original MXFP4**. The scope is a popular, reasonably
recent non-Qwen model around 20B total parameters on one 32 GB R9700. MoE active
parameter counts do not replace total checkpoint or VRAM accounting.

| Rank | Candidate | Parameters / license | Assessment |
|---|---|---|---|
| 1 | [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee) | About 21B total, 3.6B active; Apache-2.0 | Best qualified responsiveness and memory balance; original low-bit experts, reasoning and native tools. |
| 2 | [Gemma 4 12B IT](https://huggingface.co/google/gemma-4-12B-it/tree/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7) | About 12B dense; Apache-2.0 | Newer, strong quality challenger. Qualified BF16 eager serving is substantially slower and uses more VRAM. |
| 3 | [Gemma 4 E4B IT](https://huggingface.co/google/gemma-4-E4B-it/tree/ee0ef6023621cff504d758262d4e04895a5af4a2) | About 8B including PLE, 4.5B effective; Apache-2.0 | Popular smaller alternative; “E4B” is not its total stored parameter count. Metadata screened, not GPU benchmarked. |
| 4 | [Ministral 3 14B Instruct 2512](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512/tree/29439f81c2be264d8d393273f99e7db9c0961120) | 14B class, dense; Apache-2.0 | Current general assistant with an FP8 checkpoint; metadata screened, not GPU benchmarked. |

GPT-OSS was [released on August 5, 2025](https://openai.com/index/introducing-gpt-oss/),
about 13 months before qualification. Its current usage and qualified serving
balance justify its selection here. The comparison includes newer Gemma 4 models;
the Gemma 4 12B repository was created in May 2026. Freshness is one selection
factor alongside useful quality, latency, memory and deployment support.

Hub download-count snapshots on 2026-09-09 were approximately 6.55M, 3.09M,
4.81M and 0.35M respectively. These indicate current activity, not unique users
or measured quality. No Qwen model was qualified for this release.

The initial fixed ten-task screen scored GPT-OSS stock and Paiton 9/10, and
Gemma 4 12B with thinking disabled 10/10. This tiny set cannot establish general
model superiority. GPT-OSS can emit a malformed `final code` channel on the
balanced-parentheses coding prompt: vLLM exposes no final answer and llama.cpp
returns a parser error. This failure is retained rather than counted as a pass.
Gemma's low-effort setting enables thinking; its initial coding failure exhausted
the 1,024-token budget. Explicit non-thinking mode passed that case.

Initial four-request warm screens (512 input / 256 forced output tokens,
concurrency one) measured approximately 4.83 s for stock vLLM GPT-OSS, 2.20 s for
Paiton GPT-OSS, and 13.12 s for Gemma BF16 eager. These are selection screens,
not the final repeated release benchmark. Different tokenizers and thinking
modes prevent interpreting raw cross-model tokens/s as a quality ranking.
Gemma used 22.73 GiB for weights versus GPT-OSS's 14.16 GiB padded model
allocation. Its original 23.9 GB single safetensors file exceeded this host's
mmap feasibility; lossless 2 GB resharding preserved the original tensor bytes.
The first Gemma graph setup did not reach serving within a six-minute window.

## Why original MXFP4

GPT-OSS stores its expert weights as E2M1 values with E8M0 scales, one scale per
32 values. It has 24 layers, 32 experts per layer and top-4 routing. All experts
remain resident. Non-expert weights remain BF16; expert biases are promoted by
the stock loader. The checkpoint contains 13,761,264,768 tensor bytes, including
3,608,919,168 BF16 bytes. Padding, workspaces, graphs and KV cache add to that.

The qualified stock path uses OAI Triton MXFP4 experts on gfx1201. Paiton reads
those same packed weights, uses FP32 vector accumulation, and preserves the
BF16 expert activation and weighted-partial rounding boundaries. MXFP4 storage
is **not** a claim of native FP4 arithmetic. Neither path is an ordinary dense
AWQ linear kernel. KV-cache precision is separately fixed to BF16.

[Intel's INT4 AutoRound checkpoint](https://huggingface.co/Intel/gpt-oss-20b-int4-AutoRound/tree/e7ab42304caa6aa89a13aa97721d5a14150b1283)
uses symmetric INT4/G128, 512 calibration samples and 1,000 optimization
iterations, with `auto_round:auto_gptq` packing. It is derived from released
MXFP4 weights. Its card excludes vLLM, and a probe of the qualified vLLM build
rejects `quant_method=auto-round`. It is not a drop-in RDNA4 INT4 alternative.
Renaming that field would not establish compatible expert loading or execution.
Requantizing MXFP4 would add error; it cannot recover original BF16 weights.

[AMD's Quark FP8 checkpoint](https://huggingface.co/amd/gpt-oss-20b-WFP8-AFP8-KVFP8/tree/56ef2e7347c1dea3aac0ec875975e58f04755868)
changes weights, activations and KV precision. It was inspected but not selected
or GPU benchmarked. AWQ/GPTQ, Quark and compressed-tensors labels alone do not
establish biased GPT-OSS expert support on gfx1201. CUDA or CDNA support is not
substituted for an actual RDNA4 qualification.

A separate llama.cpp reference repacks the original MXFP4 values and scales into
GGUF, preserving BF16 non-expert weights, using conversion source revision
`434ddbbc0e30522e897670681e503b797c12b7c1`. It performs no INT4 calibration or
requantization. Its runtime results are reported separately from compiler gains.

GPT-OSS advertises 131,072 tokens; the Gemma and Ministral alternatives advertise
131,072 or 262,144 depending on variant. These are architecture limits, not this
package's tested context claims. GPT-OSS's alternating full/128-token sliding
attention, attention sinks, Harmony format, routing and positional encoding stay
with the stock implementation. See the [upstream reference](https://github.com/openai/gpt-oss).
