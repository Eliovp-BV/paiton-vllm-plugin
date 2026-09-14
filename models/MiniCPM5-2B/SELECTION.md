# Small-model qualification — 10 September 2026 UTC

Ranked shortlist after bounded loading, quality and quantization experiments. MiniCPM5-2B is selected; the official W4 derivative is the release representation. See BENCHMARKS.md for targets declared after the first stock measurement and before compiler work.
Date cross-check: host UTC clock and current HTTP Date headers retained in the local qualification evidence, with [UTC clock](https://time.is/UTC).

The 30-day window is August 11–September 10; the extended 60-day window is July 12–September 10. Only two credible, useful recent-base candidates were established: MiniCPM5-2B and LFM2.5-2.6B. Rivet-1B is an August from-scratch project, but its author says the instruction checkpoint is not released, and independent adoption is unestablished; it is excluded. Two clearly marked older-base alternatives below provide context rather than pretending four recent bases exist.

| Rank / exact checkpoint | Pinned revision | Total/active dense parameters | BF16 weights, bytes | Hub monthly downloads | Cumulative likes |
|---|---|---:|---:|---:|---:|
| [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | `bca24102e83b4a03a5d7e9d281e9a6790cc6e3a3` | 2,516,756,480 | 5,033,557,096 | 2,879 | 1,008 |
| [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) | `654f9463ce32b05d0429d76fe1f580b27d4c1ac0` | 2,697,198,592 | 5,394,427,456 | 109,992 | 749 |
| [ibm-granite/granite-4.2-3b](https://huggingface.co/ibm-granite/granite-4.2-3b) | `e459acceac81e5fe67c07d9cfc72329a332e7eb1` | 3,659,737,600 | 7,319,517,120 | 26,634 | 85 |
| [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | `87179e5c1f455ef22e6223592d2d61351b525bfc` | 1,080,632,832 | 2,161,290,912 | 669,403 | 1,119 |

Counters are exact-checkpoint snapshots collected September 10, retained with timestamps in evidence. Monthly downloads count qualifying HTTP file requests, including HEAD; they are not unique users or complete installations. Likes are cumulative and are not compared numerically against monthly downloads. No historical growth series is available and none is claimed. [Counting method](https://huggingface.co/docs/hub/models-download-stats).

1. **MiniCPM5-2B:** new base weights uploaded September 5, public family/instruction announcement September 7 (repository created earlier, August 27 for base and September 6 for instruction). [Base history](https://huggingface.co/openbmb/MiniCPM5-2B-Base/commits/main), [announcement](https://github.com/OpenBMB/MiniCPM#-changelog). Standard LlamaForCausalLM, 42 layers, 2 KV heads, 128K advertised context; screen at 8K, concurrency 2, BF16 KV. Approximately 4.69 GiB weights plus 1 GiB KV and runtime/activations/graphs: provisional 7–10 GiB, not measured. Chat, thinking toggle, coding and tool format; structured decoding is a runtime feature requiring tests. Apache-2.0, retain license/notices and mark modifications; [license](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE). Upstream [vLLM recipe](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/vllm.md) and native [Llama runtime](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py). September 7 [independent release discussion](https://www.reddit.com/r/LocalLLaMA/comments/1w9skjz/minicpm52b_release_day/) showed 312 votes when searched September 10; promising attention, still only three days of checkpoint adoption. Strong fit for existing compiler architecture; reasoning verbosity and early-release runtime behavior are risks. Official GPTQ derivative created September 7, revision `6c1ee6fa521aa53f47cfb32696e6d8ef5b0db805`; that date is not a separate base release.

2. **LFM2.5-2.6B:** base and instruction released August 4; [base history](https://huggingface.co/LiquidAI/LFM2.5-2.6B-Base/commits/main), [publisher announcement](https://huggingface.co/blog/LiquidAI/lfm2-5-2-6b). Base revision `c57bdaed1ef166fe3095dda07f4a5e789ad5321e`. Hybrid short-convolution/attention model, approximately 5.02 GiB BF16 weights plus state/cache/runtime: provisional 7–10 GiB. Agentic training, multilingual chat and tool calls; [native runtime](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/lfm2.py). Custom [LFM Open License](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/654f9463ce32b05d0429d76fe1f580b27d4c1ac0/LICENSE) has a $10M annual revenue commercial-use threshold and redistribution conditions; cannot be presented as unrestricted Apache-equivalent packaging. Independent [August 14 tool-use report](https://www.reddit.com/r/LocalLLaMA/comments/1vo4oxw/lfm_25_26b_is_the_best_small_model_for_tool_use_i/) and [August 7 quantization investigation](https://www.reddit.com/r/LocalLLaMA/comments/1vi0d4i/lfm2526b_modelkv_cache_quantization_report/) establish exact-family experimentation beyond marketing. Potentially excellent loading/decoding; new hybrid compiler path and license make it a less straightforward community default.

3. **Granite 4.2-3B — older-base fallback:** instruction/post-training release August 25, explicitly based on Granite 4.1-3B-Base from April 2026. Not a recent base under the requested rule. Dense Granite architecture (scaling semantics differ from Llama), 6.82 GiB BF16 weights; provisional runtime 9–13 GiB. Apache-2.0, reasoning toggle, multilingual/tools. The [model card](https://huggingface.co/ibm-granite/granite-4.2-3b) includes a custom reasoning parser and points to qwen3_coder tool parsing. [Independent measured evaluation](https://artificialanalysis.ai/models/granite-4-2-3b) provides adoption evidence; publisher figures are not our quality results. More memory and framework adaptation than the leading pair.

4. **MiniCPM5-1B — older-base fallback:** May 19 release date in upstream changelog conflicts with May 21 Hub repository creation; both precede the 60-day window. August repository edits do not make it new. Dense 1.08B, 2.01 GiB BF16 weights; provisional runtime 4–7 GiB. Apache-2.0, same standard Llama runtime, smaller load/transfer footprint; likely quality tradeoff requires measurement. Its substantial monthly downloads describe this exact 1B checkpoint, not adoption of the new 2B model.

Existing Qwen3-4B and Qwen2.5-Coder-1.5B experiments are preserved. Their earlier Studio supplied-facts writing gates failed; they are not treated as qualified alternatives merely because loading worked. MiniCPM5-2B is a new separate integration; it was not already packaged. Only the leading pair and MiniCPM representations were downloaded. Granite and 1B remain research alternatives, not measured competitors.

## Measured selection

MiniCPM5-2B direct mode completed useful concise answers without mandatory reasoning.
Its official W4 checkpoint is selected after lossless Paiton decode repacking: 2.10 GB
weights, 14/20 deterministic task quality, versus 15/20 BF16/FP16. Paiton and stock W4
matched all 20 outputs; this is parity with the derivative, not perfect task accuracy.
The publisher labels it GPTQ but actually stores asymmetric AWQ GEMM tensors, G128.
The custom compiled kernel executes on gfx1201 with FP16 dequantization and FP32
accumulation, not native INT4 arithmetic. BF16-specific kernels did not improve
full serving over native vLLM skinny GEMM and were rejected.

LFM2.5-2.6B scored 18/20 using its required thinking mode and recommended sampling.
Its first useful response took 65.55 seconds in the corrected trial; reasoning
length and its commercial license conditions weigh against a quick-chat default.
The model's template ignores enable_thinking=false: it is not a matched direct-mode
comparison. Its stronger quality result remains a reason to consider it separately.

A separate llama.cpp BF16 conversion completed its first useful response in 9.57 s
in one screen, materially faster startup than vLLM. It is not fully API/quality
qualified and does not use the compiled Paiton path. This result is reported rather
than claiming vLLM is the fastest available loader.

Selected W4 derivative adoption, observed September 10 at 11:25 UTC: **528 monthly downloads and 13 cumulative likes** for the exact `MiniCPM5-2B-GPTQ` repository. These are much smaller than the BF16 checkpoint counters and must not be substituted with family totals. Its main branch advanced during this task to `4787c007325ecec3f56595ab2dd14a5a5f9b0520` for a README cookbook row; qualification deliberately retains `6c1ee6fa521aa53f47cfb32696e6d8ef5b0db805`. No historical growth is inferred.

Runtime support dates are separate from model dates: native Llama/AutoAWQ and LFM2 execution were tested on this machine on September 10, 2026 using the pinned vLLM runtime. The first upstream support dates for each parser/backend were not established; the linked current implementations are maturity evidence, not new base-release dates. Granite and 1B runtime compatibility remain source-level assessments rather than local qualification.
