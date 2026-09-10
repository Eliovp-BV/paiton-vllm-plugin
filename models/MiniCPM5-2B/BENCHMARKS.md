# Loading, generation and quality — 10 September 2026

Target: one Radeon AI PRO R9700, gfx1201, 64 CUs / 32 WGPs, 32 GB VRAM; i5-8400 (six cores), 16.67 GB host RAM, 4.29 GB swap, Samsung 860 SATA SSD. The exclusive Studio GPU lease serialized all GPU tests. Heavy builds and storage reads used that lease too. Normal host background services remained active. No clocks, power limits, voltage or fan settings were changed.

The checkpoint, tokenizer and chat template are pinned in `checkpoint.lock.json`. Stock and Paiton use FP16 activations/KV, 8K context, two sequences, 512 scheduled tokens, 1 GiB KV, temperature 0, seed 1201, thinking off, no prefix hits or speculation. The final comparison uses TRITON_ATTN and capture-only full decode graphs for batches 1/2. O2/ROCM_ATTN, BF16, source FP16, alternate worker methods and eager configurations were screened; O2 was not the fastest stock choice. Both final modes receive the same lazy FlashAttention import patch.

## Loading

Timing starts before container launch and ends after a completed useful answer, not an open port. We report `/health` separately. “Warm weights” means explicit weight-page read-ahead outside the launch timer; runtime cache artifacts are already populated. Cold-weight trials use independent checkpoint inodes, `POSIX_FADV_DONTNEED`, and measured page residency below 1% immediately before launch. No system-wide cache dropping was used. Runtime/library filesystem pages are not independently forced cold.

| Boundary | Trials | API health, median (s) | First useful response, median (range), s |
|---|---:|---:|---:|
| Paiton, prepared runtime cache / warm weight pages | 3 | 27.65 | 27.98 (27.13–28.77) |
| Stock, prepared runtime cache / warm weight pages | 3 | 26.71 | 27.15 (25.81–31.25) |
| Paiton, empty runtime cache / warm weight pages | 3 | 45.80 | 46.53 (43.53–54.15) |
| Paiton, prepared runtime cache / cold private weight pages | 3 | 29.61 | 29.81 (29.16–29.96) |

All individual values, first visible-answer timestamps and cache boundaries are in [results.json](results.json). `release-stock-warm-1` actually populated an empty runtime cache and is excluded from the prepared-cache group; its measured first answer was 47.77 s. The first private cold trial also had an empty runtime cache: 45.57 s. These are retained rather than mislabeled.

The engineering targets declared before compiler work were ≤30 s prepared startup and ≤90 s with cached weights / empty runtime caches. All three Paiton prepared warm trials and all three empty-cache trials met their respective targets. Paiton adds about 0.83 s to the prepared-startup median versus matched stock. Three trials do not establish tail percentiles.

Weight transfer from Hugging Face took 28.497 s and SHA verification 5.718 s for the selected checkpoint in the original download experiment. All files total 2,109,727,920 bytes; weights alone are 2,099,615,880 bytes. These costs are separate from the cached-weight startup table. Verification receipts avoid rehashing unchanged files on subsequent launches; `--verify` forces a rehash.

The image reuses an existing 11.79 GB compressed public runtime. A fresh pull was not measured because those layers were already cached. The container root filesystem occupies 27.84 GB of allocated disk blocks, excluding Docker’s compressed content/build cache and external model/runtime caches. Reserve 60 GB for a first installation. New model-specific layers are small and share the existing runtime; exact final layer sizes are in `container-images.json`. Reconstructing all runtime layers was rejected after unnecessary build/storage work. Direct I/O on the private 2.10 GB weight copy measured 553 MB/s; this is a read-throughput test, not an uncached container-launch claim.

## Warm interactive chat

Two fresh processes per mode, four recorded warmups per prompt per process, then eight measured replies per prompt per process: 16 samples per row and 48 replies per mode. The declared primary metric is median end-to-end latency across this fixed three-prompt suite. Raw answers, token counts, timings and warmups are retained in `benchmark/results/*.json.gz`.

| Prompt | Output tokens | Stock median (s) | Paiton median (s) |
|---|---:|---:|---:|
| Binary search in two sentences | 48 | 0.8345 | 0.5566 |
| 17 + 25, integer only | 2 | 0.0609 | 0.0607 |
| Concise palindrome function | 40 | 0.7039 | 0.4708 |

The suite median was **0.7039 → 0.4708 s**, a **33.1% reduction** in these matched repetitions. Suite means were 0.5343 and 0.3634 s. The two-token answer is dominated by fixed serving overhead and does not improve materially.

**Timing variability limits the claim.** Earlier one-warmup trials reached suite medians of 0.3374 s stock and 0.2326 s Paiton. Some explanation runs changed from roughly 0.55 to 0.27 s within one process. Longer warmup did not eliminate the cross-process shift; its cause is not established. The 33% result is a matched repeated comparison, not a guarantee against the fastest isolated historical stock sample. Early/ramp measurements remain in the retained data. No kernel-only gain is presented as a serving gain.

## Sustained generation and prefill

Each mode uses eight rounds after two warmup rounds. A round submits 128-, 512- and 2,048-input-token requests, each forced to 256 output tokens. That is eight samples per row, 24 requests per mode/concurrency. Pool width two gives modest concurrency over this three-request mix, not permanent full saturation. Raw completions use `ignore_eos=true` to measure compute; quality is evaluated separately with natural stopping. “No final channel” in these raw-completion records is inapplicable, not a reasoning failure.

| Input / output; concurrency 1 | Stock e2e (s) | Paiton e2e (s) | Stock / Paiton TTFT (ms) | Stock / Paiton token interval (ms) |
|---|---:|---:|---:|---:|
| 128 / 256 | 1.930 | 1.215 | 43.7 / 50.4 | 7.40 / 4.57 |
| 512 / 256 | 1.945 | 1.233 | 47.6 / 53.7 | 7.44 / 4.63 |
| 2048 / 256 | 2.140 | 1.445 | 174.7 / 199.3 | 7.71 / 4.89 |

| Request-pool width | Stock aggregate output tokens/s | Paiton aggregate output tokens/s |
|---|---:|---:|
| 1 | 127.6 | 197.0 |
| 2 | 185.2 | 279.3 |

Paiton improves decode but adds prefill/host overhead: TTFT is about 6–25 ms higher in this matched sample. Effective input tokens/TTFT, per-request throughput, all ranges and standard deviations are in `results.json`. Input tokens/TTFT includes API, scheduling and first-token work; it is not a pure GPU prefill-kernel rate. Paiton keeps stock prefill math while compiling the small-token projections.

## Memory, Studio and quality

Final sustained trials: Paiton framework allocation 4.368 GB, reservation 4.538 GB, reported peak allocation 4.437 GB; stock allocation 3.308 GB, reservation 3.506 GB, peak allocation 3.378 GB. Framework peaks are read after initialization and may reflect runtime-internal resets. Continuous 0.5-second driver samples peaked at 5.096 GB (4.75 GiB) Paiton and 4.065 GB (3.79 GiB) stock; sampling may miss brief spikes. The extra layout is approximately 0.97 GiB, with zero compiler workspace and no second on-disk checkpoint. Container host-memory peaks and anon/file-cache splits are retained separately from GPU memory. Prepared runtime caches occupy 39.1 MB Paiton and 39.4 MB stock, excluding checkpoint weights.

Studio’s actual isolated runtime/store/streaming flow took 49.09 s on its first use, 0.82 s for a warm follow-up in the same container, 0.94 s to unload, and 28.46 s to reload. A separate MiniCPM → GPT-OSS → MiniCPM switch returned correct answers and removed the prior owned container each time: 29.39 s initial MiniCPM, 144.22 s to GPT-OSS with a fresh runtime cache, 40.04 s back to MiniCPM. Filesystem caches were uncontrolled in the switch test. Thus sub-30-second switching after a large model is **not** established. Only isolated stores and owned containers were changed; active user services were preserved.

The reproducible 20-case evaluation covers chat, strict instructions, facts, uncertainty, arithmetic/logic, four executable coding tasks, JSON/schema output and tools. Stock/Paiton W4 both score **14/20**, with **20/20 exact final-text matches** (transport IDs/timings excluded). Source BF16/FP16 scores **15/20**; W4 loses the `code-runs` task. Remaining W4 failures are arithmetic, logic, filtering, rate reasoning and negative-number reasoning. Paiton is parity with this derivative, not perfect accuracy. LFM2.5-2.6B scores 18/20 under its compulsory thinking mode and its recommended sampling; that is not a matched direct-answer comparison.

Kernel checks cover all four projection shapes at batches one/two, exact nibble/scale/zero repacking, zero input, invalid ABI input, numerical agreement and nonempty poisoned-output graph replay. All cosine similarities exceed 0.99999. The packaged artifact loads in a fresh `python -I -S` process without Torch or GPU device mounts; ELF dependencies contain no ATen/libtorch/libc10. Framework allocation stays in the plugin.

API checks cover natural EOS, streaming completion, multi-turn context, supplied facts, long-context retrieval, oversized-context rejection, streamed tool calls and tool results. The final package smoke adds a near-8K retrieval case; its retained result records the exact prompt-token count. Thinking-on W4 is experimental and excluded from this release qualification. Generated code runs only in bounded, unprivileged, network-disabled containers. No paid API or hosted judge was used.

## Profiling and rejected paths

Loading traces identified an unused FlashAttention/AITER import that initialized HIP in the API process and forced spawn. Deferring that import permits safe fork; the patch applies equally to stock and Paiton. Model load, artifact/weight preparation and graph/profile initialization are separate log stages. In the prepared diagnostic, weight reading was about 1.6 s, total model loading about 2.9 s and profile/graph initialization about 5 s; imports and frontend preparation dominate the remaining startup. Tokenizer initialization is included in frontend timing, not claimed as independently isolated.

BF16 standalone GEMV microbenchmarks initially appeared faster than generic Torch linear, but native vLLM skinny GEMM removed that advantage. Fused gate, row-layout, alternate runner, import-time read-ahead and wider kernel variants did not produce a useful serving improvement and were rejected. The first microbenchmark had an empty captured graph and is invalid; corrected replay tests replaced it. Native offline profiling also exposed reversed GPU timestamps in this SDK; no precise stage-percentages are inferred from those traces.

A separately converted llama.cpp BF16 screen completed its first useful answer in 9.57 s, materially faster startup than vLLM. It had only six screening prompts and no complete API/product qualification or Paiton path. This evidence remains visible: the package does not claim vLLM is the fastest possible small-model loader.

## Final versioned image validation

Both stock and Paiton passed 8/8 protocol checks in the final versioned image, including retrieval with 8,040 input tokens and over-context rejection. Its integration and artifact hashes match the full benchmark records; the shared runtime patch matches the retained source. The superseded intermediate image was unavailable for a complete filesystem comparison. Final-image chat observations are retained separately and do not replace the paired measurements above. Stock again changed latency within a run, reinforcing the variability limitation.

The final Studio adapter/image smoke test measured 29.38 s initial useful response, 0.61 s warm reply using the same container, 1.01 s unload, and 28.38 s reload. These are single final validation observations; repeated startup trials and the separate larger-model switching measurements above remain the performance evidence. Nine focused Studio tests passed.
