# R9700 qualification

**Release candidate image published; source ready for PR review.** The final matched Paiton complete-pipeline median clears speed parity. The 20% stretch objective remains unmet. Earlier experiments, including the rejected slower candidate, are retained below. The versioned image is available on GHCR; see [RELEASE.md](RELEASE.md).

Hardware: one AMD Radeon AI PRO R9700, 32 GiB VRAM, RDNA4 gfx1201, 64 compute units / 32 WGPs. Host RAM: 16 GiB. Runtime: Python 3.14.6, Torch 2.11 / ROCm 7.14, Transformers 5.14, native SDPA. No hosted judge or paid inference service was used.

## Declared primary scenario

The engineering objective is a 20% reduction in median warm complete-pipeline processing time on the 2,345.493375-second (39:05) AMI ES2004b distant-microphone recording, batch one, English, with matched checkpoint revisions, precision, decoding, VAD, diarization and summary settings. The final native-vLLM comparison clears median speed parity but does not meet the 20% stretch objective. The launcher selects the compiled variant by default; use `--stock` for the native baseline. A summary backend change is never counted as compiler acceleration. Sequential stage processes include model-switching costs. Persistent checkpoint and kernel caches are retained. First processing in a series is separate from repeated cached processing and is not described as a cleared OS page cache.

RTF is processing seconds / audio seconds. Audio hours per wall-clock hour is its reciprocal. Offline capture has no provisional/final streaming transcript latency; processing begins after recording stops or an import finishes.

## Final candidate: complete matched comparison

Frozen image `sha256:f7e6a83e38d41f0c89fc8175e8cf5d0080b333ea0a802d52369360069612e986`; initial pair plus four alternating cached repetitions per variant. Both use native safetensors prefetch and resolved timestamp postprocessing. These shared improvements are not compiler acceleration. Only prediction-LSTM execution differs between variants. No interleaved GPU probes or builds ran during the batch.

The declared median result is **291.03 seconds stock / 284.50 seconds Paiton**, a **6.53-second (2.24%) reduction**. Small samples and run-to-run variability limit generalization; the saving is not guaranteed for every recording or every run. Summary lengths can differ following small ASR differences.

| Cached complete pipeline | Stock | Paiton |
|---|---:|---:|
| Median seconds | 291.0268 | 284.4957 |
| Minimum–maximum seconds | 289.7969–312.8895 | 283.8412–301.4842 |
| Sample standard deviation | 11.1518 | 8.6216 |
| RTF | 0.124079 | 0.121295 |
| Audio hours / wall hour | 8.0594 | 8.2444 |
| ASR processing seconds | 35.3931 | 30.8138 |
| Diarization with startup seconds | 68.2879 | 69.0255 |
| Summary with startup seconds | 175.2635 | 171.7106 |
| Sampled driver peak, decimal GB | 15.790 | 15.790 |

All ten final transcripts scored WER 20.4362% / CER 15.7750%. All 448 speaker turns exactly match the previously scored community-1 baseline (14.90% DER under the documented 250 ms collar protocol). Timestamp and boundary metrics were recomputed on each final output; all mechanical source-reference checks passed. Summary objects match the earlier reviewed partial notes: the AMI example still recovers none of the four human-reference final-decision groups or the one action. The synthetic fixture separately retains its supported USB-C decision and Morgan/Tuesday action and rejects injected commitments. No 60–75% recall claim is made.

[Full stage/startup/memory statistics](benchmark/final-complete-pipeline.json) · [All ten timing rows](benchmark/final-complete-pipeline-timings.json) · [Final transcript/timestamp/reference scores](benchmark/final-output-quality.json) · [Summary assessment](benchmark/final-summary-reference-check.json).

## Earlier shared-loader candidate: completed comparison

The frozen loader-review image `sha256:492f23f9e739a28b52839b330b1df88381c10e66d2096d5b616624cc35e2b511` adds native safetensors prefetch for both stock and Paiton. Models, precision, decoding, chunking, diarization and summary settings match. Only the prediction LSTM implementation differs between variants. Initial pair plus three alternating cached repetitions each; no interleaved GPU probes or builds.

| Complete processing seconds | Stock | Paiton |
|---|---:|---:|
| Cached runs | 341.2266, 349.7866, 343.3326 | 351.9769, 329.9314, 333.6445 |
| Median | 343.3326 | 333.6445 |
| Sample standard deviation | 4.4603 | 11.8030 |
| Initial run (existing caches) | 377.5911 | 331.0003 |

**Paiton is 2.82% faster by the declared complete-pipeline median.** This clears median speed parity on this recording, with visible variability and a small sample; it does not meet the original 20% objective or guarantee a speedup on every run. Native prefetch benefits both variants and is not compiler acceleration. Generated summary lengths differ and are reported separately.

All eight final transcripts scored WER 20.4362% / CER 15.7750%. Their speaker turns exactly match the previously scored community-1 result. Timestamp metrics were recomputed on the final outputs; all mechanical source-reference checks passed. Summary coverage limitations remain: none recovers the four grouped final decisions or the one action in the human abstract.

[Complete stage, loading, memory and variability report](benchmark/loader-complete-pipeline.json) · [Raw timing rows](benchmark/loader-complete-pipeline-timings.json) · [Direct final-output scores](benchmark/loader-final-quality.json) · [Summary reference assessment](benchmark/loader-summary-reference-check.json).

## Earlier rejected source-runtime complete pipeline

Eight runs used the same frozen source-built image, audio, pinned models, FP16 ASR, BF16 native-vLLM summary, VAD, speaker diarization, timestamps, chunking and decoding settings. An initial pair is followed by three alternating cached repetitions per variant; only the prediction LSTM implementation differs. No builds or other GPU probes ran during this batch.

| Cached metric (three runs each) | Stock | Paiton |
|---|---:|---:|
| Median complete seconds | 413.8706 | 423.6724 |
| Minimum–maximum seconds | 410.5330–415.1802 | 367.3708–424.2612 |
| Sample standard deviation, seconds | 2.3962 | 32.6770 |
| RTF: processing / audio seconds | 0.176454 | 0.180633 |
| Audio hours / wall-clock hour | 5.6672 | 5.5361 |
| ASR with startup, median seconds | 107.00 | 101.29 |
| Diarization with startup, median seconds | 70.94 | 68.89 |
| Summary with startup, median seconds | 230.91 | 247.80 |
| Summary load/graph initialization, median seconds | 59.88 | 86.19 |
| Summary generation/audit, median seconds | 141.06 | 133.47 |
| Summary worker allocated / reserved, decimal GB | 15.08 / 15.21 | 15.08 / 15.21 |
| Sampled driver peak, median decimal GB | 15.79 | 15.79 |
| Peak summed process RSS, median decimal GB | 6.70 | 7.45 |

**Paiton was 2.37% slower by the declared complete-pipeline median; the 20% objective was not met.** Its faster ASR and shorter summary generation did not produce a lower complete median in this sequential deployment. Startup varied substantially; this does not establish that the compiler caused the summary-loading difference. The faster individual Paiton run is retained in the range, not substituted for the median. That earlier candidate therefore retained stock as its default; the final qualified launcher above now selects Paiton.

The initial pair took 424.50 s stock / 382.98 s Paiton after prior caches existed. These are fresh stage processes, not cleared OS caches. The first stock launcher also waited for the separately validated Studio job; that wait is excluded from its 424.50-second processing time. For an already available recording, end-of-meeting latency is complete processing plus import, queue and launch overhead. This is offline processing; streaming partial/final latency is not applicable.

Stage medians need not sum to the complete median. Allocator counters cover the native vLLM worker; ASR and diarization allocation probes are reported separately. Driver sampling covers the complete owned pipeline; samples may miss peaks. Summed RSS can double-count shared pages. Three repetitions are a small sample, and summary lengths differ between ASR variants.

The [detailed report](benchmark/release-complete-pipeline.json) includes exact image/source provenance, all stage/loading timings, memory sample counts, variability and limitations. [Raw timing rows](benchmark/release-complete-pipeline-timings.json) retain all eight runs. The final summaries remain topic-focused partial drafts: none recovers the four grouped final decisions or the one action in the AMI human abstract; stock includes one supported TV-only/no-teletext scope statement. See the [per-run reference check](benchmark/release-summary-reference-check.json). No 60–75% recall or broad factuality score is claimed.

## Earlier complete pipeline with the Transformers helper

Eight full recording runs comprise an initial pair and three alternating cached repetitions per variant. Every run starts fresh stage processes with persistent file/kernel caches. The same pinned models, precision, decoding, timestamps, diarization and summary policy are used; only the ASR prediction LSTM changes.

| Metric | Stock | Paiton |
|---|---:|---:|
| Median complete seconds | 457.7986 | 493.2165 |
| Minimum–maximum seconds | 450.3891–459.7366 | 468.3250–493.4862 |
| Sample standard deviation | 4.9334 | 14.4496 |
| RTF | 0.195182 | 0.210283 |
| Audio hours / wall hour | 5.1234 | 4.7555 |
| Summary output tokens | 6,776 | 7,582 |
| Summary generation/audit calls | 19 | 22 |

**Paiton was 7.74% slower end to end in this configuration.** Tiny ASR differences caused longer downstream drafts and additional audit calls; the ASR saving did not offset this. This does not establish a 20% complete-pipeline improvement. Initial-pair processing was 456.24 s stock and 475.92 s Paiton, with models already downloaded and prior caches present.

Median sampled driver peaks were about 9.04 GB in both variants. Median peak summed process RSS was 7.25 GB stock and 7.19 GB Paiton; RSS can double-count shared pages. Active-container intervals exclude GPU usage while the launcher waits for the shared lease. First-stock RAM monitoring began late, and sampling can miss peaks. Full stage loading/processing times, allocator counters, sample counts, variability and limitations are retained in `benchmark/transformers-complete-pipeline.json`; individual wall times are in `benchmark/transformers-complete-pipeline-timings.json`. This report pins the actual frozen image and discloses the later attribution-validation change separately.

## Measured ASR stage

Three warm repetitions per variant, following a separate first pass, use the same FP16 Parakeet checkpoint, SDPA, Silero VAD, 30-second windows, two-second margins, greedy TDT decoding budget and timestamps. Only prediction LSTM execution changes.

| Metric | Stock | Paiton |
|---|---:|---:|
| Median ASR seconds | 87.4439 | 77.3205 |
| Minimum–maximum seconds | 87.0684–88.7259 | 77.1292–77.8862 |
| Sample standard deviation | 0.8691 | 0.3936 |
| RTF | 0.037282 | 0.032966 |
| Audio hours / wall hour | 26.8228 | 30.3347 |
| WER | 20.4216% | 20.4362% |
| CER | 15.7432% | 15.7750% |

Paiton reduces ASR wall time by **11.58%**. This is not complete-pipeline acceleration. Outputs were stable across the four passes within each variant; stock/Paiton output edit WER was 0.0334%. The reference comparison uses ordered single-stream text, so overlap and unannotated room speech affect errors. Raw aggregate results are in `benchmark/asr-comparison.json`.

## Attribution and timestamps

Community-1's full recording run took 77.53 seconds; the production memory-mapped decoder path subsequently took 81.62 seconds and produced exactly the same 448 speaker turns. Six anonymous clusters were produced for the AMI participants plus room/technician speech and oversegmentation. A 250 ms collar with original manual utterance labels, full recording interval and overlap included yields 14.90% DER, versus 49.21% for the rejected Sortformer alternative. Zero-collar DER was 19.21%. This is not the upstream forced-alignment benchmark protocol.

Among 5,522 lexically matched stock words against manual AMI word annotations, median absolute start error was 80 ms (p90 210 ms, p95 270 ms); median end error was 90 ms (p90 310 ms, p95 about 440 ms). Large alignment/overlap outliers reached about 20–21 seconds and must not be hidden by the median. These figures apply only to matched words. Around chunk boundaries, the diagnostic found 41 deletions and four insertions among 283 reference words; it does not prove chunking caused those errors. Paiton results were essentially unchanged. See `benchmark/timestamp-boundary-comparison.json`.

## Supplementary multilingual speech

The [Google FLEURS dataset](https://huggingface.co/datasets/google/fleurs), CC-BY-4.0, supplied three test recordings per language. Selection used the first three referenced audio members in each pinned archive, without selecting based on model output. Both implementations used FP16 SDPA, automatic language handling and greedy transcription with timestamps. These are short read-speech examples, not multilingual meetings or a summary-language qualification.

| Language | Parakeet WER / CER | Whisper Turbo WER / CER |
|---|---:|---:|
| English (US) | 4.11% / 1.53% | 8.22% / 2.80% |
| German | 4.84% / 1.24% | 1.61% / 0.50% |
| French | 5.83% / 0.64% | 7.77% / 1.75% |
| Spanish (Latin America) | 2.94% / 0.51% | 1.96% / 0.34% |

Exact source revision, audio hashes, references and outputs are retained in `benchmark/fleurs-manifest.json` and the two `*-multilingual.json` results. `python benchmark/prepare_fleurs.py --output /path/to/fleurs-small` reproduces the bounded download selection without fetching full archives. The small sample count does not establish robust language or accent accuracy.

## Non-speech regression

The selected Paiton ASR plus Silero VAD produced zero words on three newly generated 30-second signals: silence, seeded white noise and harmonic instrumental tones. All windows were skipped by VAD; processing after model loading took 0.42–0.54 seconds per signal. This is not a claim about all music, singing or background audio. The signals are CC0, contain no human voices, and can be regenerated with `python benchmark/nonspeech.py --models /models/meeting --output /path/to/new-signals --artifact /path/to/meeting_lstm_float16_gfx1201.so`. Hashes, timings and outputs are in `benchmark/nonspeech-qualification.json`.

## Short complete-workflow example

The generated 38.90-second spoken test ran through the packaged command in 154.00 seconds on first use: ASR/startup 32.76 s, diarization/startup 40.77 s, attribution 0.06 s, playback normalization 0.19 s and summary/startup 80.19 s. The actual Studio queue completed the same input in 145.82 seconds. These small cold-start-dominated examples are functional evidence, not throughput claims.

The transcript preserved R9700, 32 GB, 64 compute units, GFX1201, BF16 and 8-bit weights with normalized spelling, but misheard “AI PRO” as “iPro.” The authored reference is in `examples/spoken-reference.json`; no broad technical-vocabulary score is claimed.

Both outputs retained the Morgan/Tuesday action and undecided-launch issue, rejected the fictional injected approval and email invitation, and omitted the explicit USB-C decision. Included references were checked against the actual transcript. The partial-summary scope accepts omissions; no 60–75% empirical recall score is asserted. A local model audit is not an independent factuality judge.

## Optional native vLLM text backend

A graph-enabled native Granite probe executed on gfx1201 using BF16 and Triton attention. Complete summary-only processing took 24.96 seconds on the spoken fixture and 138.60 seconds on the AMI transcript after a 40.13-second model load. It retained the fixture's USB-C decision, Morgan/Tuesday action and undecided launch, while rejecting the fictional approval and email invitation. The AMI notes retained later discussion and a referenced TV-only/no-teletext scope statement; open-question extraction remained imperfect.

The actual `process --summary-backend vllm` command then completed the spoken audio fixture in 191.89 seconds: ASR/startup 26.95 s, diarization/startup 21.14 s, attribution 0.06 s, playback 0.20 s and summary/startup 143.51 s. Summary model loading/graph initialization took 88.60 s and generation/auditing 27.43 s. This first-use short example is slower overall than the previous Transformers example; inference speed alone does not establish startup or end-to-end superiority. Worker allocator peaks were 14.67 GB allocated / 14.77 GB reserved, including KV and graph pools. A named local worker extension retrieves these counters without enabling pickle-based RPC serialization. The final repeated comparison is reported above.

## Summary coverage against the human AMI reference

The initial vLLM runs were compared to AMI's human-authored abstract, four grouped final-decision sentences and one action sentence. Both initial outputs recovered **0/4 final-decision groups and 0/1 action items**. Stock included a transcript-supported TV-only/no-teletext scope statement; Paiton retained eight discussion-topic labels with no decision/action claims. The scope statement is not counted as recovery of a different reference decision. Action precision is undefined when no actions are emitted.

Missing reference items include the target buyer, one-design decision, Video Plus/seashell shape, docking/manual/casing choices and the marketing task to research instruction manuals. This is a major coverage limitation in topic-focused partial drafts. These are incomplete v1 drafts, not exhaustive notes or a measured 60–75% recall result. The full transcript remains available. The short synthetic fixture separately checks correct included decisions/actions and rejection of fictional commitments. The protocol, reference text, source hash and run counts are in `benchmark/ami-summary-reference-check.json`; this is not an independent human factuality rating.

## Footprint and limits

Prepared role directories occupied about 2.51 GB Parakeet, 33.7 MB community-1, 7.33 GB Granite and 2.27 MB VAD. Each separately initialized runtime cache occupied about 815 MB after these tests. The candidate image reports 11.92 GB uncompressed, with layers shared with the existing Studio runtime. These are measured logical bytes, not promised incremental download sizes. Keep space for original audio, normalized playback, model downloads and Docker build layers.

Stages release GPU allocations before the next model loads. A prior full compact-summary run peaked at about 8.04 GB allocated / 8.58 GB reserved; diarization at 1.78 GB / 2.25 GB. Driver memory is sampled separately because it includes allocations outside Torch. The Transformers complete-pipeline telemetry and repeat statistics are reported above; the final source-runtime measurements are reported above. Quantization probes and their failure cases are documented in `QUANTIZATION.md`.

Tests cover codecs, bounded decoding, chunk stitching, evidence references, exports, cleanup, and spoken/text instruction attacks. Natural English AMI and one-speaker LibriSpeech probes support a limited qualification, not universal accuracy across accents or languages. Real microphone/tab/system capture and native Teams sessions are not validated by the simulated Chromium microphone test.

## Native vLLM complete-pipeline diagnostic

Eight runs completed on the same 39:05 recording using the frozen pre-source-build image: an initial pair followed by three cached repetitions per ASR variant. Native vLLM supplied the same Granite checkpoint, prompts, limits and sampling configuration in both variants.

| Cached complete pipeline | Stock ASR | Paiton ASR |
|---|---:|---:|
| Median seconds | 414.19 | 374.90 |
| Range seconds | 385.78–424.20 | 347.56–398.66 |
| Sample standard deviation seconds | 19.93 | 25.57 |
| RTF (processing seconds / audio seconds) | 0.17659 | 0.15984 |
| Audio hours / wall-clock hour | 5.663 | 6.256 |
| Median sampled driver peak, decimal GB | 15.790 | 15.790 |

The observed median reduction is 9.49%, below the 20% objective. Generated summary lengths differ between variants, so this complete-pipeline change is not a pure compiler-kernel speedup. A canceled NASM configure/compile attempt overlapped Paiton repetition 3 for approximately 16 seconds; retain that repetition as potentially affected by host contention. This batch is diagnostic and does not qualify the replacement source-built media image. Its exact image revision, per-stage results, allocator/process-memory measurements and limitations are in [the detailed report](benchmark/vllm-complete-pipeline-diagnostic.json); all eight raw timing rows are in [the timing file](benchmark/vllm-complete-pipeline-diagnostic-timings.json). No run was discarded from the reported statistics.

## ASR encoder and decoder breakdown

The earlier five-excerpt FP16 qualification recorded synchronized generation wall time and encoder HIP events separately. Median warm encoder time was 33.42 ms stock / 32.77 ms Paiton; the remaining generation time was 169.25 ms / 151.98 ms. The remainder includes decoder execution and Python/control overhead, and is not an isolated decoder GPU measurement. Median feature extraction plus host-to-device transfer was 18.68 ms / 18.80 ms. These short probes motivated the prediction-LSTM hot path; they do not replace the complete-pipeline benchmark. Per-excerpt measurements and their exact interpretation are in [asr-stage-profile.json](benchmark/asr-stage-profile.json).

## Earlier source-runtime transcript quality verification

Direct rescoring of all eight final outputs gives stock WER 20.4362% / CER 15.7750% in every run. Paiton WER ranges from 20.4070% to 20.4362%, and CER from 15.7345% to 15.7750%; one compiled repetition differs slightly. These final-output scores supersede any assumption that the earlier ASR-only texts were identical. All 448 speaker turns match the previously scored community-1 result exactly in every run. See [per-run transcript scores](benchmark/release-transcript-quality.json) and the [final timestamp/boundary assessment](benchmark/release-timestamp-boundary-comparison.json). Timestamp and boundary metrics were recalculated on these actual outputs using the same manual-reference protocol; no reference-based tuning was performed.
