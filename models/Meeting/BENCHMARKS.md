# R9700 qualification

Hardware: one AMD Radeon AI PRO R9700, 32 GiB VRAM, RDNA4 gfx1201, 64 compute units / 32 WGPs. Host RAM: 16 GiB. Runtime: Python 3.14.6, Torch 2.11 / ROCm 7.14, Transformers 5.14, native SDPA. No hosted judge or paid inference service was used.

## Declared primary scenario

The engineering objective is a 20% reduction in median warm complete-pipeline processing time on the 2,345.493375-second (39:05) AMI ES2004b distant-microphone recording, batch one, English, with matched checkpoint revisions, precision, decoding, VAD, diarization and summary settings. The Transformers-based complete-pipeline comparison is complete and did not meet the objective. A faster native vLLM summary backend is undergoing full-pipeline qualification; a backend change is never counted as Paiton acceleration. Sequential stage processes include model-switching costs. Persistent checkpoint and kernel caches are retained. First processing in a series is separate from repeated cached processing and is not described as a cleared OS page cache.

RTF is processing seconds / audio seconds. Audio hours per wall-clock hour is its reciprocal. Offline capture has no provisional/final streaming transcript latency; processing begins after recording stops or an import finishes.

## Complete pipeline with the Transformers helper

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

The actual `process --summary-backend vllm` command then completed the spoken audio fixture in 191.89 seconds: ASR/startup 26.95 s, diarization/startup 21.14 s, attribution 0.06 s, playback 0.20 s and summary/startup 143.51 s. Summary model loading/graph initialization took 88.60 s and generation/auditing 27.43 s. This first-use short example is slower overall than the previous Transformers example; inference speed alone does not establish startup or end-to-end superiority. Worker allocator peaks were 14.67 GB allocated / 14.77 GB reserved, including KV and graph pools. A named local worker extension retrieves these counters without enabling pickle-based RPC serialization. Repeated long-pipeline comparison is pending.

## Summary coverage against the human AMI reference

The initial vLLM runs were compared to AMI's human-authored abstract, four grouped final-decision sentences and one action sentence. Both initial outputs recovered **0/4 final-decision groups and 0/1 action items**. Stock included a transcript-supported TV-only/no-teletext scope statement; Paiton retained eight discussion-topic labels with no decision/action claims. The scope statement is not counted as recovery of a different reference decision. Action precision is undefined when no actions are emitted.

Missing reference items include the target buyer, one-design decision, Video Plus/seashell shape, docking/manual/casing choices and the marketing task to research instruction manuals. This is a major coverage limitation in topic-focused partial drafts. These are incomplete v1 drafts, not exhaustive notes or a measured 60–75% recall result. The full transcript remains available. The short synthetic fixture separately checks correct included decisions/actions and rejection of fictional commitments. The protocol, reference text, source hash and run counts are in `benchmark/ami-summary-reference-check.json`; this is not an independent human factuality rating.

## Footprint and limits

Prepared role directories occupied about 2.51 GB Parakeet, 33.7 MB community-1, 7.33 GB Granite and 2.27 MB VAD. Each separately initialized runtime cache occupied about 815 MB after these tests. The candidate image reports 11.90 GB uncompressed, with layers shared with the existing Studio runtime. These are measured logical bytes, not promised incremental download sizes. Keep space for original audio, normalized playback, model downloads and Docker build layers.

Stages release GPU allocations before the next model loads. A prior full compact-summary run peaked at about 8.04 GB allocated / 8.58 GB reserved; diarization at 1.78 GB / 2.25 GB. Driver memory is sampled separately because it includes allocations outside Torch. The Transformers complete-pipeline telemetry and repeat statistics are reported above; final-backend qualification remains in progress. Quantization probes and their failure cases are documented in `QUANTIZATION.md`.

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
