# Local meeting notes for Paiton Studio

**Review candidate, 1.0.0rc1 — not a published release.** Recording import, codec handling, ASR, anonymous diarization and compiler execution have local tests. The complete CLI and Studio processing paths passed a generated spoken regression test. Both summary-backend comparisons are complete; final source-built-image integration checks are pending. Do not treat partial drafts as approved meeting records.

The pipeline separates audio capture/import, Silero voice activity detection, Parakeet speech recognition, pyannote speaker diarization and a small local text summarizer. A speech recognizer does not capture Teams audio or identify participants by name. English is the primary qualification language. Parakeet v3 supports multiple languages upstream; this package has not yet qualified them all.

| Component | Pinned choice | Local implementation |
|---|---|---|
| Speech recognition | NVIDIA Parakeet TDT 0.6B v3 | Transformers, FP16 SDPA; optional Paiton prediction LSTM |
| Voice activity | Silero VAD | CPU TorchScript, silence gating without shifting the audio clock |
| Speaker turns | pyannote community-1 | Separate offline GPU stage; overlapping and uncertain turns retained |
| Partial summary | Granite 4.2 3B (3.7B actual parameters) | Compact text helper; native vLLM with a Transformers alternative, evidence-linked partial notes |

Exact revisions and licenses are in [models.lock.json](models.lock.json). Accept community-1 access conditions with your own Hugging Face account before running the download preparation step. The runtime uses cached files and has no network access. Model weights are not included in this candidate's source package.

No separate forced-alignment model is shipped: word timing and punctuation come from Parakeet, followed by speaker-turn attribution. Timestamp limitations are measured separately from WER.

## Recording import and capture

See [CAPTURE.md](CAPTURE.md) for the browser/OS capability matrix and a complete source-selection workflow. Shared audio may omit the local microphone; this version records one source at a time.

Studio accepts bounded, ordered uploads and preserves the imported original. Real codec round trips have been tested for WAV, FLAC, MP3, M4A/AAC, MP4/AAC, Ogg/Opus and WebM/Opus. An extension does not guarantee that every codec inside a container is supported. Invalid/no-audio files fail explicitly. The current limits are 4 GiB per imported file and eight hours of decoded audio.

Select an audio track or channel for separate feeds. Ordinary stereo can be downmixed; separate microphone/system feeds are never combined automatically. An original with multiple tracks remains intact. Transcript playback uses a normalized WAV whose zero is the selected decoded audio start, avoiding video-container start-offset errors.

| Capture path | Status and limitation |
|---|---|
| Exported/uploaded Teams or other meeting recording | Recording import tested; this is not a Teams application integration |
| Browser microphone | Visible user-started controls implemented; automated Chromium test uses a simulated microphone, not a real meeting |
| Browser tab sharing with audio | Implemented but not validated against a real Teams session; Chromium support depends on browser/OS and the selected sharing surface |
| Native Teams/system loopback | Not validated; tab sharing does not capture a native application |
| Server audio devices | The server cannot capture a microphone or system audio on a different client machine |

Browser microphone and display capture require HTTPS with a certificate trusted by the client, or localhost. Remote plain HTTP is not a working microphone-capture deployment. Studio's secure launcher accepts `PAITON_TLS_CERT` and `PAITON_TLS_KEY`; use the launcher's documented environment names when configuring it. The secure launcher passed a local HTTPS request with certificate verification and a Chromium secure-context/API check using an explicitly pinned test certificate. This does not validate a real remote client's trust configuration or capture hardware. Browser sharing prompts determine which audio is available. Chrome/Edge tab audio and Windows/ChromeOS system-audio capabilities differ; Firefox/Safari do not provide equivalent display-audio support. See [MDN screen capture](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getDisplayMedia).

Capture is recording-first, followed by offline processing. It is not a qualified streaming transcript. The UI shows start/stop state, stops on ended devices, and bounds queued capture uploads. Microphone and shared-audio capture are separate choices. Obtain participants' consent and follow your organization's recording rules.

## Local stage commands

See [REPRODUCE.md](REPRODUCE.md) for pinned model preparation, the required compiled-artifact build context and the one-command recording workflow. The base image is pinned by digest, and added Python dependencies are pinned in `requirements.lock`. The candidate runs under an unprivileged UID. Keep `/models/cache` persistent and mount original recordings read-only.

The candidate image defaults to `vllm` for `process` and checkpoint-based `summarize`; pass `--summary-backend transformers` for the alternate helper. Direct source execution defaults to Transformers unless `PAITON_MEETING_SUMMARY_BACKEND` is set. vLLM uses native BF16 Granite, Triton attention and graph caching; audio components remain outside vLLM.

The package CLI provides one-command `process`, plus `inspect`, `normalize`, `transcribe`, `diarize`, `attribute`, `summarize`, `export` and `benchmark-asr`; run `python -m paiton_meeting --help` for their arguments. Separate ASR and diarization processes release their GPU allocations before the summary model starts. Studio schedules these stages through its existing GPU queue and only stops containers it owns.

`benchmark-asr` records one first-processing pass plus at least three warm runs, transcripts, raw timings and sampled driver/allocator telemetry. RTF means processing seconds divided by audio seconds; its reciprocal is audio hours processed per wall-clock hour. This command measures the ASR stage, not end-to-end meeting processing. See [BENCHMARKS.md](BENCHMARKS.md) for measured model switching, summary latency and image-specific limitations.

JSON, plain transcript, SRT, VTT and plain-text summary exports preserve source IDs/timestamps. Speaker names are user-entered labels; no voice-based identity recognition is claimed. Summary owners and deadlines remain unspecified unless supported by the transcript. Summary generation treats spoken instructions as untrusted data and rejects invalid references or incomplete model output. Hierarchical reduction carries source quotes and IDs; a separate local audit checks claims against nearby transcript context. Unresolved-issue entries additionally require an explicit question or uncertainty cue in the cited English text, so facts and suggestions are conservatively omitted from that category. These checks do not prove factuality: outputs remain partial drafts for review against playback. This initial version accepts incomplete coverage; the full transcript remains available. No measured 60–75% recall claim is made.

Recordings, transcripts and summaries remain local. Delete controls remove the owned meeting and derived files; optional retention removes the original import copy and normalized playback after successful processing. Deleting a Studio copy does not delete the user's original source file. Redundant intermediate job files are removed after successful persistence. No raw meeting content needs to be sent to an external API.

## Qualification results so far

On the 39:05 AMI ES2004b distant-microphone recording, three matched warm FP16 ASR runs measured median 87.44 s stock and 77.32 s with the Paiton decoder: **11.58% lower ASR wall time**. WER was 20.42% and 20.44%, respectively. This is not a 20% complete-pipeline result. Word ordering during overlap and unannotated room speech affect the reference comparison.

Pyannote's first offline run took 77.53 s. With original manual utterance labels, a 250 ms collar, the complete recording interval and overlap included, DER was 14.90%, versus 49.21% for the rejected Sortformer candidate. This protocol differs from forced-alignment model-card benchmarks. Attribution remains imperfect, particularly around overlap and very short interjections.

Granite Speech 5.0 470M TurboCTC executes in BF16 and FP32 on gfx1201. It was substantially faster but less accurate than Parakeet on five matched meeting excerpts. On 20 clean LibriSpeech clips (one speaker), Parakeet/Granite WER was 2.46%/3.79%; with deterministic added white noise at 10 dB SNR, it was 7.37%/4.91%. This narrow comparison does not establish broad superiority. Parakeet remains the meeting ASR choice.

Voxtral Mini (4.7B total) executed locally but hallucinated during raw silence and failed the correction/instruction-attack summary fixture. It is not the default. VAD protects the selected ASR pipeline from processing silent chunks, but does not establish general hallucination immunity.

The complete command processed a 38.90-second synthetic spoken regression recording in 154.00 seconds on first use, including stage startup. Studio processed it in 145.82 seconds using its actual GPU queue. Both produced the same supported Morgan/Tuesday action and undecided-launch issue without accepting the fictional injected budget or email invitation as commitments. The explicit USB-C decision was omitted. Browser checks covered import, completion, speaker renaming, synchronized playback seeking and JSON export with no page errors. These short first-use results are not long-meeting throughput measurements or a broad factuality score.

The compact Granite adapter uses bounded context and hierarchical summaries. Large Qwen3.8 and GPT-OSS models are excluded from this small configuration. Summary coverage is intentionally accepted as partial for this initial version; included claims still need review against source timestamps. See [SELECTION.md](SELECTION.md), [QUANTIZATION.md](QUANTIZATION.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Licenses and attribution

Package integration code follows this repository's Apache-2.0 license. NVIDIA Parakeet v3 and pyannote community-1 are CC-BY-4.0 checkpoints; Silero VAD is MIT; Voxtral Mini, Granite 4.2 and Granite Speech TurboCTC are Apache-2.0. Consult each upstream model card and the inherited runtime's notices. Gated access is separate from copyright licensing, and this package does not accept access conditions on a user's behalf.

The public AMI evaluation recording and annotations are from the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/), distributed under CC-BY-4.0. Cite the AMI corpus creators when redistributing those recordings or derived examples. No private meeting audio or derived content belongs in release evidence.
