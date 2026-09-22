# Local meeting audio — standalone package

## Model weights and existing downloads

Run the following examples from the repository root. This package needs the
pinned Parakeet, pyannote, Granite and Silero components in
[models.lock.json](models.lock.json). It does not download models during recording
processing. Complete the preparation step before starting a job.

### First download

Follow [Prepare persistent models](REPRODUCE.md#prepare-persistent-models),
including the separate download environment and your own pyannote access
approval. Then [pull the image](REPRODUCE.md#pull-the-image) and run:

```bash
export PAITON_MEETING_CACHE="$HOME/.cache/paiton-meeting"
./models/Meeting/run-docker.sh /absolute/path/to/meeting.mp4 /absolute/path/to/new-result
```

The launcher mounts `$PAITON_MEETING_CACHE/models` read-only and keeps runtime
caches separately. Creating an empty `models/` directory is not preparation.

### Already in a local folder

For an existing prepared package cache, select its parent directory:

```bash
export PAITON_MEETING_CACHE="/absolute/path/to/prepared-meeting-cache"
./models/Meeting/run-docker.sh /absolute/path/to/meeting.mp4 /absolute/path/to/new-result
```

It must contain `models/parakeet`, `models/pyannote`, `models/granite-summary`,
`models/silero/silero_vad.jit` and `models/model-provenance.json`. Keep the
`models/hub` files referenced by any snapshot links. The runtime creates its
writable `runtime/` sibling as needed.

If your complete prepared **models folder itself** is stored elsewhere, mount
it explicitly instead of rearranging it:

```bash
export PAITON_MODEL_DIR="/absolute/path/to/prepared-meeting-models"
export PAITON_RECORDING="/absolute/path/to/meeting.mp4"
export PAITON_RESULTS="$HOME/paiton-meeting-results"
export PAITON_MEETING_RUNTIME="$HOME/.cache/paiton-meeting/runtime"
mkdir -p "$PAITON_RESULTS" "$PAITON_MEETING_RUNTIME"
flock "/tmp/paiton-studio-gpu-$(id -u).lock" \
  docker run --rm --network none --device /dev/kfd --device /dev/dri \
  --group-add "$(stat -c '%g' /dev/kfd)" --user "$(id -u):$(id -g)" --shm-size 2g \
  -e HOME=/tmp -e OMP_NUM_THREADS=1 -e HF_HUB_OFFLINE=1 \
  --mount "type=bind,src=$PAITON_MODEL_DIR,dst=/models/meeting,readonly" \
  --mount "type=bind,src=$PAITON_MEETING_RUNTIME,dst=/models/cache" \
  --mount "type=bind,src=$PAITON_RECORDING,dst=/recording/input,readonly" \
  --mount "type=bind,src=$PAITON_RESULTS,dst=/results" \
  ghcr.io/eliovp/paiton-vllm-plugin:meeting-rdna4-v1.0.0-rc1 \
  process /recording/input --models /models/meeting --output /results/new-result \
  --artifact /opt/paiton/meeting-artifacts/meeting_lstm_float16_gfx1201.so
```

`new-result` must not already exist. This keeps the package's GPU lease and
network-disabled recording processing. For unprepared Hub downloads, use the
next option to create the required links and provenance record first.

### Already in the Hugging Face cache

A Hub cache alone is not the prepared meeting directory. Mount the cache into
preparation and processing at the same location so its snapshot links survive.
The following preparation reads the three pinned Hub snapshots without
redownloading them. It fetches the small pinned Silero file if that file is not
already in the prepared model directory. No recording is mounted for preparation.

```bash
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
export PAITON_MEETING_CACHE="$HOME/.cache/paiton-meeting-existing"
mkdir -p "$PAITON_MEETING_CACHE/models" "$PAITON_MEETING_CACHE/runtime"
docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HUB_OFFLINE=1 --entrypoint python3 \
  --mount "type=bind,src=$PWD/models/Meeting,dst=/prepare,readonly" \
  --mount "type=bind,src=$PAITON_MEETING_CACHE/models,dst=/models/meeting" \
  --mount "type=bind,src=$HF_HUB_CACHE,dst=/models/meeting/hub,readonly" \
  ghcr.io/eliovp/paiton-vllm-plugin:meeting-rdna4-v1.0.0-rc1 \
  /prepare/scripts/prepare.py --models /models/meeting
```

Use a fresh preparation directory if it contains incompatible role links.
All three Hub snapshots must already be complete at their locked revisions.
For a cache on another drive, set `HF_HUB_CACHE` to that absolute directory.
The preparation step is separate from recording processing; it does not need
a GPU or access to your Hugging Face token when these snapshots are complete.

After preparation succeeds, keep the extra Hub mount during processing:

```bash
export PAITON_RECORDING="/absolute/path/to/meeting.mp4"
export PAITON_RESULTS="$HOME/paiton-meeting-results"
mkdir -p "$PAITON_RESULTS"
flock "/tmp/paiton-studio-gpu-$(id -u).lock" \
  docker run --rm --network none --device /dev/kfd --device /dev/dri \
  --group-add "$(stat -c '%g' /dev/kfd)" --user "$(id -u):$(id -g)" --shm-size 2g \
  -e HOME=/tmp -e OMP_NUM_THREADS=1 -e HF_HUB_OFFLINE=1 \
  --mount "type=bind,src=$PAITON_MEETING_CACHE/models,dst=/models/meeting,readonly" \
  --mount "type=bind,src=$HF_HUB_CACHE,dst=/models/meeting/hub,readonly" \
  --mount "type=bind,src=$PAITON_MEETING_CACHE/runtime,dst=/models/cache" \
  --mount "type=bind,src=$PAITON_RECORDING,dst=/recording/input,readonly" \
  --mount "type=bind,src=$PAITON_RESULTS,dst=/results" \
  ghcr.io/eliovp/paiton-vllm-plugin:meeting-rdna4-v1.0.0-rc1 \
  process /recording/input --models /models/meeting --output /results/new-cached-result \
  --artifact /opt/paiton/meeting-artifacts/meeting_lstm_float16_gfx1201.so
```

Choose an unused result name. The ordinary `run-docker.sh` does not add this
external Hub mount, so retain the explicit command for this layout.
[Cache path guide](../../docs/MODEL_WEIGHTS.md).

## Serving interface

Keep the existing recording workflow: after [preparation](REPRODUCE.md), run
`./models/Meeting/run-docker.sh --paiton /path/to/meeting.mp4 /path/to/new-result`
from the repository root. The package manages its own stages, including its
text helper; no separately served chat endpoint is required. Existing flags,
cache locations and commands remain available. The separate `paiton serve`
command is explained in the [main launch guide](../../README.md#quick-start).


**Release candidate, 1.0.0rc1 — image published on GHCR; source branch ready for PR review.** The tested Paiton pipeline meets the declared median speed-parity gate. Studio integration is a separate task; this package runs independently through its CLI/container.

Recording import, codec handling, ASR, anonymous diarization and compiler execution have local tests. The complete CLI passed a generated spoken regression test and a repeated 39-minute meeting benchmark. Summaries are partial drafts for review against the recording.

The pipeline separates audio capture/import, Silero voice activity detection, Parakeet speech recognition, pyannote speaker diarization and a small local text summarizer. A speech recognizer does not capture Teams audio or identify participants by name. English is the primary qualification language. Parakeet v3 supports multiple languages upstream; this package has not yet qualified them all.

| Component | Pinned choice | Local implementation |
|---|---|---|
| Speech recognition | NVIDIA Parakeet TDT 0.6B v3 | Transformers, FP16 SDPA; optional Paiton prediction LSTM |
| Voice activity | Silero VAD | CPU TorchScript, silence gating without shifting the audio clock |
| Speaker turns | pyannote community-1 | Separate offline GPU stage; overlapping and uncertain turns retained |
| Partial summary | Granite 4.2 3B (3.7B actual parameters) | Compact text helper; native vLLM with a Transformers alternative, evidence-linked partial notes |

The final cached complete-pipeline median is **291.03 seconds stock / 284.50 seconds Paiton**, a **6.53-second (2.24%) reduction** on the 39:05 meeting (four runs each). The original 20% objective was not met. The launcher enables Paiton by default; `--stock` runs the matched native baseline. See [BENCHMARKS.md](BENCHMARKS.md) for variability and stage boundaries, and [RELEASE.md](RELEASE.md) for the published image digest and artifact status.

Exact revisions and licenses are in [models.lock.json](models.lock.json). Accept community-1 access conditions with your own Hugging Face account before running the download preparation step. The runtime uses cached files and has no network access. Model weights are not included in this candidate's source package.

No separate forced-alignment model is shipped: word timing and punctuation come from Parakeet, followed by speaker-turn attribution. Timestamp limitations are measured separately from WER.

## Recording import

Export a consented recording from Teams or another meeting application, then run this package locally. Tested formats include WAV, FLAC, MP3, M4A/MP4 with AAC, and Ogg/WebM with Opus. The input is read-only; choose a new output directory for each run. Select separate audio tracks/channels rather than automatically combining duplicate microphone/system feeds. Decoding is bounded to eight hours; supported containers can still contain unsupported codecs.

```bash
# After the one-time model preparation and local image build:
./run-docker.sh --paiton /path/to/meeting.mp4 /path/to/new-result
# Matched stock comparison, using a different output directory:
./run-docker.sh --stock /path/to/meeting.mp4 /path/to/new-stock-result
```

Results contain `result.json` (timestamped transcript, anonymous speakers, partial notes and source references) and `playback.wav`. [REPRODUCE.md](REPRODUCE.md) contains the complete setup, cache, launch and export instructions. No Studio installation or manually managed vLLM server is required.

This package processes imported recordings. It does not capture Teams audio or provide browser recording, Studio integration or a streaming transcript.

## Local stage commands

See [REPRODUCE.md](REPRODUCE.md) for pinned model preparation, the required compiled-artifact build context and the one-command recording workflow. The base image is pinned by digest, and added Python dependencies are pinned in `requirements.lock`. The candidate runs under an unprivileged UID. Keep `/models/cache` persistent and mount original recordings read-only.

The candidate image defaults to `vllm` for `process` and checkpoint-based `summarize`; pass `--summary-backend transformers` for the alternate helper. Direct source execution defaults to Transformers unless `PAITON_MEETING_SUMMARY_BACKEND` is set. vLLM uses native BF16 Granite, Triton attention and graph caching; audio components remain outside vLLM.

The package CLI provides one-command `process`, plus `inspect`, `normalize`, `transcribe`, `diarize`, `attribute`, `summarize`, `export` and `benchmark-asr`; run `python -m paiton_meeting --help` for their arguments. Separate ASR and diarization processes release their GPU allocations before the summary model starts. The standalone launcher uses a host GPU lease to avoid colliding with cooperating local workloads.

`benchmark-asr` records one first-processing pass plus at least three warm runs, transcripts, raw timings and sampled driver/allocator telemetry. RTF means processing seconds divided by audio seconds; its reciprocal is audio hours processed per wall-clock hour. This command measures the ASR stage, not end-to-end meeting processing. See [BENCHMARKS.md](BENCHMARKS.md) for measured model switching, summary latency and image-specific limitations.

JSON, plain transcript, SRT, VTT and plain-text summary exports preserve source IDs/timestamps. Speaker names are user-entered labels; no voice-based identity recognition is claimed. Summary owners and deadlines remain unspecified unless supported by the transcript. Summary generation treats spoken instructions as untrusted data and rejects invalid references or incomplete model output. Hierarchical reduction carries source quotes and IDs; a separate local audit checks claims against nearby transcript context. Unresolved-issue entries additionally require an explicit question or uncertainty cue in the cited English text, so facts and suggestions are conservatively omitted from that category. These checks do not prove factuality: outputs remain partial drafts for review against playback. This initial version accepts incomplete coverage; the full transcript remains available. No measured 60–75% recall claim is made.

Recordings, transcripts and summaries remain local. Delete a CLI output directory to remove its derived content; the original recording is preserved. Intermediate stage files are removed unless `--keep-intermediates` is requested. No meeting content is sent to an external API.

## Qualification results so far

On the 39:05 AMI ES2004b distant-microphone recording, three matched warm FP16 ASR runs measured median 87.44 s stock and 77.32 s with the Paiton decoder: **11.58% lower ASR wall time**. WER was 20.42% and 20.44%, respectively. This is not a 20% complete-pipeline result. Word ordering during overlap and unannotated room speech affect the reference comparison.

Pyannote's first offline run took 77.53 s. With original manual utterance labels, a 250 ms collar, the complete recording interval and overlap included, DER was 14.90%, versus 49.21% for the rejected Sortformer candidate. This protocol differs from forced-alignment model-card benchmarks. Attribution remains imperfect, particularly around overlap and very short interjections.

Granite Speech 5.0 470M TurboCTC executes in BF16 and FP32 on gfx1201. It was substantially faster but less accurate than Parakeet on five matched meeting excerpts. On 20 clean LibriSpeech clips (one speaker), Parakeet/Granite WER was 2.46%/3.79%; with deterministic added white noise at 10 dB SNR, it was 7.37%/4.91%. This narrow comparison does not establish broad superiority. Parakeet remains the meeting ASR choice.

Voxtral Mini (4.7B total) executed locally but hallucinated during raw silence and failed the correction/instruction-attack summary fixture. It is not the default. VAD protects the selected ASR pipeline from processing silent chunks, but does not establish general hallucination immunity.

The current standalone compiled command processed the 38.90-second synthetic recording in 120.66 seconds including startup. It retained the USB-C decision, Morgan/Tuesday action and undecided launch, with valid source references, and rejected injected commitments. [Actual CLI transcript and notes](examples/spoken-timestamp-runtime.json). This short first-use test is not a stock/Paiton comparison or broad summary/speaker accuracy evidence.

A short excerpt from that actual output:

| Partial notes | Transcript reference |
|---|---|
| Decision: use USB-C for the charging connector. | `s000001`, 00:00.00 |
| Action: Morgan will send the connector drawing by Tuesday. | `s000002`, 00:03.44 |
| Open issue: the launch date remains undecided. | `s000007` |

The compact Granite adapter uses bounded context and hierarchical summaries. Large Qwen3.8 and GPT-OSS models are excluded from this small configuration. Summary coverage is intentionally accepted as partial for this initial version; included claims still need review against source timestamps. See [SELECTION.md](SELECTION.md), [QUANTIZATION.md](QUANTIZATION.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Licenses and attribution

Package integration code follows this repository's Apache-2.0 license. NVIDIA Parakeet v3 and pyannote community-1 are CC-BY-4.0 checkpoints; Silero VAD is MIT; Voxtral Mini, Granite 4.2 and Granite Speech TurboCTC are Apache-2.0. Consult each upstream model card and the inherited runtime's notices. Gated access is separate from copyright licensing, and this package does not accept access conditions on a user's behalf.

The public AMI evaluation recording and annotations are from the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/), distributed under CC-BY-4.0. Cite the AMI corpus creators when redistributing those recordings or derived examples. No private meeting audio or derived content belongs in release evidence.
