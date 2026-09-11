# Reproduce the local meeting candidate

Linux, Docker and an accessible gfx1201 Radeon AI PRO R9700 are required. All inference runs locally with Docker networking disabled. The current image is a local candidate; the proposed GHCR tag is not published.

## Standalone user workflow

This is a CLI/container package in `paiton-vllm-plugin`; no Studio installation or running vLLM HTTP server is required. Audio stages run directly and the package starts native vLLM internally for the summary. Publication is blocked until matched Paiton end-to-end performance is at least as fast as stock. The prior candidate failed that gate; the new loader candidate is still being qualified.

1. Prepare the pinned model cache once, with your own approved community-1 access.
2. Build the local image with the reviewed binary overlay. After an approved release, users can pull its versioned image instead; the proposed tag is not available yet.
3. Run `./run-docker.sh --paiton meeting.mp4 new-result`. For comparison, use `--stock` and a different output directory.
4. Review `new-result/result.json` and `new-result/playback.wav`; export transcript/subtitles/notes with the `export` subcommand.

Record or export the meeting using your meeting application or an authorized local recorder first. This standalone package imports recordings; it does not capture Teams audio. Obtain the applicable recording consent.

## Prepare persistent models

Run these commands from `models/Meeting`. Accept the community-1 access conditions with your own Hugging Face account. Preparation downloads model files only; it never accesses recordings. A small separate environment provides the pinned download client:

```bash
python3 -m venv "$HOME/.cache/paiton-meeting-download-env"
source "$HOME/.cache/paiton-meeting-download-env/bin/activate"
python -m pip install "huggingface_hub==1.23.0"
hf auth login
export PAITON_MEETING_CACHE="$HOME/.cache/paiton-meeting"
python scripts/prepare.py --models "$PAITON_MEETING_CACHE/models"
```

The script uses the exact revisions and download patterns in `models.lock.json`, creates persistent snapshot links and records file hashes. It preserves existing checkpoints and refuses to replace a different role directory. Use a new cache location if you already have incompatible files. Gated credentials are needed for preparation only; do not pass them to inference containers.

## Build the image

The named `meeting_overlay` context must contain the release-provided `meeting_lstm_float16_gfx1201.so` and adjacent JSON manifest. The manifest pins its SHA-256, C ABI, dtype, shape and target. The candidate overlay is prepared locally; approved release distribution is pending. It contains compiled artifacts, not compiler source.

```bash
export PAITON_MEETING_OVERLAY=/path/to/approved-meeting-overlay
export PAITON_MEETING_MEDIA_SOURCES="$PAITON_MEETING_CACHE/media-sources"
python scripts/prepare_media_sources.py --output "$PAITON_MEETING_MEDIA_SOURCES"
docker build --build-context "meeting_media_sources=$PAITON_MEETING_MEDIA_SOURCES" \
  --build-context "meeting_overlay=$PAITON_MEETING_OVERLAY" \
  -t paiton-meeting:local-candidate .
export PAITON_MEETING_IMAGE=paiton-meeting:local-candidate
```

The Dockerfile pins its inherited runtime digest and added dependencies. It builds PyAV 18.1.0 against unmodified FFmpeg 8.1.2 with GPL/nonfree components and networking disabled, retaining LAME and Opus for audio codecs. It installs that wheel before the Python requirements, avoiding the upstream binary wheel and its bundled video encoders. The six source/build inputs are hash-pinned in `media-sources.lock.json`; source archives, build recipe and component notices are retained under `/opt/meeting-media/share/paiton-media` in the image. The source-built runtime passed all 42 package tests and produced bit-identical decoded samples for 23 cases, including the complete 39-minute AMI recording. The artifact loads through a plain C ABI without importing Torch; tensor handling stays in the Python integration layer. No checkpoint is modified or embedded in this image.

The reviewed artifact hash and compiler build revision are recorded in `artifact.lock.json`. Verify its dependency closure and standalone loading without GPU execution or a Torch import:

```bash
docker run --rm --network none --entrypoint python3 \
  -v "$PWD/scripts:/checks:ro" "$PAITON_MEETING_IMAGE" \
  /checks/check_artifact.py \
  /opt/paiton/meeting-artifacts/meeting_lstm_float16_gfx1201.so
```

## Process a recording

```bash
./run-docker.sh /path/to/meeting.mp4 /path/to/new-meeting-result
./run-docker.sh --paiton /path/to/meeting.mp4 /path/to/new-paiton-result
./run-docker.sh /path/to/meeting.mp4 /path/to/new-transformers-result --summary-backend transformers
```

Stock ASR is the default because it had the lower measured complete-pipeline median. `--paiton` explicitly enables the compiled prediction LSTM; `--stock` remains an accepted explicit stock selector. Each output directory must be new. Results contain `result.json` and synchronized `playback.wav`. Original audio is mounted read-only. ASR, diarization and summary execute sequentially in separate processes, releasing GPU allocations between stages. The launcher cooperates with Studio's GPU lease and uses a persistent runtime cache. First use can take longer while ROCm kernels initialize.

For separate sources, pass `--track 1` or `--channel 0` as appropriate. Track numbering is zero-based among audio streams. Do not downmix duplicated microphone/system feeds. All original tracks remain in the source recording. `--keep-intermediates` retains stage results for local diagnosis or benchmarking; otherwise they are removed. Delete the output directory to remove CLI-derived content; the source recording is preserved.

The CLI also exposes `inspect`, `normalize`, `transcribe`, `diarize`, `attribute`, `summarize`, `export` and `benchmark-asr`. Run the image with `--help`, or a subcommand with `--help`, to inspect arguments. `export` supports JSON, TXT, SRT, VTT and summary text. Summary output is a partial draft with source references; absence of an extracted decision is not evidence that no decision occurred.

To export readable notes and subtitles from an existing result, mount that result directory and run the same image (no GPU is needed for export):

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "/path/to/new-result:/results" "$PAITON_MEETING_IMAGE" \
  export /results/result.json --format summary.txt --output /results/notes.txt
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "/path/to/new-result:/results" "$PAITON_MEETING_IMAGE" \
  export /results/result.json --format srt --output /results/transcript.srt
```

Export refuses to replace an existing file. Use `--format txt` for a plain transcript or `--format vtt` for WebVTT. Anonymous labels describe speaker clusters, not verified participant identities. To name a known speaker, add a top-level `"speaker_names": {"SPEAKER_00": "Morgan"}` mapping to a copy of `result.json`, then export that copy. Transcript and subtitle exports use these aliases while the underlying speaker IDs and timestamps remain available. Names are supplied by the user; they are not recognized from voices.

For a complete-pipeline comparison, the host-side harness runs an initial pair and three alternating cached repetitions of each variant. It preserves raw logs, stage results and sampled host/driver memory. Allow enough time for eight full processing runs; all use the shared GPU lease.

```bash
python benchmark/complete_pipeline.py --launcher "$PWD/run-docker.sh" \
  --recording /path/to/meeting.wav --output /path/to/new-benchmark --repeats 3
```

Specify `--summary-backend vllm` on the benchmark command to match the candidate image default consistently across both ASR variants. The Linux harness also samples the named owned container's process RSS; summed RSS can double-count shared pages.

Launcher wall time can include waiting for another workload. The reported pipeline time starts after lease acquisition and excludes Docker startup; individual stage times include interpreter/model startup and switching. This serial deployment is intentionally separate from a resident-model warm ASR microbenchmark.


## Investigate summary startup

The native helper now explicitly prefetches checkpoint pages before loading weights. The same setting applies to stock and Paiton. This does not alter precision or generation settings. Three alternating loader probes measured median initialization of 41.10 seconds with lazy loading and 34.41 seconds with prefetch; this is not a complete-pipeline speedup claim. Full results are in [benchmark/summary-loader-comparison.json](benchmark/summary-loader-comparison.json).

The reproducible loader probe accepts `--model`, `--transcript` (a CLI `result.json`), `--strategy lazy|prefetch`, and a new `--output` file. Run it inside the prepared GPU container with the benchmark directory mounted, using `python /benchmark/summary_loader.py`. It generates exactly 128 tokens three times per fresh engine and records token hashes; those probes are intentionally incomplete summaries. Use identical input and cache paths and alternate strategies across separate processes.

## Separate Studio work

Studio is not a dependency or a deliverable of this standalone package task. Existing integration notes are preserved in [STUDIO_HANDOFF.md](STUDIO_HANDOFF.md) for the separate implementation task.
