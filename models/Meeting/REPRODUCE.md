# Reproduce the local meeting candidate

Linux, Docker and an accessible gfx1201 Radeon AI PRO R9700 are required. All inference runs locally with Docker networking disabled. The current image is a local candidate; the proposed GHCR tag is not published.

## Prepare persistent models

Accept the community-1 access conditions with your own Hugging Face account, then authenticate with `hf auth login`. Preparation downloads model files only; it never accesses recordings. Run from this model directory in an environment containing `huggingface_hub`:

```bash
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
  -t paiton-meeting:studio-candidate .
export PAITON_MEETING_IMAGE=paiton-meeting:studio-candidate
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

For a complete-pipeline comparison, the host-side harness runs an initial pair and three alternating cached repetitions of each variant. It preserves raw logs, stage results and sampled host/driver memory. Allow enough time for eight full processing runs; all use the shared GPU lease.

```bash
python benchmark/complete_pipeline.py --launcher "$PWD/run-docker.sh" \
  --recording /path/to/meeting.wav --output /path/to/new-benchmark --repeats 3
```

Specify `--summary-backend vllm` on the benchmark command to match the candidate image default consistently across both ASR variants. The Linux harness also samples the named owned container's process RSS; summed RSS can double-count shared pages.

Launcher wall time can include waiting for another workload. The reported pipeline time starts after lease acquisition and excludes Docker startup; individual stage times include interpreter/model startup and switching. This serial deployment is intentionally separate from a resident-model warm ASR microbenchmark.

## Studio configuration

Use the coordinated Studio feature version with meeting controls. Set these keys in that instance's local configuration, using paths accessible to its Docker daemon:

```json
{
  "meeting_enabled": true,
  "meeting_compiler_enabled": false,
  "meeting_image": "paiton-meeting:studio-candidate",
  "meeting_models_dir": "/path/to/paiton-meeting/models"
}
```

Set `meeting_compiler_enabled` to `true` only to opt into the tested Paiton ASR path. Each queued job keeps its selected backend; changing configuration does not silently change an already queued job. Completed Studio exports record `asr_backend`. Studio verifies the prepared provenance receipt before queuing inference. Open Meetings, import a recording, select the audio source, then choose **Transcribe and summarize locally**. The queue shares the GPU with existing Studio tools. Review anonymous speaker labels, enter known participant names, follow timestamp links, export the result or delete the owned recording and derived content.

For a browser on another machine, configure `PAITON_TLS_CERT` and `PAITON_TLS_KEY` with a certificate trusted by that client and launch `run-meetings-secure.sh`. The default port is 8877; `PAITON_STUDIO_PORT` overrides it. The server's microphone is not the remote browser's microphone. Recording import is separately validated from browser capture; see the capture table in the README.

## Studio API

The coordinated Studio version exposes the same recording workflow under `/api/meetings`. First request `GET /api/session`, retain its `studio_session` cookie, and send the returned token in `x-studio-token` for mutations. Keep that token local; it is not a model-download credential. Remote clients must use the configured trusted HTTPS endpoint.

| Method and path | Purpose / body |
|---|---|
| `POST /api/meetings` | Create an import: `{"name":"Design review","filename":"meeting.mp4","source":"import","retention":"keep"}` |
| `POST /api/meetings/{id}/chunks/{index}` | Upload ordered, zero-based multipart `file` chunks, each at most 4 MiB; identical retries are accepted |
| `POST /api/meetings/{id}/finish` | Finalize and inspect the recording after the last chunk |
| `POST /api/meetings/{id}/process` | Queue local transcription, speaker attribution and notes: `{"track":0,"channel":null}` |
| `GET /api/meetings/{id}` | Poll state and retrieve the completed transcript/notes |
| `GET /api/meetings/{id}/recording` | Playback on the transcript clock after processing |
| `GET /api/meetings/{id}/export` | Export recording metadata, transcript, notes and references as JSON |
| `POST /api/meetings/{id}/cancel` | Cancel the owned processing job |
| `POST /api/meetings/{id}/speakers/{speaker}` | Rename an assigned anonymous cluster: `{"name":"Morgan"}` |
| `POST /api/meetings/{id}/retention` | Set `{"retention":"delete-recording-after-processing"}` or `"keep"` |
| `DELETE /api/meetings/{id}` | Delete the owned recording and derived content when no job is active |

For a recording already imported through Studio, this standard-library Python command queues processing without printing the session credential. Obtain the recording ID from `GET /api/meetings` or its existing import response. Processing uses the configured local image and the shared GPU queue.

```bash
export PAITON_STUDIO_URL=http://127.0.0.1:8877
export PAITON_MEETING_ID=recording-id-from-import
python3 - <<'PY'
import http.cookiejar, json, os, urllib.request
base = os.environ['PAITON_STUDIO_URL'].rstrip('/')
identity = os.environ['PAITON_MEETING_ID']
client = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
with client.open(base + '/api/session') as response:
    token = json.load(response)['token']
request = urllib.request.Request(
    base + '/api/meetings/' + identity + '/process',
    data=json.dumps({'track': 0, 'channel': None}).encode(),
    headers={'Content-Type': 'application/json', 'x-studio-token': token},
    method='POST')
with client.open(request) as response:
    result = json.load(response)
print(json.dumps({key: result[key] for key in ('id', 'job', 'state')}))
PY
```

This API submits imported recordings; it does not capture a remote microphone or Teams audio. Start browser capture visibly in Studio, or import a complete authorized recording. For headless batch use, the `run-docker.sh` command above avoids the Studio HTTP/session layer.
