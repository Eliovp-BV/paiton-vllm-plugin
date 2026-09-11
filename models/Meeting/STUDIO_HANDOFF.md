# Separate Studio integration handoff

Historical integration notes are preserved for the separate Studio task. The standalone meeting package does not require Studio. Further Studio implementation is outside this task.

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

## Preserved prototype qualification history

The complete command processed a 38.90-second synthetic spoken regression recording in 154.00 seconds on first use, including stage startup. Studio processed it in 145.82 seconds using its actual GPU queue. Both produced the same supported Morgan/Tuesday action and undecided-launch issue without accepting the fictional injected budget or email invitation as commitments. The explicit USB-C decision was omitted. Browser checks covered import, completion, speaker renaming, synchronized playback seeking and JSON export with no page errors. These short first-use results are not long-meeting throughput measurements or a broad factuality score.

The final stock-default Studio check produced [this transcript and partial summary](examples/spoken-stock-default.json), retaining the USB-C decision, Morgan/Tuesday action and undecided launch with source references. Import, completion, speaker renaming, playback seeking and export passed without browser errors. The earlier compiled-ASR Studio example remains available separately.
