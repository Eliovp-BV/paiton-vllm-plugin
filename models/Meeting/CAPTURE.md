# Recording sources and capture limits

Recording import is the validated path for complete meetings. Export or upload an authorized recording that contains the voices you need. A microphone captures its input; shared browser audio can omit your own microphone. Studio currently records one selected source at a time. It does not combine microphone and shared audio into a complete live meeting recording.

## Browser and operating system

The following separates upstream capability reports from local testing. Browser prompts and returned audio tracks are authoritative for a particular session; requesting `systemAudio: "include"` is only a hint. [MDN capture options](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getDisplayMedia).

| Client | Upstream shared-audio capability report | Local qualification |
|---|---|---|
| Chrome/Chromium or Edge on Linux | Tab audio; entire-system audio is not reported by the compatibility snapshot | Chromium 151 simulated microphone and local HTTPS checks only |
| Chrome/Edge on Windows or ChromeOS | Tab audio; system audio may be offered for an entire-screen share | Not tested here |
| Chrome/Edge on macOS | Tab audio in the compatibility snapshot; do not assume native-application/system capture | Not tested here |
| Firefox or Safari desktop | Display capture does not imply display-audio support; the snapshot reports no display audio | Not tested here; recording import remains the recommended path |

These reports come from the [MDN browser compatibility data](https://github.com/mdn/browser-compat-data/blob/main/api/MediaDevices.json), inspected on 2026-09-10. They are not guarantees about every version or OS configuration. Chrome's [sharing controls](https://developer.chrome.com/docs/web-platform/screen-sharing-controls) explain how the selected surface and audio checkbox determine the offered stream. Studio checks for an actual audio track and fails clearly when none is returned.

Microphone APIs also require browser permission and a supported input/recording codec. Studio negotiates WebM/Opus, Ogg/Opus or MP4 audio. Only the stated Chromium simulation has exercised browser recording here; codec import tests are separate from Safari/Firefox/device tests.

## Practical workflow

1. Obtain participants' consent and follow your organization's recording rules.
2. For a complete meeting, import its authorized recording. Select the intended track/channel when the source contains separate feeds. Preserve the original to retain all tracks; avoid downmixing duplicated feeds.
3. For a browser recording, select microphone or shared audio and press **Start recording**. With shared audio, choose the meeting's browser tab and enable audio if offered. Tab sharing does not capture a native Teams application, and neither source guarantees every participant is present.
4. Check the visible recording state, then stop explicitly. Processing starts after recording/upload completion; there are no streaming transcript previews. A browser may require a video sharing surface, but Studio's recorder receives only the audio tracks.
5. Review playback before relying on the transcript. Use import when the required voices are absent or the browser offers no audio.

Studio can serve a browser on another machine. The server's ALSA devices cannot capture that client's microphone or loopback audio. The inspected server has ALSA capture nodes but no identified desktop Teams session or tested system-audio capture service. No native Teams or OS loopback integration is claimed.

Remote browser capture needs HTTPS with a certificate trusted by that client; plain remote HTTP is insufficient. Use Studio's `run-meetings-secure.sh` with `PAITON_TLS_CERT` and `PAITON_TLS_KEY`. The launcher passed a local certificate-verified request and a Chromium secure-context check with a pinned test certificate. A real remote client's certificate trust and physical audio paths remain unvalidated.
