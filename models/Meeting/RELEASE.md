# Meeting package — release candidate

Version: `meeting-rdna4-v1.0.0-rc1`. **Image published on GHCR; community source branch ready for PR review.** Existing releases and tags are preserved.

## Distribution

- Community source and instructions: `Eliovp-BV/paiton-vllm-plugin`, separate `models/Meeting` entry.
- Container: `ghcr.io/eliovp/paiton-vllm-plugin:meeting-rdna4-v1.0.0-rc1`.
- Local-only archive (not uploaded): `meeting-rdna4-v1.0.0-rc1-artifacts.tar.gz`, containing the compiled prediction LSTM, manifest and license notices. No compiler source or model weights.

The published OCI index digest is `sha256:f7e6a83e38d41f0c89fc8175e8cf5d0080b333ea0a802d52369360069612e986`. It was verified against GHCR with anonymous manifest access and matches the locally tested image identity. The frozen inference source revision is `90cd1b3cce3298ed9d8ddd0e544245b861451eaa`; later source changes prepare instructions, evidence and launcher defaults.

The artifact archive SHA-256 is `6c3886ecd28b53c2bdd70811018d84f0562baf19df9211bc2c461ce560d80523` (119,247 bytes). The binary hash and ABI are pinned in [artifact.lock.json](artifact.lock.json). Model revisions and separate licenses are pinned in [models.lock.json](models.lock.json).

## Use the published image

From `models/Meeting`, after preparing the model cache as described in [REPRODUCE.md](REPRODUCE.md):

```bash
export PAITON_MEETING_IMAGE=ghcr.io/eliovp/paiton-vllm-plugin:meeting-rdna4-v1.0.0-rc1
docker pull "$PAITON_MEETING_IMAGE"
./run-docker.sh --paiton /path/to/meeting.mp4 /path/to/new-result
```

The launcher uses `--pull=never`: pull the image once before processing. Model weights remain separate pinned downloads.

Results contain the transcript, speaker labels and referenced partial notes in `result.json`, plus `playback.wav`. Export instructions, persistent caches, track selection, speaker aliases and deletion are documented in [REPRODUCE.md](REPRODUCE.md).

## Review evidence

- [Measured performance and limitations](BENCHMARKS.md).
- [Actual synthetic transcript and notes from this image](examples/spoken-timestamp-runtime.json).
- [Model selection](SELECTION.md), [quantization qualification](QUANTIZATION.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

English recording import is qualified on the documented fixtures. Speaker clustering is imperfect. Notes are partial and can omit final decisions; review included claims against the referenced transcript. No standalone live Teams capture or streaming transcript is claimed. Studio integration is outside this package's scope.
