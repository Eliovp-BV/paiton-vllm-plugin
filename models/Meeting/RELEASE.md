# Meeting package — local release review

Version: `meeting-rdna4-v1.0.0-rc1`. Prepared locally; **not pushed or published**. Existing releases and tags are preserved.

## Proposed publication targets

- Community source and instructions: `Eliovp-BV/paiton-vllm-plugin`, separate `models/Meeting` entry.
- Container: `ghcr.io/eliovp/paiton-vllm-plugin:meeting-rdna4-v1.0.0-rc1`.
- Release asset: `meeting-rdna4-v1.0.0-rc1-artifacts.tar.gz`, containing the compiled prediction LSTM, manifest and license notices. No compiler source or model weights.

The local image ID is `sha256:f7e6a83e38d41f0c89fc8175e8cf5d0080b333ea0a802d52369360069612e986`. This is the Docker image/config identity, **not a registry manifest digest**. The frozen inference source revision is `90cd1b3cce3298ed9d8ddd0e544245b861451eaa`; later source changes prepare instructions, evidence and launcher defaults.

The artifact archive SHA-256 is `6c3886ecd28b53c2bdd70811018d84f0562baf19df9211bc2c461ce560d80523` (119,247 bytes). The binary hash and ABI are pinned in [artifact.lock.json](artifact.lock.json). Model revisions and separate licenses are pinned in [models.lock.json](models.lock.json).

## Use the locally prepared image

From `models/Meeting`, after preparing the model cache as described in [REPRODUCE.md](REPRODUCE.md):

```bash
export PAITON_MEETING_IMAGE=ghcr.io/eliovp/paiton-vllm-plugin:meeting-rdna4-v1.0.0-rc1
./run-docker.sh --paiton /path/to/meeting.mp4 /path/to/new-result
```

The launcher uses `--pull=never`: the image must already exist locally. A new user's pull command becomes usable only after a separately approved publication:

```bash
# After publication only; currently unavailable from GHCR:
docker pull ghcr.io/eliovp/paiton-vllm-plugin:meeting-rdna4-v1.0.0-rc1
```

Results contain the transcript, speaker labels and referenced partial notes in `result.json`, plus `playback.wav`. Export instructions, persistent caches, track selection, speaker aliases and deletion are documented in [REPRODUCE.md](REPRODUCE.md).

## Review evidence

- [Measured performance and limitations](BENCHMARKS.md).
- [Actual synthetic transcript and notes from this image](examples/spoken-timestamp-runtime.json).
- [Model selection](SELECTION.md), [quantization qualification](QUANTIZATION.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

English recording import is qualified on the documented fixtures. Speaker clustering is imperfect. Notes are partial and can omit final decisions; review included claims against the referenced transcript. No standalone live Teams capture or streaming transcript is claimed. Studio integration is outside this package's scope.
