# Meeting package notices

Integration code follows this repository's Apache-2.0 license. Paiton generated runtime artifacts are provided separately from private compiler source, under the repository's applicable runtime distribution terms. Preserve inherited runtime notices when building or redistributing the image.

| Downloaded component | Copyright attribution / upstream | License |
|---|---|---|
| Parakeet TDT 0.6B v3 | NVIDIA; model card and exact revision in `models.lock.json` | CC-BY-4.0 |
| Speaker diarization community-1 | pyannote; model card and exact revision in `models.lock.json` | CC-BY-4.0; gated download conditions also apply |
| Granite 4.2 3B | IBM Granite; model card and exact revision in `models.lock.json` | Apache-2.0 |
| Silero VAD | Silero Team; pinned `snakers4/silero-vad` artifact | MIT |

The image contains recipes and integration code, not these model weights. The download cache retains upstream model cards; the preparation manifest records revisions and file hashes. Model attribution must accompany redistributed checkpoints or derivatives. Gated access conditions are accepted by the downloading user, separately from copyright licensing. This package does not redistribute account credentials.

The inherited image contains ROCm, PyTorch, Transformers, vLLM and other runtime components. Their licenses and notices remain in the inherited layers and this repository's top-level `LICENSES` and `THIRD_PARTY_NOTICES.md`. Added pyannote, PyAV and numerical dependencies are pinned in `requirements.lock`; retain their installed distribution license metadata. ASR and diarization execute outside vLLM.

Public evaluation uses the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) recording ES2004b and its annotations under CC-BY-4.0. Attribute the AMI corpus creators, including the corpus description by Carletta et al., when redistributing audio, references or derived examples. LibriSpeech clean/noise probes use [OpenSLR resource 12](https://www.openslr.org/12), CC-BY-4.0, via a pinned small test subset. Added deterministic noise is an evaluation modification. The small subset contains one speaker and is not a broad accent benchmark.

The generated spoken regression fixture contains newly authored test text rendered with eSpeak NG. It uses synthetic voices, not recordings or clones of participants. Its text and generated test audio are designated CC0; the test-only eSpeak NG software retains its GPL license and is not added to the inference image. Generated fixtures are not substitutes for natural speech evaluation.

The supplementary multilingual samples and references come from [Google FLEURS](https://huggingface.co/datasets/google/fleurs), CC-BY-4.0, with attribution to its dataset authors. The exact dataset revision and selected file hashes are in `benchmark/fleurs-manifest.json`. References are case/punctuation-normalized for WER/CER comparison; recognition outputs are generated derivatives. The small selection is for reproducible evaluation and is not presented as a representative meeting benchmark.

No private recording, transcript or summary is included in public release evidence. Example uploads and publication remain subject to explicit approval.

The candidate applies one pinned-source change to vLLM's optional TorchCodec video import: an unavailable native video library (`OSError`) follows its existing unavailable-backend fallback. The inherited CUDA TorchCodec wheel is not used for meeting audio; CPU PyAV decoding is unchanged. The patch does not add CUDA libraries. vLLM retains its Apache-2.0 license and contributor notices.
