# Why this small modular pipeline

Research snapshot: 2026-09-10 UTC. English is the primary local qualification language. Checkpoint releases below are distinct from later quantization uploads and repository updates.

| Component | Base release | Local finding |
|---|---|---|
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | August 2025 | Selected dedicated multilingual ASR; timestamps, strong English meeting accuracy and measured gfx1201 execution |
| [Granite Speech 5.0 470M TurboCTC](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) | August 25, 2026 | Very fast English ASR; less accurate on all five meeting excerpts; meaningful future speed option, not enabled by default |
| [Voxtral Mini 3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | July 15, 2025 | Actual total 4.68B parameters including audio encoder; executed locally but hallucinated on silence and failed summary correction/injection probes |
| [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | 2025; exact release day not established here | Selected anonymous speaker diarization; substantially lower local DER than the tested Sortformer alternative |
| [Granite 4.2 3B](https://huggingface.co/ibm-granite/granite-4.2-3b) | August 25, 2026 | Selected compact text helper, actual 3.66B parameters; partial evidence-linked summaries with a separate local claim check |

The exact Granite Speech checkpoint had 21,361 monthly Hub downloads and 61 likes on the research date. Voxtral Mini had 453,224 and 675; Granite 4.2 3B had 26,634 and 86; community-1 had 5,024,921 and 1,556. These are dated checkpoint counts, not family popularity or claims about every quantization. Community-1's April 15 repository creation and September 29 update are not treated as its base-release date; its [upstream announcement](https://www.pyannote.ai/blog/community-1) describes the release alongside pyannote.audio 4.0. Parakeet uses native Transformers TDT support; pyannote uses its maintained audio pipeline; Granite uses native Transformers causal-language-model support. The ASR and diarization stages do not require vLLM.

The speech model handles audio. VAD detects speech intervals. Diarization estimates who spoke when using anonymous clusters. A separate compact text model writes notes from timestamped transcript segments. None of these components captures Teams audio, verifies participant identities or substitutes for recording permissions.

The pipeline preserves the original audio clock and TDT timestamps without adding a separate forced-alignment model. Punctuation comes from ASR output. Extra alignment could improve selected cases but is not part of the measured profile. Overlap and uncertain speaker attribution are retained; labels can be renamed by users who know the participants.

The summary helper receives bounded chronological batches, reduces notes with original references and revisits the latest transcript batch for corrections. It then checks extracted claims against nearby source context. This is a partial-draft workflow: it can omit decisions or fail to recover an owner from a nearby turn. Included claims remain reviewable against playback. Model self-checking does not prove factuality or broad prompt-injection resistance.

The audit includes 30 seconds around cited speech because diarization can split one utterance into many small fragments. An explicit prior-meeting decision recap in that context suppresses the extracted decision before the model audit: the small model otherwise accepted an exact quote despite its historical qualifier. This conservative rule can also omit a new decision made immediately after a recap. That recall cost is accepted for the initial partial-draft scope and is not described as perfect temporal reasoning.

On a generated spoken test, the complete pipeline preserved Morgan's connector-drawing task and Tuesday deadline, retained the undecided launch issue, and did not accept an injected fictional budget approval or turn an email invitation into a task. It omitted the explicit USB-C decision. That omission is disclosed and accepted for the initial partial-summary scope; it is not counted as complete decision coverage.

Upstream multilingual support is not equivalent to local multilingual qualification. The current package's measured quality evidence is English. Real regional accents, unfamiliar participant names and dense overlapping speech remain important limitations. Browser capture and the native Teams application require separate validation from exported-recording import.
