# Precision qualification on gfx1201

The selected Parakeet path uses FP16 weights and activations. The summary helper uses BF16. Quantization is not required to fit the sequential pipeline on the 32 GiB R9700.

Five matched 30-second AMI excerpts were tested with stock FP16, stock BF16 and encoder-only INT8 weights. Each probe ran twice; warm generation is reported separately from loading and feature extraction. These short probes do not establish a complete-pipeline ranking.

| Parakeet implementation | Peak allocated bytes | Warm generation range | Outcome |
|---|---:|---:|---|
| Stock FP16 | 1,432,969,216 | 0.197–0.322 s | Selected and qualified on the full recording |
| Stock BF16 | 1,457,582,592 | 0.189–0.292 s | Same excerpt WER as FP16; not the full-recording compiler comparison |
| INT8 encoder weights, FP16 activations | 937,552,896 | 1.076–1.096 s | Empty speech output on all five excerpts; rejected |
| INT8 encoder weights, BF16 activations | 962,157,056 | 0.195–0.314 s | Runs, but slower than stock BF16 and slightly changes errors |

The experimental conversion used `torchao==0.17.0`, `Int8WeightOnlyConfig`, per-row symmetric INT8 weights on 217 encoder `Linear` modules. It needs no calibration corpus. Decoder, convolutions, normalization, feature extraction and non-linear operators were excluded. Original checkpoints were preserved; conversion occurred on an in-memory copy. The FP16 failure's numerical cause has not been established. CUDA extension availability was not used as evidence of AMD execution: these results came from actual gfx1201 inference.

INT8 is not shipped as a supported profile. INT4 audio conversion is not qualified. The approximately half-gigabyte allocation saving did not justify worse throughput or failed transcription. No quantization speedup is attributed to the compiler.

The compiler comparison keeps the exact Parakeet revision, FP16 dtype, SDPA backend, VAD, 30-second windows, overlap, greedy TDT decoding and timestamps matched. Only the prediction LSTM execution changes. Reproduction inputs and deployment choices are pinned in `models.lock.json`; no converted checkpoint is redistributed.
