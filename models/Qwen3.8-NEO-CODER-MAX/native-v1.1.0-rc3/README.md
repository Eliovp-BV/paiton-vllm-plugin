# Native v1.1.0-rc3: unqualified experiment

This local 2,048-token prefill experiment is **not qualified for deployment**.
Do not use its faster HTTP timings as a qualified performance claim.

Although its fixed text/image checks and generated token IDs matched the prior
candidate, a full-vocabulary comparison after 4,096-token prefill exceeded the
preset numerical bounds during 128-token decode: maximum logit difference
3.825344 and minimum cosine similarity 0.987213 (limits 0.1 and 0.9999).
The standalone native consumer reproduced the API log probabilities within
1.4e-6, so the discrepancy was not caused by the external test harness.

The larger rocBLAS GEMM shape changes reduction order. An isolated experiment
retaining 1,024-row GEMMs inside 2,048-token prefill restored all 31,784,960
compared logits exactly. This correction is evaluated in the separately
identified rc4 candidate. The qualified [rc2 profile](../native-v1.1.0-rc2/README.md)
remains available. No image was published and existing release defaults remain
unchanged.
