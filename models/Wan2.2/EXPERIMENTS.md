# Compact experiment record

These are exploratory measurements, not the final matched container benchmark table. Each promoted option was checked on complete clips. Original failures and raw local traces were retained; failed runs are excluded from performance claims.

| Experiment | Observed result on R9700 | Decision |
| --- | --- | --- |
| Untiled vs tiled VAE, base 832×480×49 | Tiled decode ~10 s versus ~7.5 s before MIOpen qualification; driver high-water memory remained about 30.3 GiB | Untiled default; tiling is not a blanket low-memory claim |
| BF16 vs FP16 VAE | FP16 decode ~7.5 s did not improve BF16 | BF16 |
| MIOpen opt-in | FastWan full clip ~11.2 → ~8.75 s; decode ~7.37 → ~5.59 s | Enable for **both** stock and Paiton; not compiler acceleration |
| CK vs SDPA attention | FastWan denoise ~1.76 versus ~2.07 s | CK for both engines |
| Block `torch.compile` | Stock denoise ~1.73 s, complete ~8.60–8.62 s; works with dynamic-loader graph breaks | Shared strongest stock baseline |
| Denoiser normalization/modulation and gated residual fusion | Small denoiser gain; encoded PSNR 26.07 dB in exploratory fox comparison | Compiled/tested, but not enabled in release default |
| VAE normalization + SiLU fusion | Decode ~5.31–5.33 versus ~5.56–5.57 s at 832×480×49; denoised latents bitwise equal | Promote; complete gain is smaller and encoding-sensitive |
| Four generic BF16 projection tiles | Slower than native rocBLAS/hipBLAS kernels | Reject |
| Channels-last-3D hot VAE convolution including conversion | ~65–76 ms versus native ~17–18 ms | Reject |
| Four implicit-GEMM VAE convolution rewrites | ~25–37 ms versus MIOpen ~18.42 ms | Reject |
| Full resident denoiser graph capture | Actual failure: unpinned CPU-to-GPU copy inside capture | Disable full-pipeline capture; standalone artifact graph replay tests pass |
| Native FP8 denoiser | ~1.75 s denoise, no improvement over compiled BF16; coherent but changed fox pixels | Keep BF16; no extra format downloaded |
| A14B SDNQ storage vs W4A8 computation | Warm full clips ~59–60 s versus ~53.6–54.0 s; W4A8 first run includes substantial compilation | Real low-bit benefit, but slower/heavier host footprint than selected 5B settings |

The initial FastWan raw-loader attempts had key-layout mismatches. A later diagnostic omitted latent unnormalization; its output is invalid and not quality evidence. The corrected implementation performs Comfy's latent-format output transform before decoding. A large VAE profiling trace was processed with a bounded-memory streaming parser after full-trace postprocessing proved too costly for the host.

The initial container UI check exposed absolute host paths inside reused TorchInductor cache entries. The package now separates container and host compiler caches. All three packaged workflow checks were rerun successfully after the fix.

## Numerical checks

The compiled ABI tests exercise BF16/FP16 denoiser arithmetic, temporal modulation, batch one/two, invalid arguments, nondefault streams and graph replay. An initial FP16 residual discrepancy came from half-precision FMA contraction; explicit FP32 opmath restored stock rounding. VAE tests cover zero input and measured large C=256/512/1024 shapes. Four GPU tests pass.

The DMD sampler reference checks preserve upstream clean-prediction arithmetic, re-noising and final CPU generator state on four fixtures. Stock and Paiton use the same sampler, guidance and RNG consumption. Pixel similarity metrics are supporting evidence, not proof of semantic quality parity.
