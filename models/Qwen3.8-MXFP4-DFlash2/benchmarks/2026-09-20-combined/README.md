# Qwen3.8 full benchmark — 2026-09-20

R9700, 300 W; vLLM 0.29 / ROCm 10; 65,536 context; maximum eight sequences; APC off; thinking off. Temperature 0.7, top-p 0.95, top-k 20, seed 42. Each arm uses a fresh process and the same fixed warmup.

| Category | GGZ published update p50 ms | New Paiton update p50 ms | GGZ published tok/s | Previous post Paiton tok/s | Fresh release control tok/s | New package tok/s |
|---|---:|---:|---:|---:|---:|---:|
| chat | 42.3 | 33.2 | 67.1 | 123.6 | 123.2 | 121.3 |
| code | 42.5 | 33.4 | 120.5 | 169.1 | 168.7 | 180.6 |
| file edit | 42.5 | 33.3 | 138.0 | 185.7 | 185.5 | 181.2 |
| json | 42.4 | 33.4 | 153.1 | 216.0 | 215.8 | 218.1 |
| math | 42.5 | 33.5 | 140.0 | 182.5 | 182.7 | 184.3 |
| prose | 42.3 | 33.4 | 69.2 | 75.4 | 75.4 | 79.8 |
| reasoning | 34.3 | 33.4 | 123.9 | 112.4 | 112.5 | 117.8 |
| summarization | 34.2 | 33.3 | 141.7 | 115.0 | 115.1 | 138.8 |

Weighted decode: **146.85 → 154.42 tok/s (+5.16%)**.

| Concurrent requests | GGZ published | Historical GGZ local | Previous post Paiton | Fresh release control | New package |
|---:|---:|---:|---:|---:|---:|
| 1 | 120 | 110.20 | 115.96 | 115.43 | 122.62 |
| 2 | 215 | 193.01 | 202.85 | 200.55 | 207.18 |
| 4 | 322 | 289.02 | 301.75 | 300.15 | 308.17 |
| 8 | 471 | 396.45 | 406.49 | 410.71 | 421.20 |

Concurrency values are aggregate generated tokens per second over each complete 48-request workload.

| Nominal prefill depth | GGZ published input tok/s | Previous post Paiton | Fresh release control | New package |
|---:|---:|---:|---:|---:|
| 2,000 | 3,552 | 3,510 | 3501 | 3501 |
| 8,000 | 3,536 | 3,532 | 3535 | 3692 |
| 16,000 | 3,619 | 3,535 | 3540 | 3694 |
| 32,000 | 3,437 | 3,389 | 3394 | 3584 |
| 64,000 | 3,192 | 3,116 | 3118 | 3293 |

Update p50 is the median streamed-update gap, not per-token latency or TTFT.
Sampled output content and accepted-token work can differ; these are serving-throughput measurements, not identical-output timing.
Nominal prefill depths correspond to median actual prompt lengths 1516.5, 5894.5, 11802, 23549.5 and 47016.5.
GGZ comparison values are historical references from the previous post; GGZ was not rerun.

Both arms: 290/290 successful benchmark requests. Candidate matches all 12/12 greedy controls before timing. Each arm repeats 12/12 greedy outputs after the benchmark. No unhandled serving errors.

Decode has five scored requests per category after one warmup. Prefill has eight scored requests per depth after two warmups. These are individual complete runs; category values are not selected across repeats.
