# Paiton versus GGZ14: matched quick comparison

Benchmark revision, source hash, profile, corpus bytes/IDs, model ID and all sampling/config fields match.

Positive change means a larger Paiton value. Throughput favors larger values; latency favors smaller values.

| Workload | Metric | Paiton | GGZ14 | Paiton change | Unit |
|---|---|---:|---:|---:|---|
| summary: weighted serial | decode | 98.57 | 83.93 | +17.45% | tok/s |
| category: chat | decode | 76.44 | 56.52 | +35.24% | tok/s |
| category: chat | TTFT | 102.19 | 113.82 | -10.22% | ms |
| category: code | decode | 90.28 | 75.11 | +20.20% | tok/s |
| category: code | TTFT | 101.76 | 102.61 | -0.83% | ms |
| category: file_edit | decode | 119.91 | 105.97 | +13.15% | tok/s |
| category: file_edit | TTFT | 102.53 | 98.98 | +3.58% | ms |
| category: json | decode | 127.99 | 106.77 | +19.88% | tok/s |
| category: json | TTFT | 102.68 | 98.41 | +4.35% | ms |
| category: math | decode | 113.34 | 105.61 | +7.32% | tok/s |
| category: math | TTFT | 107.04 | 95.62 | +11.95% | ms |
| category: prose | decode | 91.23 | 85.77 | +6.37% | tok/s |
| category: prose | TTFT | 108.68 | 98.02 | +10.88% | ms |
| category: reasoning | decode | 85.23 | 66.81 | +27.58% | tok/s |
| category: reasoning | TTFT | 102.51 | 98.31 | +4.27% | ms |
| category: summarization | decode | 95.63 | 85.55 | +11.78% | tok/s |
| category: summarization | TTFT | 105.22 | 106.41 | -1.12% | ms |
| concurrency: C1 | aggregate | 73.98 | 67.31 | +9.92% | tok/s |
| concurrency: C1 | per-request decode | 78.60 | 71.37 | +10.13% | tok/s |
| concurrency: C1 | queue-inclusive TTFT | 103.54 | 96.54 | +7.25% | ms |
| concurrency: C2 | aggregate | 129.98 | 111.19 | +16.90% | tok/s |
| concurrency: C2 | per-request decode | 80.09 | 66.26 | +20.87% | tok/s |
| concurrency: C2 | queue-inclusive TTFT | 164.33 | 148.02 | +11.02% | ms |
| concurrency: C4 | aggregate | 201.32 | 156.94 | +28.28% | tok/s |
| concurrency: C4 | per-request decode | 62.99 | 65.03 | -3.14% | tok/s |
| concurrency: C4 | queue-inclusive TTFT | 221.77 | 267.89 | -17.22% | ms |
| concurrency: C8 | aggregate | 281.90 | 154.78 | +82.13% | tok/s |
| concurrency: C8 | per-request decode | 47.62 | 56.35 | -15.50% | tok/s |
| concurrency: C8 | queue-inclusive TTFT | 416.21 | 3030.91 | -86.27% | ms |
| prefill: 2000 | prefill | 3375.80 | 2987.85 | +12.98% | tok/s |
| prefill: 2000 | TTFT | 458.62 | 518.57 | -11.56% | ms |
| prefill: 7000 | prefill | 3395.58 | 2978.68 | +14.00% | tok/s |
| prefill: 7000 | TTFT | 1541.41 | 1757.15 | -12.28% | ms |

Actual prefill input tokens:

- Target 2000: Paiton [1560, 1530]; GGZ14 [1560, 1530].
- Target 7000: Paiton [5211, 5257]; GGZ14 [5211, 5257].

Speculative counters cover the entire client interval, including discarded warmups:

- paiton: 1921 drafts, 5350/13447 draft tokens accepted (39.79%); 2.785 accepted tokens/draft; 3.785 accepted-plus-one/draft (counter-derived proxy).
- ggz: 1919 drafts, 5340/12534 draft tokens accepted (42.60%); 2.783 accepted tokens/draft; 3.783 accepted-plus-one/draft (counter-derived proxy).

Observed Paiton regressions (including small changes that may be noise):

- category file_edit, TTFT: +3.58%.
- category json, TTFT: +4.35%.
- category math, TTFT: +11.95%.
- category prose, TTFT: +10.88%.
- category reasoning, TTFT: +4.27%.
- concurrency C1, queue-inclusive TTFT: +7.25%.
- concurrency C2, queue-inclusive TTFT: +11.02%.
- concurrency C4, per-request decode: -3.14%.
- concurrency C8, per-request decode: -15.50%.

Serial completions at token limit: Paiton 15/16; GGZ14 14/16.

- Separate sequential runs; these are descriptive comparisons, not interleaved statistical pairs.
- No significance or percentile claims from this short screen; small changes may be noise.
- TTFT includes server queueing and HTTP cost, excluding client semaphore wait.
- Sampled decoding can produce different output paths despite identical seed.
- The weighted score omits chat and math; all eight categories are reported individually.
- Served model ID matches; checkpoint hashes and shared GPU/cache settings are recorded in ../../checkpoint.lock.json and provenance.json.
