# 10 October 2026: exact-arithmetic prefill (the `-nopf` modes), prefill part only, development stack

Two R9700, the development stack of the same plugin commit and bundles (not re-run on the release image), `--mode prefill-long`
equivalent with the two native-GDN-prefill flags off; BetterBench 0.6.0 standard profile, prefill depths 2K-128K x 8 plus the exact
64,000-token probe; exact bf16 wire, cold prefix cache. Two arms: `w8a8/` = the shipped W8A8 prompt rows on the exact path (what
`run-flashnext-exact.sh` / `--mode prefill-long-nopf` run), `w8a16/` = W8A16 prompt rows on the exact path for reference.

| prefill tok/s @ 2K / 8K / 16K / 32K / 64K / 128K | exactly 64,000 tokens (full / steady) |
|---|---|
| W8A8 prompt rows (shipped `-nopf` modes): 7,575 / 7,785 / 7,762 / 7,711 / 7,607 / 7,422 | 7,673 / 7,666 |
| W8A16 prompt rows: 7,189 / 7,524 / 7,513 / 7,458 / 7,368 / 7,170 | 7,394 / 7,384 |

About 7 % below the default (native) prefill path; decode rows are identical between the paths. [numbers.json](numbers.json) holds the W8A8 arm.
