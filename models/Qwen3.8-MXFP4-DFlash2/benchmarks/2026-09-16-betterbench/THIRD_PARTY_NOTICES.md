# Benchmark attribution

The benchmark, original prompt corpus and standalone report template are from
[GGZ14/BetterBench](https://github.com/GGZ14/BetterBench) version 0.6.0, commit
`d00ad5ec8098c06584a88ec3468bacd37d5ed098`.

Copyright 2026 BetterBench contributors. The upstream license file is reproduced
unchanged as [BETTERBENCH-LICENSE](BETTERBENCH-LICENSE).

The files in `corpus/` derive from BetterBench's `corpus/v1/`. The only change
is capping `max_tokens` at 128; messages, ordering and IDs are unchanged. The
original and modified file hashes are listed in
[bounded-corpus-manifest.json](bounded-corpus-manifest.json).

The HTML and Markdown reports were generated with the unchanged upstream
renderer from metadata-sanitized measurements; generated trailing whitespace is normalized. The comparison charts and their
rendering script were prepared for this repository using Matplotlib.

[GGZ14/vllm-mxfp4](https://github.com/GGZ14/vllm-mxfp4) and
[StillDeadcode/libr4d](https://codeberg.org/StillDeadcode/libr4d) identify the
external serving implementation and library evaluated here. Their implementation
source and runtime binaries are not redistributed in this benchmark directory.
The reference loader helper changes checkpoint I/O only.

Model/runtime attribution and licenses remain documented in the model directory's
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md). Naming a project here
does not imply endorsement by its authors.
