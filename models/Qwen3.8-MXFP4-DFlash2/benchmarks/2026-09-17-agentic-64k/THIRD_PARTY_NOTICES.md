# Benchmark attribution

The benchmark, prompt corpus, and standalone report renderer are from
[GGZ14/BetterBench](https://github.com/GGZ14/BetterBench), version 0.6.0,
commit `d00ad5ec8098c06584a88ec3468bacd37d5ed098`.

Copyright 2026 BetterBench contributors. Its license is reproduced unchanged
in [BETTERBENCH-LICENSE](BETTERBENCH-LICENSE). Request payloads derive from the
same original corpus used in the [previous comparison](../2026-09-16-betterbench/README.md),
with the recorded 128-token cap and unchanged messages/IDs. HTML and Markdown
reports were regenerated using the unchanged pinned renderer from measurements
whose environment metadata was sanitized.

The static chart was generated with Matplotlib. The separately tested client
was official OpenCode `opencode-ai@1.18.31`; its implementation is not redistributed
here. Model/runtime attribution remains in the model's
[third-party notices](../../THIRD_PARTY_NOTICES.md). Naming a project does not imply
endorsement by its authors.
