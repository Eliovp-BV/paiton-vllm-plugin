# Reproduce the bounded 64K check

Launch the published 64K image using the model's [v1.1.0 launch instructions](../../LAUNCH-agentic-v1.1.0.md), with its defaults, an otherwise idle GPU, and the pinned model cache. Do not run the 64K and 200K images on the GPU together. The endpoint below is an example; substitute the port of the server you own.

Use [BetterBench 0.6.0 at `d00ad5ec8098c06584a88ec3468bacd37d5ed098`](https://github.com/GGZ14/BetterBench/tree/d00ad5ec8098c06584a88ec3468bacd37d5ed098) with its documented Python dependencies. From this benchmark directory, with `BETTERBENCH_SOURCE` pointing to that checkout:

```bash
PYTHONPATH="$BETTERBENCH_SOURCE" BETTERBENCH_NO_UPDATE_CHECK=1 \
python3 -m betterbench.cli run \
  --endpoint http://localhost:8000/v1 \
  --model Qwen3.8-27B-Quark-AWQ-MXFP4 \
  --config profile-quick.json \
  --corpus ../2026-09-16-betterbench/corpus \
  --max-model-len 65536 \
  --out results-new.json \
  --no-update-check \
  --note 'thinking=server default false; request has no chat_template_kwargs'
```

Use a fresh output filename and preserve every run. This is 52 measured requests and 10 warmups, with the original phase order and nonce RNGs. It is the configured historical quick workload; no additional CLI `--quick` flag is necessary. The inherited corpus retains original messages/IDs and caps generation at 128 tokens. Some original entries have a smaller budget. Prefill depths are approximate; report actual returned token counts.

The archived measurement additionally wrapped the unchanged upstream functions to retain every exact request payload and `RunResult`, which the normal concurrency summary does not keep. It saved `/metrics` before/after each serial category, prefill depth and concurrency level, outside concurrent timed sections. Small post-request Python bookkeeping remains inside aggregate wall time. Counter deltas include the labelled warmups. Capture generic scheduler logs for the same time interval if repeating the analysis.

Model pins are in the shared [checkpoint lock](../../checkpoint.lock.json): target revision `5233554c5fa56afda40150556b95573c2d7d29c0`, drafter revision `ee0cb26a8279b7910cc28d82a8a3e15e4728d56f`. The final image was tested without a code bind mount. Check [image receipts](evidence/image-64k.json) and [profile](profile-quick.json) before comparing results.

The [README](README.md) defines per-request generation, aggregate throughput, TTFT and prefill separately. A historical thinking-enabled run is not directly comparable to this no-thinking default run. The long-context retrieval and OpenCode checks are separate functional evidence, not part of this BetterBench command or its timing score.
