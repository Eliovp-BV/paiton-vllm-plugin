# Ornith 1.5 RDNA4 benchmark

## Result

Paiton produced an average of **44.628 output tokens per second** across two
qualified runs on one Radeon AI PRO R9700. The fastest obtained stock vLLM
result was **35.132 output tokens per second**.

| Runtime | Output tokens/s | Median TPOT | Relative output |
| --- | ---: | ---: | ---: |
| Fastest obtained stock vLLM | 35.132 | 27.457 ms | 1.000x |
| Paiton with DFlash 16, run 1 | 44.538 | 18.857 ms | 1.268x |
| Paiton with DFlash 16, run 2 | 44.717 | 18.871 ms | 1.273x |
| Paiton with DFlash 16, mean | **44.628** | **18.864 ms** | **1.270x** |

Paiton delivered **27.03% more output tokens per second**. With equal GPU
purchase price and operating time, the same output volume therefore costs
about **21.28% less in GPU time**.

The stock row is the fastest obtained stock result. Significant time was
spent testing the available stock execution, kernel, mixture-of-experts, and
graph settings before selecting it. This gives stock the benefit of its best
observed performance band rather than averaging in a slower stock run.

## Hardware and software

- AMD Radeon AI PRO R9700 with 32 GB VRAM
- ROCm 7.14
- GPU performance level `AUTO`
- GPU power profile `COMPUTE`
- vLLM revision `39bd959b582c85e78e7e0326d49042ce7c3c07ed`
- model revision `9e488f46c0f7969f84c9923ee0256311cd50316e`
- tensor parallel size 1
- maximum sequence count 1
- maximum model length 8,192 tokens
- 3 GiB KV cache reservation
- prefix caching disabled

The checkpoint is the public 22.9 GB MXFP4 model. No BF16 target checkpoint
was used. Paiton also loaded the public 772 MB DFlash draft.

## Workload

Each measured run used:

- 3 warmup requests
- 12 measured requests
- random dataset with seed 42
- requested input length 256 tokens
- output length exactly 256 tokens
- 3,072 measured output tokens in total
- request rate unlimited with concurrency 1
- temperature 0
- EOS ignored
- thinking disabled

All 24 Paiton requests completed successfully. Every request returned exactly
256 output tokens. A separate deterministic natural-language request ran
before and after the benchmark. Both responses were readable, stopped
naturally, contained finite log probabilities, and were byte-identical.

## Fastest stock settings

The selected stock configuration used the pinned vanilla ROCm vLLM runtime
with:

- optimization level O2
- graph capture enabled for batch size 1
- Triton attention
- Triton GDN decode
- unfused Triton mixture-of-experts backend
- DFlash disabled

The fastest stock cell completed 12 of 12 requests and generated all 3,072
requested output tokens. Its output throughput was 35.132 tokens per second
and its median TPOT was 27.457 ms.

## Paiton settings

The public package used:

- the compiled Paiton Ornith text path
- eager target execution
- DFlash with at most 16 speculative tokens
- ROCm attention for the DFlash draft
- the same target checkpoint, tokenizer, workload, GPU profile, and hardware

The two Paiton cells completed in 68.974 and 68.698 seconds. Their output
throughputs were 44.538 and 44.717 tokens per second.

## Reproduce the Paiton benchmark

Start the server first:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git && cd paiton-vllm-plugin && ./models/Ornith-1.5/serve-docker.sh
```

After the server is ready, run the same client workload:

```bash
docker run --rm --network host \
  --mount type=volume,src=paiton-ornith-cache,dst=/models/cache,readonly \
  --entrypoint /opt/venv/bin/vllm \
  ghcr.io/eliovp/paiton-vllm-plugin:ornith15-mxfp4-rdna4-v1.0.0 \
  bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model ornith \
  --tokenizer /models/cache/ornith/9e488f46c0f7969f84c9923ee0256311cd50316e \
  --dataset-name random \
  --seed 42 \
  --num-warmups 3 \
  --num-prompts 12 \
  --random-input-len 256 \
  --random-output-len 256 \
  --random-range-ratio 0 \
  --random-prefix-len 0 \
  --request-rate inf \
  --max-concurrency 1 \
  --temperature 0 \
  --ignore-eos \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99
```

Performance varies with thermals, clocks, host activity, and driver state.
Use the same GPU profile for comparisons and report complete runs rather than
isolated token bursts.
