"""Compare native checkpoint loading; fixed-token probes are not complete summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--transcript', type=Path, required=True)
    parser.add_argument('--strategy', choices=['lazy', 'prefetch'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new file.')
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from paiton_meeting.summary import SYSTEM, SUMMARY_SCHEMA, transcript_batches
    model_path = args.model.resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    transcript = json.loads(args.transcript.read_text())
    batch = next(transcript_batches(transcript['segments'], lambda s: len(tokenizer.encode(s))))
    messages = [dict(role='system', content=SYSTEM + '\nReturn JSON matching this schema: ' + json.dumps(SUMMARY_SCHEMA)),
                dict(role='user', content=json.dumps({'transcript': batch}))]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                           enable_thinking=True, low_effort=True, tokenize=False)
    started = time.perf_counter()
    model = LLM(model=str(model_path), tokenizer=str(model_path), dtype='bfloat16',
                max_model_len=16384, max_num_seqs=1, max_num_batched_tokens=4096,
                gpu_memory_utilization=.45, enforce_eager=False, enable_prefix_caching=False,
                trust_remote_code=False, load_format='safetensors',
                safetensors_load_strategy=args.strategy, disable_log_stats=True,
                seed=1201, attention_backend='TRITON_ATTN')
    loading = time.perf_counter() - started
    runs = []
    for repeat in range(3):
        started = time.perf_counter()
        response = model.generate([prompt], SamplingParams(max_tokens=128, min_tokens=128,
                                  temperature=1.0, top_p=.95, seed=1201), use_tqdm=False)[0]
        output = response.outputs[0]
        runs.append(dict(repeat=repeat, seconds=time.perf_counter()-started,
                         prompt_tokens=len(response.prompt_token_ids), output_tokens=len(output.token_ids),
                         token_sha256=hashlib.sha256(json.dumps(output.token_ids).encode()).hexdigest()))
    result = dict(strategy=args.strategy, loading_seconds=loading, runs=runs,
                  limitation='Fresh engine, persistent caches; no shared caches cleared. Fixed 128-token probe, not summary quality or full-pipeline speed.')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
