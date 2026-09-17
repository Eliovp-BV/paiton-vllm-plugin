"""Short, capped generation probe; does not score completed coding tasks."""
import argparse
import json
from pathlib import Path
from apc_benchmark import MODEL, request, metrics, metric_delta, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18981')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    payload = dict(model=MODEL, messages=[dict(role='user', content=(
        'Write a Python module implementing a bounded LRU cache with get, put, '
        'delete, resizing, clear, hit/miss statistics, and iteration from oldest '
        'to newest. Include type hints and unit tests for eviction, replacement, '
        'a zero capacity cache, resizing and missing keys. Explain the runtime '
        'complexities after the implementation.'))], temperature=0, top_p=1,
        seed=42, max_tokens=512, stream=True, stream_options=dict(include_usage=True),
        chat_template_kwargs=dict(enable_thinking=False))
    save(args.out / 'request.json', payload)
    results = []
    for i in range(3):
        before = metrics(args.url, args.out / f'{i}-before.txt')
        response = request(args.url, payload, 180)
        save(args.out / f'{i}-response.json', response)
        after = metrics(args.url, args.out / f'{i}-after.txt')
        record = dict(arm=args.arm, index=i, warmup=i == 0,
            status=response['status'], ttft_seconds=response['ttft_seconds'],
            elapsed_seconds=response['elapsed_seconds'],
            decode_tps=response['short_output_decode_tps'], usage=response.get('usage'),
            finish_reason=response.get('finish_reason'), metrics=metric_delta(before, after),
            limitation='512-token capped code generation; no completed-task quality claim')
        results.append(record)
        save(args.out / 'summary.json', results)
        print(json.dumps({k: record[k] for k in ('arm','index','warmup','status','ttft_seconds','decode_tps','usage','finish_reason')}), flush=True)
        if response['status'] != 200 or response.get('error'):
            raise RuntimeError(response.get('error', 'HTTP generation failed'))


if __name__ == '__main__':
    main()
