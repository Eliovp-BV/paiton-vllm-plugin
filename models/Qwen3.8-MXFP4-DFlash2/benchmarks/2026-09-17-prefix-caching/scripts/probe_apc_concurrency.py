#!/usr/bin/env python3
"""Private shared-history concurrency smoke; --check makes no HTTP requests.

Uses the frozen 8K archive, primes it once, submits distinct exact-answer
branches at each client concurrency, then reuses one branch after release.
This is a functional APC/scheduler probe, not the earlier throughput benchmark.
"""
import argparse
import concurrent.futures
import copy
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.error
import urllib.request

import apc_benchmark as bench


def concurrency_values(value):
    try:
        values = [int(part) for part in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('Use unique comma-separated integers between 1 and 8') from error
    if not values or len(set(values)) != len(values) or any(n < 1 or n > 8 for n in values):
        raise argparse.ArgumentTypeError('Use unique comma-separated integers between 1 and 8')
    return values


def base_case(corpus):
    case = copy.deepcopy(corpus['cases'][0])
    if case['payload'].get('temperature') != 0 or case['payload'].get('top_p') != 1:
        raise ValueError('Expected frozen greedy corpus')
    if not case['payload'].get('stream') or not case['payload'].get('stream_options', {}).get('include_usage'):
        raise ValueError('Expected streamed frozen corpus with usage')
    if set(case['expected']) != {'document_tag', 'KEY_1', 'KEY_2', 'KEY_3', 'latest_tail'}:
        raise ValueError('Expected five-key archive retrieval corpus')
    case['name'] = 'prime'
    return case


def branches(corpus, count):
    base = base_case(corpus)
    shared = copy.deepcopy(base['payload']['messages'])
    shared.append(dict(role='assistant', content=json.dumps(base['expected'], ensure_ascii=False)))
    result = []
    for index in range(count):
        tail = f'concurrent-c{count}-branch-{index:02d}-{base["expected"]["document_tag"]}'
        payload = copy.deepcopy(base['payload'])
        payload['messages'] = copy.deepcopy(shared) + [dict(role='user', content=
            bench.FILLER * 8 + '\nTAIL_UPDATE: ' + tail +
            '\nReturn the five-key JSON using the original archive and this newest tail update. No explanation.')]
        result.append(dict(name=f'c{count}-branch-{index:02d}', payload=payload,
                           expected={**base['expected'], 'latest_tail': tail}))
    return result


def streaming_request(url, payload, timeout, sse_path):
    """Independent per-thread parser retaining exact wire SSE and event times."""
    start = time.monotonic()
    response = dict(status=None, started_utc=bench.now(), events=[], content='', reasoning='',
                    finish_reason=None, done_received=False)
    first = last = None
    request = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
    try:
        with sse_path.open('wb') as wire, urllib.request.urlopen(request, timeout=timeout) as stream:
            response['status'] = stream.status
            for raw_line in stream:
                elapsed = time.monotonic() - start
                wire.write(raw_line)
                line = raw_line.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                if line[5:].strip() == '[DONE]':
                    response['done_received'] = True
                    break
                event = json.loads(line[5:].strip())
                response['events'].append(dict(elapsed_seconds=elapsed, data=event))
                if event.get('usage'):
                    response['usage'] = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta') or {}
                    text = delta.get('content') or ''
                    reasoning = delta.get('reasoning') or delta.get('reasoning_content') or ''
                    if text or reasoning or delta.get('tool_calls'):
                        first = elapsed if first is None else first
                        last = elapsed
                    response['content'] += text
                    response['reasoning'] += reasoning
                    if delta.get('tool_calls'):
                        response['unexpected_tool_call'] = True
                    response['finish_reason'] = choice.get('finish_reason') or response['finish_reason']
    except urllib.error.HTTPError as error:
        response.update(status=error.code, error=error.read().decode('utf-8', 'replace'))
    except Exception as error:
        response['error'] = type(error).__name__ + ': ' + str(error)
    response.update(elapsed_seconds=time.monotonic() - start, ttft_seconds=first, last_update_seconds=last,
        update_window_seconds=last-first if first is not None and last is not None else None)
    completion = response.get('usage', {}).get('completion_tokens')
    response['short_output_decode_tps'] = ((completion - 1) / (last - first)
        if completion and completion > 1 and first is not None and last is not None and last > first else None)
    return response


def check_response(case, response):
    parsed, correct = None, False
    try:
        content = response['content'].strip()
        if content.startswith('```json') and content.endswith('```'):
            content = content[7:-3].strip()
        parsed = json.loads(content)
        correct = parsed == case['expected']
    except (ValueError, TypeError, KeyError):
        pass
    usage = response.get('usage') or {}
    result = dict(case=case['name'], expected=case['expected'], actual=parsed,
        status=response.get('status'), finish_reason=response.get('finish_reason'),
        stream_done=response.get('done_received', False), usage=usage,
        ttft_seconds=response.get('ttft_seconds'), elapsed_seconds=response.get('elapsed_seconds'),
        short_output_decode_tps=response.get('short_output_decode_tps'),
        error=response.get('error'), metrics_scope='Whole batch only; no per-request counter attribution')
    result['passed'] = bool(correct and result['status'] == 200 and result['finish_reason'] == 'stop'
        and result['stream_done'] and not result['error'] and not response.get('unexpected_tool_call')
        and isinstance(usage.get('completion_tokens'), int) and usage['completion_tokens'] > 0)
    return result


def distribution(values):
    values = [value for value in values if value is not None]
    return dict(min=min(values), median=statistics.median(values), max=max(values)) if values else None


def run_batch(args, name, cases):
    folder = args.out / name
    folder.mkdir()
    for index, case in enumerate(cases):
        bench.save(folder / f'{index:02d}-request.json', case['payload'])
    before = bench.metrics(args.url, folder / 'metrics-before.txt')
    gate = threading.Event()
    started = [None]

    def execute(index, case):
        gate.wait()
        offset = time.monotonic() - started[0]
        response = streaming_request(args.url, case['payload'], args.timeout, folder / f'{index:02d}-raw.sse')
        response['batch_start_offset_seconds'] = offset
        return response

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
        futures = [pool.submit(execute, index, case) for index, case in enumerate(cases)]
        started[0] = time.monotonic()
        gate.set()
        responses = [future.result() for future in futures]
        wall_seconds = time.monotonic() - started[0]
    # Serialize responses before requesting counters so a metrics failure cannot
    # discard the observed completions or their exact SSE.
    results = []
    for index, (case, response) in enumerate(zip(cases, responses)):
        bench.save(folder / f'{index:02d}-response.json', response)
        results.append(check_response(case, response))
    bench.save(folder / 'requests-summary.json', results)
    after = bench.metrics(args.url, folder / 'metrics-after.txt')
    completion_tokens = sum(result['usage'].get('completion_tokens', 0) for result in results)
    batch = dict(name=name, client_concurrency=len(cases), requests=len(cases),
        passed=all(result['passed'] for result in results), wall_seconds=wall_seconds,
        completion_tokens=completion_tokens,
        aggregate_output_tokens_per_second=completion_tokens / wall_seconds,
        aggregate_definition='All returned completion tokens divided by batch wall time, including queueing/prefill',
        queue_inclusive_ttft_seconds=distribution([result['ttft_seconds'] for result in results]),
        request_elapsed_seconds=distribution([result['elapsed_seconds'] for result in results]),
        short_output_decode_tps=distribution([result['short_output_decode_tps'] for result in results]),
        metrics=bench.metric_delta(before, after), request_results=results)
    bench.save(folder / 'summary.json', batch)
    print(json.dumps({key: batch[key] for key in ('name', 'client_concurrency', 'passed', 'wall_seconds',
        'completion_tokens', 'aggregate_output_tokens_per_second', 'queue_inclusive_ttft_seconds')}), flush=True)
    return batch


def run(args):
    corpus_bytes = args.corpus.read_bytes()
    corpus = json.loads(corpus_bytes)
    initial = base_case(corpus)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(started_utc=bench.now(), arm=args.arm, endpoint=args.url,
        corpus=str(args.corpus.resolve()), corpus_sha256=bench.digest(corpus_bytes),
        harness_sha256=bench.digest(Path(__file__).read_bytes()), client_concurrencies=args.concurrency,
        settings=initial['payload'], scope='Functional/shared-history concurrency smoke with short exact JSON answers; '
            'not a matched comparison with the earlier throughput benchmark',
        history_policy='Frozen correct assistant JSON shared by all branch requests; actual answers independently checked',
        cache_policy='Prime once, then distinct branches; reuse the first branch of the last batch after all requests finish. '
                     'A priming request is not necessarily cold: use measured cache counters.',
        metrics_policy='Prometheus snapshots bracket each complete batch; counters are never attributed to individuals.',
        server_policy='Caller controls max_num_seqs, cache, backend and context. Client concurrency alone does not prove '
                      'simultaneous scheduler execution; retain actual server configuration and scheduler logs.',
        limitations=['Synthetic marker retrieval, not broad quality or token-level numerical validation',
                     'Short-output decode rates are not sustained generation throughput',
                     'Global counters assume no unrelated clients during the run'], status='RUNNING')
    # Requests themselves are saved separately; avoid duplicating the archive.
    manifest['settings'] = {key: value for key, value in initial['payload'].items() if key != 'messages'}
    bench.save(args.out / 'manifest.json', manifest)
    summary = []
    try:
        bench.save(args.out / 'models.json', bench.http_json(args.url, '/v1/models'))
        if args.server_metadata:
            bench.save(args.out / 'server-metadata.json', json.loads(args.server_metadata.read_text()))
        summary.append(run_batch(args, '00-prime', [initial]))
        bench.save(args.out / 'summary.json', summary)
        if not summary[-1]['passed']:
            raise ValueError('Priming answer failed')
        last = None
        for index, count in enumerate(args.concurrency, start=1):
            cases = branches(corpus, count)
            summary.append(run_batch(args, f'{index:02d}-c{count}', cases))
            bench.save(args.out / 'summary.json', summary)
            if not summary[-1]['passed']:
                raise ValueError(f'Concurrent C{count} branch answer failed')
            last = copy.deepcopy(cases[0])
        last['name'] = 'reuse-after-release-' + last['name']
        summary.append(run_batch(args, f'{len(args.concurrency)+1:02d}-reuse-after-release', [last]))
        bench.save(args.out / 'summary.json', summary)
        if not summary[-1]['passed']:
            raise ValueError('Reuse-after-release answer failed')
        manifest['status'] = 'PASS'
    except BaseException as error:
        manifest.update(status='FAIL', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        manifest['finished_utc'] = bench.now()
        bench.save(args.out / 'manifest.json', manifest)


def check():
    corpus = dict(cases=bench.build_cases(16, 20260917, max_tokens=192))
    initial = base_case(corpus)
    for count in (1, 2, 4, 8):
        cases = branches(corpus, count)
        assert len(cases) == count
        assert len({case['expected']['latest_tail'] for case in cases}) == count
        for case in cases:
            assert case['payload']['messages'][:2] == initial['payload']['messages']
            assert case['payload']['messages'][:3] == cases[0]['payload']['messages'][:3]
            assert case['expected']['latest_tail'] in case['payload']['messages'][-1]['content']
            assert {key: value for key, value in case['payload'].items() if key != 'messages'} == {
                key: value for key, value in initial['payload'].items() if key != 'messages'}
            response = dict(status=200, content=json.dumps(case['expected']), finish_reason='stop',
                            done_received=True, usage=dict(completion_tokens=90))
            assert check_response(case, response)['passed']
            response['content'] = json.dumps(initial['expected'])
            assert not check_response(case, response)['passed']
        replay = copy.deepcopy(cases[0])
        assert replay['payload'] == cases[0]['payload']
    print(json.dumps(dict(status='PASS', network_calls=0, concurrency=[1, 2, 4, 8],
                         correct_and_stale_answer_checks=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', type=bench.local_url, default='http://127.0.0.1:18981')
    parser.add_argument('--corpus', type=Path, default=Path(__file__).with_name('corpus-8k.json'))
    parser.add_argument('--arm', choices=bench.ARMS, default='D')
    parser.add_argument('--concurrency', type=concurrency_values, default=[1, 2, 4, 8])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--server-metadata', type=Path)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check()
    elif args.out is None:
        parser.error('--out is required unless --check is used')
    else:
        run(args)


if __name__ == '__main__':
    main()
