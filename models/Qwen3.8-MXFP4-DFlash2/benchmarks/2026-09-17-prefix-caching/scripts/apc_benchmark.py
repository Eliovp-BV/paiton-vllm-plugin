#!/usr/bin/env python3
"""Local HTTP APC experiment. Preparation tokenizes only; run submits inference.

The frozen corpus is reused byte-for-byte for A/B/C. No server control, GPU
libraries, credentials, package installation, or publication is performed.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

MODEL = 'Qwen3.8-27B-Quark-AWQ-MXFP4'
FILLER = 'Archive note: ordinary background records discuss gardens, books, weather, travel, kitchen utensils, and local timetables.\n'
CASE_ORDER = ['cold', 'repeat', 'grow1', 'grow2', 'changed_prefix', 'branch_b', 'return_a']
ARMS = {'A': 'native compact GDN, APC off, mamba none',
        'B': 'stock GDN fallback, APC off, mamba none',
        'C': 'stock GDN fallback, APC on, mamba align',
        'D': 'stock GDN cache/decode, native recurrent prefill, APC align'}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def local_url(value):
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in ('http', 'https') or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise argparse.ArgumentTypeError('This harness permits a local server only')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError('Credentials/query/fragment are not supported')
    if parsed.path.rstrip('/') not in ('', '/v1'):
        raise argparse.ArgumentTypeError('Use the server root or /v1 URL')
    return urllib.parse.urlunparse(parsed._replace(path='', params='')).rstrip('/')


def http_json(url, route, payload=None, timeout=120):
    request = urllib.request.Request(url + route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def build_cases(repetitions, seed, max_tokens=128):
    tag = digest(str(seed).encode())[:10]
    base_expected = {'document_tag': 'archive-' + tag, 'KEY_1': 'cobalt-731-lantern',
        'KEY_2': 'maple-482-circuit', 'KEY_3': 'violet-965-harbor', 'latest_tail': 'none'}
    system = ('Read the archive exactly. Return only one JSON object with the five keys '
        'document_tag, KEY_1, KEY_2, KEY_3, latest_tail. Preserve their exact strings. '
        'document_tag and KEY_1/KEY_2/KEY_3 come from the archive. latest_tail is none '
        'until a later user message gives TAIL_UPDATE; thereafter use the most recent '
        'TAIL_UPDATE in this conversation. Background notes do not change any record.')
    parts = [f'DOCUMENT_TAG: {base_expected["document_tag"]}\nRead this archive and retain its exact records.\n']
    counts = [repetitions // 4] * 3 + [repetitions - 3 * (repetitions // 4)]
    for index, count in enumerate(counts):
        parts.append(FILLER * count)
        if index < 3:
            parts.append(f'UNIQUE RECORD KEY_{index+1}: {base_expected[f"KEY_{index+1}"]}\n')
    parts.append('\nReturn the five-key JSON object now. No explanation.')
    base = [{'role': 'system', 'content': system}, {'role': 'user', 'content': ''.join(parts)}]

    def answer(expected):
        return {'role': 'assistant', 'content': json.dumps(expected, ensure_ascii=False)}

    def turn(tail):
        return {'role': 'user', 'content': FILLER * 8 + '\nTAIL_UPDATE: ' + tail +
            '\nReturn the five-key JSON using the original archive and the newest tail update. No explanation.'}

    expected_a1 = {**base_expected, 'latest_tail': 'branch-a-first-' + tag}
    expected_a2 = {**base_expected, 'latest_tail': 'branch-a-second-' + tag}
    expected_b = {**base_expected, 'latest_tail': 'branch-b-only-' + tag}
    changed_expected = {**base_expected, 'document_tag': 'edited-' + tag}
    grow1 = base + [answer(base_expected), turn(expected_a1['latest_tail'])]
    grow2 = grow1 + [answer(expected_a1), turn(expected_a2['latest_tail'])]
    branch_b = base + [answer(base_expected), turn(expected_b['latest_tail'])]
    changed = copy.deepcopy(base)
    changed[1]['content'] = changed[1]['content'].replace(base_expected['document_tag'], changed_expected['document_tag'], 1)
    definitions = [('cold', base, base_expected), ('repeat', base, base_expected),
        ('grow1', grow1, expected_a1), ('grow2', grow2, expected_a2),
        ('changed_prefix', changed, changed_expected), ('branch_b', branch_b, expected_b),
        ('return_a', grow2, expected_a2)]
    return [dict(name=name, expected=expected, payload=dict(model=MODEL, messages=messages,
        temperature=0, top_p=1, seed=42, max_tokens=max_tokens,
        stream=True, stream_options=dict(include_usage=True),
        chat_template_kwargs=dict(enable_thinking=False))) for name, messages, expected in definitions]


def tokenize(url, payload):
    return http_json(url, '/tokenize', dict(model=payload['model'], messages=payload['messages'],
        add_generation_prompt=True, chat_template_kwargs=payload['chat_template_kwargs']))


def prefix_length(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def prepare(args):
    if args.out.exists():
        raise FileExistsError('Refusing to overwrite frozen corpus: ' + str(args.out))
    models = http_json(args.url, '/v1/models')
    selected = [m for m in models['data'] if m['id'] == MODEL]
    if not selected:
        raise ValueError('Expected model is not served')
    per_line = http_json(args.url, '/tokenize', dict(model=MODEL, prompt=FILLER))['count']
    repetitions = max(1, (args.tokens - 250) // per_line)
    calibration = []
    for _ in range(6):
        cases = build_cases(repetitions, args.seed, args.max_tokens)
        actual = tokenize(args.url, cases[0]['payload'])['count']
        calibration.append(dict(repetitions=repetitions, actual_prompt_tokens=actual))
        if abs(args.tokens - actual) <= per_line:
            break
        repetitions = max(1, repetitions + (args.tokens - actual) // per_line)
    cases = build_cases(repetitions, args.seed, args.max_tokens)
    token_ids = {}
    for case in cases:
        result = tokenize(args.url, case['payload'])
        case['prompt_tokens'] = result['count']
        ids = result.get('tokens')
        if isinstance(ids, list) and all(isinstance(t, int) for t in ids):
            if len(ids) != result['count']:
                raise ValueError('Tokenization count/list disagree')
            case['token_ids_sha256'] = digest(canonical(ids))
            case['common_prefix_tokens_with_previous_cases'] = {name: prefix_length(old, ids) for name, old in token_ids.items()}
            token_ids[case['name']] = ids
        case['payload_sha256'] = digest(canonical(case['payload']))
        limit = selected[0].get('max_model_len')
        if limit and case['prompt_tokens'] + args.max_tokens > limit:
            raise ValueError(f'{case["name"]} would exceed served context {limit}')
    assert cases[0]['payload'] == cases[1]['payload']
    assert cases[3]['payload'] == cases[6]['payload']
    corpus = dict(schema=1, prepared_utc=now(), model=MODEL, target_prompt_tokens=args.tokens,
        actual_base_prompt_tokens=cases[0]['prompt_tokens'], repetitions=repetitions, seed=args.seed,
        max_tokens=args.max_tokens, calibration=calibration, server_models=models,
        scope='Synthetic marker retrieval with frozen correct assistant history; not broad quality or sustained decode benchmark',
        history_policy='Fixed correct assistant JSON; previous generated answers are checked but never substituted into later payloads',
        cases=cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(args.out, corpus)
    print(json.dumps(dict(corpus=str(args.out), sha256=digest(args.out.read_bytes()),
        prompt_tokens={c['name']: c['prompt_tokens'] for c in cases}), indent=2))


METRIC = re.compile(r'^(vllm:[\w:]+)(\{[^}]*\})?\s+([^\s]+)$')


def metrics(url, path):
    with urllib.request.urlopen(url + '/metrics', timeout=20) as response:
        raw = response.read()
    path.write_bytes(raw)
    result = {}
    for line in raw.decode().splitlines():
        match = METRIC.match(line)
        if match:
            name, labels, value = match.groups()
            result[(name, labels or '')] = float(value)
    return result


def metric_delta(before, after):
    counters, positions, sources = {}, {}, {}
    selected = ('spec_decode_num_', 'prefix_cache_', 'prompt_tokens', 'generation_tokens',
        'num_preemptions', 'request_success', 'request_queue_time_seconds',
        'request_prefill_time_seconds', 'request_decode_time_seconds',
        'request_inference_time_seconds', 'time_to_first_token_seconds',
        'request_prefill_kv_computed_tokens', 'e2e_request_latency_seconds')
    for key in before.keys() | after.keys():
        name, labels = key
        short = name.removeprefix('vllm:')
        if not short.startswith(selected) or not short.endswith(('_total', '_sum', '_count')):
            continue
        value = after.get(key, 0) - before.get(key, 0)
        counters[short] = counters.get(short, 0) + value
        if short == 'spec_decode_num_accepted_tokens_per_pos_total':
            positions[re.search(r'position="(\d+)"', labels).group(1)] = value
        if short == 'prompt_tokens_by_source_total':
            sources[re.search(r'source="([^"]+)"', labels).group(1)] = value
    drafted = counters.get('spec_decode_num_draft_tokens_total', 0)
    accepted = counters.get('spec_decode_num_accepted_tokens_total', 0)
    rounds = counters.get('spec_decode_num_drafts_total', 0)
    return dict(counters=counters, prompt_tokens_by_source=sources,
        accepted_per_position=positions, draft_acceptance=accepted/drafted if drafted else None,
        mean_acceptance_length=1+accepted/rounds if rounds else None)


def request(url, payload, timeout):
    start = time.monotonic()
    output = dict(status=None, started_utc=now(), events=[], content='', reasoning='', finish_reason=None)
    first = last = None
    req = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            output['status'] = response.status
            for raw_line in response:
                timestamp = time.monotonic() - start
                line = raw_line.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                if line[5:].strip() == '[DONE]':
                    output['done_received'] = True
                    break
                event = json.loads(line[5:].strip())
                output['events'].append(dict(elapsed_seconds=timestamp, data=event))
                if event.get('usage'):
                    output['usage'] = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta') or {}
                    text = delta.get('content') or ''
                    reasoning = delta.get('reasoning') or delta.get('reasoning_content') or ''
                    if text or reasoning or delta.get('tool_calls'):
                        first = timestamp if first is None else first
                        last = timestamp
                    output['content'] += text
                    output['reasoning'] += reasoning
                    output['finish_reason'] = choice.get('finish_reason') or output['finish_reason']
    except urllib.error.HTTPError as error:
        output.update(status=error.code, error=error.read().decode('utf-8', 'replace'))
    except Exception as error:
        output['error'] = type(error).__name__ + ': ' + str(error)
    output.update(elapsed_seconds=time.monotonic()-start, ttft_seconds=first,
        last_update_seconds=last, update_window_seconds=last-first if first is not None and last is not None else None)
    completion = output.get('usage', {}).get('completion_tokens')
    output['short_output_decode_tps'] = ((completion-1)/(last-first)
        if completion and completion > 1 and first is not None and last is not None and last > first else None)
    return output


def run(args):
    corpus_bytes = args.corpus.read_bytes()
    corpus = json.loads(corpus_bytes)
    wanted = CASE_ORDER if args.cases == 'all' else args.cases.split(',')
    if len(set(wanted)) != len(wanted) or any(c not in CASE_ORDER for c in wanted):
        raise ValueError('Cases must be unique names from: ' + ','.join(CASE_ORDER))
    cases = [next(c for c in corpus['cases'] if c['name'] == name) for name in wanted]
    args.out.mkdir(parents=True, exist_ok=False)
    models = http_json(args.url, '/v1/models')
    save(args.out / 'models.json', models)
    served = next(m for m in models['data'] if m['id'] == corpus['model'])
    manifest = dict(arm=args.arm, arm_description=ARMS[args.arm], started_utc=now(), endpoint=args.url,
        corpus_path=str(args.corpus.resolve()), corpus_sha256=digest(corpus_bytes),
        harness_sha256=digest(Path(__file__).read_bytes()), cases=wanted,
        settings_source='Arm name is a label, not proof of backend dispatch; retain actual server config/logs',
        limitations=['One short marker answer per case; decode rate is not sustained throughput',
            'Same frozen payloads across arms; fixed correct assistant text in growing histories',
            'Only metrics identify actual cache hits; cold label alone is not proof',
            'No token-level numerical equivalence or broad quality claim'])
    if args.server_metadata:
        save(args.out / 'server-metadata.json', json.loads(args.server_metadata.read_text()))
    save(args.out / 'manifest.json', manifest)
    summary = []
    try:
        for index, case in enumerate(cases):
            if digest(canonical(case['payload'])) != case['payload_sha256']:
                raise ValueError('Frozen payload hash mismatch: ' + case['name'])
            if served.get('max_model_len') and case['prompt_tokens'] + case['payload']['max_tokens'] > served['max_model_len']:
                raise ValueError('Request exceeds current served context')
            stem = f'{index:02d}-{case["name"]}'
            save(args.out / (stem + '-request.json'), case['payload'])
            before = metrics(args.url, args.out / (stem + '-metrics-before.txt'))
            response = request(args.url, case['payload'], args.timeout)
            save(args.out / (stem + '-response.json'), response)
            after = metrics(args.url, args.out / (stem + '-metrics-after.txt'))
            try:
                content = response['content'].strip()
                if content.startswith('```json') and content.endswith('```'):
                    content = content[7:-3].strip()
                parsed = json.loads(content)
                correct = parsed == case['expected']
            except (ValueError, TypeError):
                parsed, correct = None, False
            usage = response.get('usage') or {}
            delta = metric_delta(before, after)
            result = dict(case=case['name'], status=response['status'], expected=case['expected'],
                actual=parsed, retrieval_pass=correct, finish_reason=response.get('finish_reason'),
                stream_done=response.get('done_received', False), expected_prompt_tokens=case['prompt_tokens'],
                prompt_tokens_match=usage.get('prompt_tokens') == case['prompt_tokens'], usage=usage,
                ttft_seconds=response['ttft_seconds'], elapsed_seconds=response['elapsed_seconds'],
                short_output_decode_tps=response['short_output_decode_tps'], metrics=delta,
                error=response.get('error'))
            result['passed'] = bool(correct and result['prompt_tokens_match'] and response['status'] == 200
                and result['stream_done'] and result['finish_reason'] == 'stop' and not result['error'])
            summary.append(result)
            save(args.out / 'summary.json', summary)
            print(json.dumps({key: result[key] for key in (
                'case', 'passed', 'ttft_seconds', 'elapsed_seconds',
                'short_output_decode_tps', 'usage', 'error')}), flush=True)
            if response['status'] != 200 or result['error']:
                break
        manifest['status'] = 'PASS' if len(summary) == len(cases) and all(r['passed'] for r in summary) else 'FAIL'
    except BaseException as error:
        manifest.update(status='FAIL', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        manifest['finished_utc'] = now()
        save(args.out / 'manifest.json', manifest)
    if manifest['status'] != 'PASS':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Root-run tokenizer-only preparation; no generation')
    p.add_argument('--url', type=local_url, default='http://127.0.0.1:18981')
    p.add_argument('--tokens', type=int, required=True)
    p.add_argument('--seed', type=int, default=20260917)
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--out', type=Path, required=True)
    p.set_defaults(action=prepare)
    p = sub.add_parser('run', help='Root-owned generation run against an already launched arm')
    p.add_argument('--url', type=local_url, default='http://127.0.0.1:18981')
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--cases', default='all')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--server-metadata', type=Path)
    p.add_argument('--timeout', type=float, default=600)
    p.set_defaults(action=run)
    p = sub.add_parser('check', help='CPU-only corpus structure check; makes no HTTP calls')
    def check(_):
        cases = build_cases(16, 20260917)
        assert [c['name'] for c in cases] == CASE_ORDER
        assert cases[0]['payload'] == cases[1]['payload']
        assert cases[3]['payload'] == cases[6]['payload']
        assert cases[2]['payload']['messages'][:2] == cases[0]['payload']['messages']
        assert cases[3]['payload']['messages'][:4] == cases[2]['payload']['messages']
        assert cases[5]['payload']['messages'][:2] == cases[0]['payload']['messages']
        assert len({tuple(c['expected'].items()) for c in cases}) == 5
        print(json.dumps(dict(status='PASS', cases=CASE_ORDER, network_calls=0)))
    p.set_defaults(action=check)
    args = parser.parse_args()
    if args.command == 'prepare' and (args.tokens < 512 or args.max_tokens < 96):
        parser.error('Use at least 512 prompt tokens and 96 output tokens')
    args.action(args)


if __name__ == '__main__':
    main()
