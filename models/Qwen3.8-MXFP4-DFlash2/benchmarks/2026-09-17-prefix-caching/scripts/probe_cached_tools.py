#!/usr/bin/env python3
"""Local long-prefix tool continuation probe; only a harness-owned fixture is read.

The model sees /workspace/apc_probe.json, mapped to a new fixture under --out.
No model-selected path or arbitrary tool is executed. Server control and GPU
execution belong to the caller; --check is strictly CPU-only with no HTTP.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import secrets
import time
from unittest.mock import patch

import apc_benchmark as bench

PROBE_PATH = Path(__file__).with_name('probe_tools.py')
spec = importlib.util.spec_from_file_location('previous_tool_probe', PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
VIRTUAL_PATH = '/workspace/apc_probe.json'


def initial_payload(corpus):
    case = corpus['cases'][0]
    original = case['payload']['messages']
    if len(original) != 2 or original[0]['role'] != 'system' or original[1]['role'] != 'user':
        raise ValueError('Expected frozen system/archive first case')
    payload = copy.deepcopy(case['payload'])
    # Preserve the original system's retrieval instruction, replacing its output
    # schema and tail-update policy for this actual tool continuation probe.
    system = original[0]['content'].split(' Return only', 1)[0]
    system += (' Preserve the exact KEY_1, KEY_2 and KEY_3 archive strings. '
        'Before answering, call read_file for /workspace/apc_probe.json. '
        'The tool result supplies an unpredictable tool_phrase; never guess it. '
        'After the tool result, return only a JSON object with exactly KEY_1, '
        'KEY_2, KEY_3 and tool_phrase. Use the tool_phrase in this conversation\'s '
        'actual tool result. Background records do not alter the unique keys.')
    suffix = '\nReturn the five-key JSON object now. No explanation.'
    archive = original[1]['content']
    if not archive.endswith(suffix):
        raise ValueError('Frozen archive ending differs from expected corpus')
    archive = archive[:-len(suffix)] + ('\nUse read_file now to read ' + VIRTUAL_PATH +
        '. You need its actual contents before returning the four-key JSON. Do not guess the file contents.')
    payload.update(messages=[dict(role='system', content=system), dict(role='user', content=archive)],
        tools=[copy.deepcopy(probe.READ)], tool_choice='auto', temperature=0, top_p=1,
        seed=42, max_tokens=256, stream=True, stream_options=dict(include_usage=True),
        chat_template_kwargs=dict(enable_thinking=False))
    return payload, {key: case['expected'][key] for key in ('KEY_1', 'KEY_2', 'KEY_3')}


def validate_call(response):
    calls = response.get('tool_calls') or []
    if response.get('status') != 200 or response.get('finish_reason') != 'tool_calls' or len(calls) != 1:
        raise ValueError('Expected exactly one successful read_file tool call')
    call = calls[0]
    arguments = json.loads(call['function']['arguments'])
    if (call.get('type') != 'function' or not call.get('id')
            or call['function']['name'] != 'read_file' or arguments != {'path': VIRTUAL_PATH}):
        raise ValueError('Tool call is outside the one allowed fixture read')
    return calls


def continuation(payload, first_response, fixture):
    calls = validate_call(first_response)
    # Read only the harness-owned path passed by our caller. Never resolve the
    # model's path string against the filesystem.
    actual_contents = fixture.read_text()
    tool_data = json.loads(actual_contents)
    follow = copy.deepcopy(payload)
    assistant = dict(role='assistant', content=first_response.get('content') or None,
                     tool_calls=copy.deepcopy(calls))
    if first_response.get('reasoning'):
        assistant['reasoning'] = first_response['reasoning']
    follow['messages'] += [assistant, dict(role='tool', tool_call_id=calls[0]['id'], content=actual_contents),
        dict(role='user', content='Return only the four-key JSON using the original archive keys '
             'and the exact tool_phrase from the tool result above. Do not call any more tools.')]
    return follow, tool_data['tool_phrase']


def capture_request(url, payload, timeout, sse_path):
    original_open = probe.urllib.request.urlopen

    class TeeResponse:
        def __init__(self, response, output):
            self.response, self.output = response, output

        def __enter__(self):
            self.response.__enter__()
            return self

        def __exit__(self, *args):
            return self.response.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.response, name)

        def __iter__(self):
            for line in self.response:
                self.output.write(line)
                yield line

    with sse_path.open('wb') as output:
        def opened(*args, **kwargs):
            return TeeResponse(original_open(*args, **kwargs), output)
        start = time.monotonic()
        try:
            with patch.object(probe.urllib.request, 'urlopen', opened):
                result = probe.request(url, payload, timeout=timeout)
        except Exception as error:
            result = dict(status=None, error=type(error).__name__ + ': ' + str(error),
                          elapsed_seconds=time.monotonic() - start)
    result['stream_done'] = any(line.strip() == b'data: [DONE]' for line in sse_path.read_bytes().splitlines())
    usage = [event['usage'] for event in result.get('events', []) if event.get('usage')]
    result['usage'] = usage[-1] if usage else None
    return result


def run(args):
    corpus_bytes = args.corpus.read_bytes()
    corpus = json.loads(corpus_bytes)
    payload, markers = initial_payload(corpus)
    args.out.mkdir(parents=True, exist_ok=False)
    fixture = args.out / 'apc_probe.json'
    manifest = dict(started_utc=bench.now(), arm=args.arm, endpoint=args.url,
        corpus=str(args.corpus.resolve()), corpus_sha256=bench.digest(corpus_bytes),
        harness_sha256=bench.digest(Path(__file__).read_bytes()),
        tool_request_helper=str(PROBE_PATH), tool_request_helper_sha256=bench.digest(PROBE_PATH.read_bytes()),
        virtual_fixture=VIRTUAL_PATH, actual_fixture=str(fixture.resolve()),
        scope='Three streamed requests: actual tool call, actual fixture result continuation, '
              'then same assistant tool call with changed fixture result; short marker correctness only',
        comparability='Fixture phrases are freshly generated per run; this is a functional cache '
                      'freshness probe, not an identical-payload speed comparison',
        status='RUNNING')
    bench.save(args.out / 'manifest.json', manifest)
    summary = []

    def measure(name, request, expected=None):
        bench.save(args.out / (name + '-request.json'), request)
        before = bench.metrics(args.url, args.out / (name + '-metrics-before.txt'))
        response = capture_request(args.url, request, args.timeout, args.out / (name + '-raw.sse'))
        bench.save(args.out / (name + '-response.json'), response)
        after = bench.metrics(args.url, args.out / (name + '-metrics-after.txt'))
        parsed, error = None, None
        try:
            if expected is None:
                validate_call(response)
                correct = True
            else:
                content = (response.get('content') or '').strip()
                if content.startswith('```json') and content.endswith('```'):
                    content = content[7:-3].strip()
                parsed = json.loads(content)
                correct = parsed == expected and not response.get('tool_calls') and response.get('finish_reason') == 'stop'
        except (ValueError, TypeError, KeyError) as exception:
            correct, error = False, str(exception)
        result = dict(case=name, status=response.get('status'), actual=parsed, expected=expected,
            ttft_seconds=response.get('ttft_seconds'), elapsed_seconds=response.get('elapsed_seconds'),
            finish_reason=response.get('finish_reason'), stream_done=response.get('stream_done'),
            usage=response.get('usage'), metrics=bench.metric_delta(before, after),
            error=response.get('error') or error)
        result['passed'] = bool(correct and response.get('status') == 200 and response.get('stream_done') and not result['error'])
        summary.append(result)
        bench.save(args.out / 'summary.json', summary)
        print(json.dumps(result), flush=True)
        if not result['passed']:
            raise ValueError('Tool continuation probe failed: ' + name)
        return response

    try:
        first = measure('00-read-tool', payload)
        phrases = []
        for name in ('01-tool-result-a', '02-tool-result-b'):
            phrase = 'fixture-' + secrets.token_hex(12)
            if phrase in json.dumps(payload) or phrase in phrases:
                raise RuntimeError('Unexpected fixture phrase collision')
            phrases.append(phrase)
            fixture.write_text(json.dumps({'tool_phrase': phrase}) + '\n')
            fixture_snapshot = args.out / (name + '-fixture.json')
            fixture_snapshot.write_bytes(fixture.read_bytes())
            follow, actual_phrase = continuation(payload, first, fixture)
            measure(name, follow, {**markers, 'tool_phrase': actual_phrase})
        manifest['status'] = 'PASS'
    except BaseException as error:
        manifest.update(status='FAIL', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        manifest['finished_utc'] = bench.now()
        bench.save(args.out / 'manifest.json', manifest)


def check():
    import tempfile
    corpus = dict(cases=bench.build_cases(16, 20260917))
    payload, markers = initial_payload(corpus)
    first = dict(status=200, finish_reason='tool_calls', content='', tool_calls=[dict(
        id='test-call', type='function', function=dict(name='read_file', arguments=json.dumps({'path': VIRTUAL_PATH})))])
    with tempfile.TemporaryDirectory() as temporary:
        fixture = Path(temporary) / 'apc_probe.json'
        fixture.write_text('{"tool_phrase":"first-fixture-only"}')
        left, phrase = continuation(payload, first, fixture)
        fixture.write_text('{"tool_phrase":"second-fixture-only"}')
        right, other = continuation(payload, first, fixture)
    assert phrase != other and left['messages'][:3] == right['messages'][:3]
    assert left['messages'][2]['tool_calls'] == first['tool_calls']
    assert phrase not in json.dumps(payload) and other not in json.dumps(payload)
    assert set(markers) == {'KEY_1', 'KEY_2', 'KEY_3'}
    invalid = copy.deepcopy(first)
    invalid['tool_calls'][0]['function']['arguments'] = '{"path":"/etc/passwd"}'
    try:
        validate_call(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Unexpected model-selected path accepted')
    print(json.dumps(dict(status='PASS', network_calls=0, actual_fixture_reads=2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', type=bench.local_url, default='http://127.0.0.1:18981')
    parser.add_argument('--corpus', type=Path, default=Path(__file__).with_name('corpus-40k.json'))
    parser.add_argument('--arm', choices=bench.ARMS, default='C')
    parser.add_argument('--out', type=Path)
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
