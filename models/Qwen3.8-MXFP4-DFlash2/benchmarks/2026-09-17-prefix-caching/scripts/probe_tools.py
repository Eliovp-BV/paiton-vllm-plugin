"""Read-only local API diagnostic. Tool results are simulated, never executed."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

MODEL = 'Qwen3.8-27B-Quark-AWQ-MXFP4'
READ = dict(type='function', function=dict(name='read_file',
    description='Read a UTF-8 text file and return its exact contents.',
    parameters=dict(type='object', properties=dict(path=dict(type='string')),
                    required=['path'], additionalProperties=False)))
WRITE = dict(type='function', function=dict(name='write_file',
    description='Write a UTF-8 text file.',
    parameters=dict(type='object', properties=dict(path=dict(type='string'), content=dict(type='string')),
                    required=['path', 'content'], additionalProperties=False)))


def request(url, payload, timeout=180):
    start = time.monotonic()
    req = urllib.request.Request(url + '/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    events, content, reasoning, calls, finish, first = [], '', '', {}, None, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if not payload.get('stream'):
                raw = json.load(response)
                msg = raw['choices'][0]['message']
                return dict(status=response.status, raw=raw, content=msg.get('content'),
                    reasoning=msg.get('reasoning') or msg.get('reasoning_content'),
                    tool_calls=msg.get('tool_calls') or [], finish_reason=raw['choices'][0]['finish_reason'],
                    elapsed_seconds=time.monotonic()-start)
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith('data: '):
                    continue
                if line[6:] == '[DONE]':
                    break
                event = json.loads(line[6:]); events.append(event)
                if not event.get('choices'):
                    continue
                choice = event['choices'][0]
                delta = choice.get('delta', {})
                if first is None and any(delta.get(key) for key in ('content', 'reasoning', 'reasoning_content', 'tool_calls')):
                    first = time.monotonic()-start
                content += delta.get('content') or ''
                reasoning += delta.get('reasoning') or delta.get('reasoning_content') or ''
                for tool in delta.get('tool_calls') or []:
                    assembled = calls.setdefault(tool['index'], dict(id='', type='function', function=dict(name='', arguments='')))
                    if tool.get('id'):
                        assembled['id'] = tool['id']
                    function = tool.get('function') or {}
                    assembled['function']['name'] += function.get('name') or ''
                    assembled['function']['arguments'] += function.get('arguments') or ''
                finish = choice.get('finish_reason') or finish
        return dict(status=200, events=events, content=content, reasoning=reasoning,
            tool_calls=[calls[i] for i in sorted(calls)], finish_reason=finish,
            ttft_seconds=first, elapsed_seconds=time.monotonic()-start)
    except urllib.error.HTTPError as error:
        return dict(status=error.code, error=error.read().decode(), elapsed_seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:18981')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    cases = [
        ('read', READ, 'Call read_file now to read /workspace/config.json. You need the file contents to answer; do not guess them.'),
        ('write', WRITE, 'Use write_file to create /workspace/hello.txt containing exactly these two lines: Hello, "Ada"!\nSecond line. Do not merely show the intended call; invoke the tool.'),
    ]
    summary = []
    for label, tool, prompt in cases:
        for stream in (False, True):
            name = label + ('-stream' if stream else '-ordinary')
            payload = dict(model=MODEL, messages=[dict(role='user', content=prompt)],
                tools=[tool], tool_choice='auto', temperature=.1, top_p=.95, seed=42,
                max_tokens=512, stream=stream, chat_template_kwargs=dict(enable_thinking=False))
            result = request(args.url, payload)
            valid = False
            try:
                calls = result['tool_calls']
                arguments = json.loads(calls[0]['function']['arguments']) if len(calls) == 1 else {}
                expected_path = '/workspace/config.json' if label == 'read' else '/workspace/hello.txt'
                valid = (len(calls) == 1 and calls[0]['function']['name'] == tool['function']['name']
                    and isinstance(arguments, dict) and arguments.get('path') == expected_path
                    and (label == 'read' or arguments.get('content', '').rstrip('\n') == 'Hello, "Ada"!\nSecond line.')
                    and result['finish_reason'] == 'tool_calls')
            except (KeyError, ValueError):
                pass
            record = dict(name=name, request=payload, response=result, parsed_tool_valid=valid)
            (args.out / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
            summary.append(dict(name=name, valid=valid, finish=result.get('finish_reason'),
                elapsed_seconds=result['elapsed_seconds'], status=result['status']))
            print(json.dumps(summary[-1]), flush=True)
            if label == 'read' and valid:
                follow = dict(payload)
                follow['messages'] = payload['messages'] + [dict(role='assistant',
                    content=result.get('content') or None, tool_calls=calls),
                    dict(role='tool', tool_call_id=calls[0]['id'],
                         content='{"mode":"safe","verification_phrase":"cobalt lantern 731"}'),
                    dict(role='user', content='What exact verification_phrase did the file contain? Reply with only that phrase.')]
                follow['max_tokens'] = 64
                second = request(args.url, follow)
                (args.out / (name + '-roundtrip.json')).write_text(json.dumps(
                    dict(request=follow, response=second, expected_phrase='cobalt lantern 731',
                         phrase_seen='cobalt lantern 731' in (second.get('content') or '')), indent=2)+'\n')
    # Thinking is deliberately left at the checkpoint default in this control.
    payload = dict(model=MODEL, messages=[dict(role='user', content=cases[0][2])],
        tools=[READ], tool_choice='auto', temperature=.1, seed=42, max_tokens=512, stream=True)
    result = request(args.url, payload)
    (args.out / 'default-thinking-stream.json').write_text(json.dumps(dict(request=payload,response=result),indent=2)+'\n')
    payload['chat_template_kwargs'] = dict(enable_thinking=True, reasoning_effort='low')
    result = request(args.url, payload)
    (args.out / 'thinking-enabled-stream.json').write_text(json.dumps(dict(request=payload,response=result),indent=2)+'\n')
    (args.out / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    main()
