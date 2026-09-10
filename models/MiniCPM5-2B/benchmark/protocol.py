"""Bounded API contract checks; retain complete requests and streamed responses."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import urllib.error


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8036')
    p.add_argument('--model', default='candidate')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    results = []

    def call(name, body, check, expected_status=200):
        payload = dict(model=a.model, max_tokens=512, temperature=0, seed=1201,
                       chat_template_kwargs={'enable_thinking': False})
        payload.update(body)
        started = time.perf_counter()
        record = dict(name=name, request=payload)
        request = urllib.request.Request(a.url+'/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                record['status'] = response.status
                if payload.get('stream'):
                    chunks = []
                    done = False
                    for line in response:
                        if line.strip() == b'data: [DONE]':
                            done = True
                        elif line.startswith(b'data: '):
                            chunks.append(dict(seconds=time.perf_counter()-started,
                                               data=json.loads(line[6:])))
                    record.update(chunks=chunks, done=done)
                else:
                    record['response'] = json.load(response)
        except urllib.error.HTTPError as error:
            record.update(status=error.code, error=error.read().decode())
        record['seconds'] = time.perf_counter()-started
        record['passed'] = record['status'] == expected_status and check(record)
        results.append(record)
        a.out.write_text(json.dumps(results, indent=2))
        return record

    def content(record):
        return record['response']['choices'][0]['message'].get('content') or ''

    stream = call('stream-eos', dict(messages=[dict(role='user', content='Return only the word Mercury.')],
        stream=True, stream_options={'include_usage':True}),
        lambda r: r['done'] and ''.join(c['data']['choices'][0].get('delta',{}).get('content') or ''
            for c in r['chunks'] if c['data'].get('choices')).strip()=='Mercury'
            and any(c['data'].get('choices') and c['data']['choices'][0].get('finish_reason')=='stop' for c in r['chunks']))
    call('multiturn', dict(messages=[dict(role='user',content='My project codename is Cedar. Reply OK.'),
        dict(role='assistant',content='OK'),dict(role='user',content='What is my project codename? Reply with the name only.')]),
        lambda r: content(r).strip()=='Cedar')
    notes = 'A workshop starts Tuesday at 10 AM and lasts two hours. '
    call('supplied-facts', dict(messages=[dict(role='system',content='Use only supplied facts. Never invent people, locations, prices or contact details. Use [placeholders] for missing facts. Return a short draft only.'),
        dict(role='user',content='Write a one-sentence invitation using these notes: '+notes)]),
        lambda r: 'Tuesday' in content(r) and '10' in content(r) and ('two' in content(r) or '2' in content(r)))
    filler = 'The project stores local notes on disk. The settings can be reviewed. '
    call('long-context-retrieval', dict(messages=[dict(role='user',content=
        filler*180+'\nThe access phrase for this exercise is amber-tulip-73.\n'+filler*180+
        '\nReturn only the access phrase from the notes.')]),
        lambda r: content(r).strip()=='amber-tulip-73')
    # Exercise the configured limit, not just the advertised 128K upstream limit.
    lo,hi=180,400
    while lo<hi:
        repeat=(lo+hi+1)//2
        text=filler*repeat+'\nThe access phrase is amber-tulip-73.\n'+filler*repeat+'\nReturn only the access phrase.'
        token_body=dict(model=a.model,messages=[dict(role='user',content=text)],
                        chat_template_kwargs={'enable_thinking':False},add_generation_prompt=True)
        token_request=urllib.request.Request(a.url+'/tokenize',data=json.dumps(token_body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(token_request,timeout=20) as response: count=json.load(response)['count']
        if count<=8050:lo=repeat
        else:hi=repeat-1
    text=filler*lo+'\nThe access phrase is amber-tulip-73.\n'+filler*lo+'\nReturn only the access phrase.'
    call('near-8k-context',dict(messages=[dict(role='user',content=text)],max_tokens=64),
         lambda r: content(r).strip()=='amber-tulip-73' and 7800<=r['response']['usage']['prompt_tokens']<=8128)
    call('context-rejection', dict(messages=[dict(role='user',content='hello')], max_tokens=9000),
        lambda r: 'context' in r.get('error','').lower() or 'max_tokens' in r.get('error',''), expected_status=400)
    tools = [dict(type='function', function=dict(name='weather', description='Get weather for a city.',
        parameters=dict(type='object',properties={'city':{'type':'string'}},required=['city']))) ]
    messages = [dict(role='user',content='Use the weather tool to check Paris.')]
    def check_tools(r):
        calls = {}
        for chunk in r['chunks']:
            for choice in chunk['data'].get('choices',[]):
                for value in choice.get('delta',{}).get('tool_calls') or []:
                    item = calls.setdefault(value['index'], {'name':'','arguments':'','id':''})
                    item['id'] += value.get('id') or ''
                    for key in ('name','arguments'):
                        item[key] += value.get('function',{}).get(key) or ''
        r['assembled_tools'] = calls
        try:
            return r['done'] and len(calls)==1 and calls[0]['name']=='weather' and json.loads(calls[0]['arguments'])=={'city':'Paris'}
        except (ValueError, KeyError):
            return False
    record = call('stream-tool', dict(messages=messages,tools=tools,tool_choice='auto',stream=True),check_tools)
    if record['passed']:
        tool = record['assembled_tools'][0]
        messages += [dict(role='assistant', content=None, tool_calls=[dict(id=tool['id'],type='function',
            function=dict(name=tool['name'],arguments=tool['arguments']))]),
            dict(role='tool', tool_call_id=tool['id'], content='{"city":"Paris","temperature_c":18,"condition":"clear"}'),
            dict(role='user',content='What is the temperature? Answer only with the number of degrees Celsius.')]
        call('tool-result',dict(messages=messages,tools=tools),lambda r: '18' in content(r))
    print(json.dumps({'passed':sum(r['passed'] for r in results),'total':len(results)}))


if __name__ == '__main__':
    main()
