"""Terminal chat with the running local OpenAI-compatible server."""
import argparse,json,urllib.request
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8036');p.add_argument('--thinking',action='store_true');a=p.parse_args()
messages=[]
while True:
    try: line=input('You: ')
    except (EOFError,KeyboardInterrupt): break
    if not line.strip(): continue
    messages.append(dict(role='user',content=line))
    body=dict(model='minicpm5-2b',messages=messages,max_tokens=1024,temperature=0,
              chat_template_kwargs={'enable_thinking':a.thinking},stream=True)
    request=urllib.request.Request(a.url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    answer='';print('MiniCPM: ',end='',flush=True)
    with urllib.request.urlopen(request,timeout=180) as response:
        for event in response:
            if not event.startswith(b'data: ') or event.strip()==b'data: [DONE]':continue
            data=json.loads(event[6:]);choices=data.get('choices',[])
            if choices:
                delta=choices[0].get('delta',{}).get('content') or ''
                answer+=delta;print(delta,end='',flush=True)
    print();messages.append(dict(role='assistant',content=answer))
