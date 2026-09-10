"""Terminal chat with the running local OpenAI-compatible server."""
import argparse,json,urllib.request
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8036');p.add_argument('--thinking',action='store_true',help='Enable experimental W4 thinking mode');p.add_argument('--max-tokens',type=int,default=1024,help='Total output budget, including thinking (default: 1024)');p.add_argument('--timeout',type=float,default=180,help='HTTP timeout in seconds (default: 180)');a=p.parse_args()
if a.max_tokens < 1 or a.timeout <= 0:p.error('--max-tokens and --timeout must be positive')
messages=[]
while True:
    try: line=input('You: ')
    except (EOFError,KeyboardInterrupt): break
    if not line.strip(): continue
    messages.append(dict(role='user',content=line))
    body=dict(model='minicpm5-2b',messages=messages,max_tokens=a.max_tokens,temperature=0,
              chat_template_kwargs={'enable_thinking':a.thinking},stream=True)
    request=urllib.request.Request(a.url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    answer='';print('MiniCPM: ',end='',flush=True)
    with urllib.request.urlopen(request,timeout=a.timeout) as response:
        for event in response:
            if not event.startswith(b'data: ') or event.strip()==b'data: [DONE]':continue
            data=json.loads(event[6:]);choices=data.get('choices',[])
            if choices:
                delta=choices[0].get('delta',{}).get('content') or ''
                answer+=delta;print(delta,end='',flush=True)
    print();messages.append(dict(role='assistant',content=answer))
