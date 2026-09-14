"""Reproduce fixed-token NEO streaming comparisons; write raw results outside source."""
import argparse,json,pathlib,statistics,subprocess,time,urllib.request
p=argparse.ArgumentParser();p.add_argument('--engine',choices=['llama','paiton'],required=True);p.add_argument('--label',required=True);p.add_argument('--output-dir',type=pathlib.Path,required=True);p.add_argument('--base-url',required=True);p.add_argument('--owned-container',required=True);p.add_argument('--workloads',default='128:1,1024:1,4096:1,128:128,1024:128,4096:128');a=p.parse_args()
out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);rows=[]
if (out/(a.label+'-openai-raw.json')).exists():raise SystemExit('Choose a fresh label; existing raw results are preserved')
def guard_gpu():
 top=subprocess.check_output(['docker','top',a.owned_container,'-eo','pid'],text=True)
 owned={int(line.strip()) for line in top.splitlines()[1:] if line.strip().isdigit()}
 clients=pathlib.Path('/sys/class/kfd/kfd/proc')
 if not clients.is_dir():raise RuntimeError('Cannot inspect KFD ownership on this host')
 active={int(p.name) for p in clients.iterdir() if p.name.isdigit()}
 if not active or not active.issubset(owned):raise RuntimeError(f'Unexpected GPU ownership: {sorted(active-owned)}')
for inp,nout in [tuple(map(int,item.split(':'))) for item in a.workloads.split(',')]:
 for run in range(6):
  guard_gpu()
  body=dict(model='qwen38-neo',prompt=' vertex'*(inp-1)+[' amber',' cedar',' maple',' marble',' copper',' blue'][run],max_tokens=nout,temperature=0,seed=711,ignore_eos=True,stream=True,stream_options={'include_usage':True},logprobs=1)
  # Disable the llama.cpp slot's prompt reuse; vLLM prefix caching is disabled at startup.
  if a.engine=='llama':body['cache_prompt']=False
  request=urllib.request.Request(a.base_url.rstrip('/')+'/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  events=[];start=time.perf_counter()
  with urllib.request.urlopen(request,timeout=1200) as response:
   for line in response:
    if not line.startswith(b'data: '):continue
    elapsed=time.perf_counter()-start
    if line[6:].strip()==b'[DONE]':break
    events.append(dict(seconds=elapsed,data=json.loads(line[6:])))
  total=time.perf_counter()-start
  token_events=[e for e in events if any(c.get('text') or (c.get('logprobs') or {}).get('tokens') for c in e['data'].get('choices',[]))]
  usage=next((e['data']['usage'] for e in reversed(events) if e['data'].get('usage')),None)
  if not usage or usage['prompt_tokens']!=inp or usage['completion_tokens']!=nout:raise RuntimeError(f'unmatched token workload: {usage}')
  record=dict(prompt_contract='repeated vertex with six single-token suffixes; asserted API usage',engine=a.engine,label=a.label,input_tokens=inp,output_tokens=nout,run=run,warmup=run==0,total_seconds=total,ttft_seconds=token_events[0]['seconds'],usage=usage,request=body,events=events)
  if nout>1:record['decode_tokens_per_second']=(nout-1)/(token_events[-1]['seconds']-token_events[0]['seconds'])
  rows.append(record);(out/(a.label+'-openai-raw.json')).write_text(json.dumps(rows,indent=2));print(inp,nout,run,round(total,4),flush=True)
summary=[]
for inp,nout in sorted(set((r['input_tokens'],r['output_tokens']) for r in rows)):
 group=[r for r in rows if (r['input_tokens'],r['output_tokens'])==(inp,nout) and not r['warmup']];item=dict(input_tokens=inp,output_tokens=nout,samples=len(group))
 for metric in ['total_seconds','ttft_seconds']+(['decode_tokens_per_second'] if nout>1 else []):
  values=sorted(r[metric] for r in group);item[metric+'_median']=statistics.median(values);item[metric+'_p95']=values[3]*.2+values[4]*.8
 summary.append(item)
(out/(a.label+'-openai-summary.json')).write_text(json.dumps(summary,indent=2))
