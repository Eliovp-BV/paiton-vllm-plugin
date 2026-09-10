"""Alternating complete-pipeline runs; fresh processes, persistent kernel cache."""
import argparse,json,os,statistics,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--launcher',required=True);p.add_argument('--recording',required=True);p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--summary-backend',choices=['transformers','vllm']);a=p.parse_args()
if a.repeats<3:p.error('--repeats must be at least 3 for qualification')
root=Path(a.output);root.mkdir(mode=0o700,exist_ok=False)
def process_rows(pid):
 rows=[]
 try:
  path=Path('/proc')/str(pid);fields={line.split(':')[0]:line.split(':',1)[1].strip() for line in (path/'status').read_text().splitlines()}
  rows.append(dict(pid=pid,name=fields.get('Name'),rss_bytes=int(fields.get('VmRSS','0 kB').split()[0])*1024,hwm_bytes=int(fields.get('VmHWM','0 kB').split()[0])*1024))
  for child in (path/'task'/str(pid)/'children').read_text().split():rows.extend(process_rows(int(child)))
 except (OSError,ValueError):pass
 return rows

rows=[]
# First pair reports first processing in this output series, not an empty OS/model cache.
for repeat in range(a.repeats+1):
 for backend in (('stock','paiton') if repeat%2==0 else ('paiton','stock')):
  output=root/f'{backend}-{repeat}';log=root/f'{backend}-{repeat}.log'
  command=[a.launcher]+(['--stock'] if backend=='stock' else [])+[a.recording,str(output),'--keep-intermediates']
  if a.summary_backend:command+=['--summary-backend',a.summary_backend]
  started=time.time()
  with log.open('x') as stream:
   run=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=os.environ)
   with (root/f'{backend}-{repeat}-telemetry.jsonl').open('x') as telemetry:
    while run.poll() is None:
     name=f'paiton-meeting-{os.getuid()}-{run.pid}'
     inspected=subprocess.run(['docker','inspect','--format','{{.State.Pid}}',name],capture_output=True,text=True)
     if inspected.returncode==0 and int(inspected.stdout.strip() or 0)>0:
      processes=process_rows(int(inspected.stdout.strip()))
      with (root/'process-memory.jsonl').open('a') as memory:
       memory.write(json.dumps(dict(time=time.time(),container=name,processes=processes,rss_sum_bytes=sum(r['rss_bytes'] for r in processes)))+'\n')
     cards={str(v.parent.parent.name):int(v.read_text()) for v in Path('/sys/class/drm').glob('card*/device/mem_info_vram_used')}
     mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:','SwapFree:'))}
     telemetry.write(json.dumps(dict(time=time.time(),driver_vram=cards,host=mem))+'\n');telemetry.flush();time.sleep(.5)
  if run.returncode:raise RuntimeError(f'{backend}-{repeat} failed: see {log}')
  result=json.loads((output/'result.json').read_text())
  row=dict(backend=backend,repeat=repeat,cached=repeat>0,launcher_wall_seconds=time.time()-started,**result['pipeline_timings'])
  rows.append(row);(root/'timings.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
report={}
for backend in ('stock','paiton'):
 values=[r['complete_pipeline_seconds'] for r in rows if r['backend']==backend and r['cached']]
 report[backend]=dict(n=len(values),median=statistics.median(values),stddev=statistics.stdev(values),minimum=min(values),maximum=max(values))
report['wall_time_reduction_percent']=100*(1-report['paiton']['median']/report['stock']['median'])
report['scenario']='Complete recording pipeline with sequential stage processes and persistent file/kernel caches. Launcher wall time can include shared GPU lease waiting; pipeline time excludes waiting and Docker startup.'
(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
