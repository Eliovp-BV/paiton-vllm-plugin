"""Summarize raw full-pipeline measurements without excluding component movement."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--measured',type=int,nargs='+',default=[1,2]);a=p.parse_args()
e=[json.loads(x) for x in (a.directory/'events.jsonl').read_text().splitlines()]
t=[json.loads(x) for x in (a.directory/'telemetry.jsonl').read_text().splitlines()]
result={'measured_run_indices':a.measured,'runs':[],'startup_events':[x for x in e if x['event']=='setup_complete' or (x['event']=='phase_end' and x['run']==-1)]}
for c in [x for x in e if x['event']=='clip_complete']:
    run=c['run'];samples=[x for x in t if x['run']==run];phases=[x for x in e if x['event']=='phase_end' and x['run']==run]
    ev=next(x for x in e if x['event']=='denoiser_evaluations' and x['run']==run)
    row={**c,'measured':run in a.measured,'component_seconds':{x['phase']:x['seconds'] for x in phases},'component_disk_read_bytes':{x['phase']:x['read_bytes'] for x in phases},'actual_denoiser_evaluations':ev['count'],'nfe_gpu_ms':ev['gpu_ms']}
    row['sampled_peak']={k:max(int(x.get(k,0)) for x in samples) for k in ('VmRSS_bytes','RssAnon_bytes','VmSwap_bytes','mem_info_vram_used','temp1_input','temp2_input','temp3_input','power1_average')}
    row['sampled_host_available_min_bytes']=min(x['MemAvailable_bytes'] for x in samples)
    row['torch_peak_allocated_bytes']=max(x['gpu_peak_allocated'] for x in phases)
    row['torch_peak_reserved_bytes']=max(x['gpu_peak_reserved'] for x in phases)
    row['process_lifetime_high_water_mark_bytes']=max(x['host']['VmHWM_bytes'] for x in phases)
    row['aimdo_phase_end_max_bytes']=max(x.get('aimdo_vram') or 0 for x in phases)
    result['runs'].append(row)
measured=[x for x in result['runs'] if x['measured']]
if measured:
    result['mean_seconds']=statistics.mean(x['seconds'] for x in measured)
    result['mean_pipeline_before_encoding_seconds']=statistics.mean(x['pipeline_before_encoding_seconds'] for x in measured)
    result['mean_clips_per_hour']=3600/result['mean_seconds']
    result['mean_video_seconds_per_wall_second']=measured[0]['actual_frames']/measured[0]['fps']/result['mean_seconds']
(a.directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('runs','startup_events')},indent=2))
