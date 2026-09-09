"""Fresh-conditioning stock Wan TI2V benchmark through pinned Comfy nodes."""
import argparse,json,os,sys,time,threading,resource
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--transfer-profile',action='store_true')
p.add_argument('--data',type=Path,default=Path(os.environ.get('PAITON_WAN_DATA','/data')))
p.add_argument('--comfy-root',type=Path,default=Path(os.environ.get('PAITON_COMFY_ROOT','/opt/comfyui')))
p.add_argument('--fastwan',action='store_true');p.add_argument('--paiton-artifact',type=Path);p.add_argument('--paiton-runtime',type=Path);p.add_argument('--warmups',type=int,default=1);p.add_argument('--output',type=Path,required=True);p.add_argument('--runs',type=int,default=3)
p.add_argument('--width',type=int,default=832);p.add_argument('--height',type=int,default=480)
p.add_argument('--frames',type=int,default=49);p.add_argument('--steps',type=int,default=20)
p.add_argument('--seed',type=int,default=1201);p.add_argument('--guidance',type=float,default=5.)
p.add_argument('--attention',choices=['sdpa','ck'],default='ck');p.add_argument('--dtype',choices=['fp16','bf16'],default='bf16')
p.add_argument('--paiton-vae-artifact',type=Path);p.add_argument('--vae-events',action='store_true');p.add_argument('--miopen',action='store_true');p.add_argument('--vae-dtype',choices=['bf16','fp16','fp32'],default='bf16');p.add_argument('--image',type=Path);p.add_argument('--profile',action='store_true');p.add_argument('--compile',action='store_true')
p.add_argument('--native-fp8',action='store_true');p.add_argument('--weight-dtype',default='default',choices=['default','fp8_e4m3fn_fast']);p.add_argument('--profile-vae',action='store_true');p.add_argument('--tiled',action='store_true');p.add_argument('--prompt',default='A small red fox walks across a sunlit woodland clearing, pauses and turns its head toward the camera. Detailed natural fur, coherent anatomy, smooth continuous motion. The camera slowly tracks right.')
a=p.parse_args()
if a.fastwan:
 a.steps=3;a.guidance=1.
 if a.image:raise ValueError('FastWan preset is text-only')
a.output=a.output.resolve()
if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(f'Output directory must be new or empty: {a.output}')
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'command.json').write_text(json.dumps({'argv':sys.argv,'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}},indent=2))
ROOT=a.data.resolve();COMFY=a.comfy_root.resolve()
os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL','1');os.environ.setdefault('MIOPEN_FIND_MODE','FAST')
if a.miopen:os.environ['COMFYUI_ENABLE_MIOPEN']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(ROOT/'cache/inductor'))
sys.path.insert(0,str(COMFY));sys.argv=['main.py','--base-directory',str(ROOT),'--disable-all-custom-nodes','--disable-api-nodes','--use-ck-attention' if a.attention=='ck' else '--use-pytorch-cross-attention','--'+a.dtype+'-unet','--'+a.vae_dtype+'-vae','--fast-disk','--reserve-vram','2','--cache-none','--preview-method','none']
if a.native_fp8:sys.argv.append('--supports-fp8-compute')
startup=time.perf_counter()
import main as comfy_main
import torch,nodes
import comfy.model_management as mm
from comfy_extras.nodes_wan import Wan22ImageToVideoLatent
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from comfy_extras.nodes_custom_sampler import BasicScheduler
import numpy as np
from PIL import Image
import av

torch.set_num_threads(4)
events=(a.output/'events.jsonl').open('a',buffering=1);tele=(a.output/'telemetry.jsonl').open('a',buffering=1)
state={'run':-1,'phase':'setup'};stop=threading.Event();phases={};results=[]
def emit(event,**kw):
 row={'event':event,'time':time.time(),**state,**kw};events.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
def snapshot():
 r={}
 for line in Path('/proc/self/status').read_text().splitlines():
  key,_,val=line.partition(':')
  if key in ['VmRSS','VmHWM','VmSwap','RssAnon','RssFile']:r[key]=int(val.split()[0])*1024
 for line in Path('/proc/meminfo').read_text().splitlines():
  key,_,val=line.partition(':')
  if key in ['MemAvailable','SwapFree']:r[key]=int(val.split()[0])*1024
 devices=[c/'device' for c in Path('/sys/class/drm').glob('card[0-9]*') if (c/'device/mem_info_vram_used').exists()]
 dev=devices[0] if len(devices)==1 else Path('/nonexistent')
 for f in ['mem_info_vram_used','gpu_busy_percent','pp_dpm_sclk','pp_dpm_mclk']:
  try:r[f]=(dev/f).read_text().strip()
  except OSError:pass
 for hw in (dev/'hwmon').glob('hwmon*'):
  for f in ['temp1_input','temp2_input','temp3_input','power1_average','freq1_input','freq2_input']:
   try:r[f]=int((hw/f).read_text())
   except OSError:pass
 return r
def monitor():
 while not stop.wait(.5):
  r=snapshot();tele.write(json.dumps({'time':time.time(),**state,**r})+'\n')
  if r['MemAvailable']<350*2**20 and r['SwapFree']<512*2**20:os.kill(os.getpid(),2);return
threading.Thread(target=monitor,daemon=True).start()
def phase(name,fn):
 state['phase']=name;torch.cuda.synchronize();t=time.perf_counter();torch.cuda.reset_peak_memory_stats();emit('phase_start')
 prof=None
 if a.transfer_profile and state['run']==2:
  prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False);prof.__enter__()
 elif a.profile_vae and name=='decode' and state['run']==0:
  prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True);prof.__enter__()
 out=fn();torch.cuda.synchronize();dt=time.perf_counter()-t;phases[name]=dt
 if prof and a.transfer_profile:
  prof.__exit__(None,None,None)
  rows=[{'name':e.name,'device_type':str(e.device_type),'microseconds':e.device_time_total,'cpu_microseconds':e.cpu_time_total} for e in prof.events() if any(k in e.name.lower() for k in ('memcpy','memcopy'))]
  (a.output/f'transfers-{name}.json').write_text(json.dumps(rows,indent=2))
 elif prof:
  prof.__exit__(None,None,None);prof.export_chrome_trace(str(a.output/'vae-trace.json'));(a.output/'vae-profile.txt').write_text(prof.key_averages(group_by_input_shape=True).table(sort_by='self_device_time_total',row_limit=60))
 emit('phase_end',seconds=dt,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),host=snapshot());return out
try:
 with torch.inference_mode():
  emit('startup',seconds=time.perf_counter()-startup,torch=torch.__version__,hip=torch.version.hip)
  clip=phase('load_encoder',lambda:nodes.CLIPLoader().load_clip('umt5_xxl_fp8_e4m3fn_scaled.safetensors','wan','default')[0])
  vae=phase('load_vae',lambda:nodes.VAELoader().load_vae('wan2.2_vae.safetensors')[0])
  if a.paiton_vae_artifact:
   sys.path.insert(0,str(a.paiton_runtime))
   from paiton_wan.vae import install as install_vae
   vae=install_vae(vae,a.paiton_vae_artifact)
  vae_events=[]
  if a.vae_events:
   for module_name,module in vae.first_stage_model.named_modules():
    if isinstance(module,(torch.nn.Conv2d,torch.nn.Conv3d)) or type(module).__name__=='RMS_norm':
     base_forward=module.forward
     def measured(*args,_forward=base_forward,_name=module_name,_module=module,**kwargs):
      if state['run']!=1:return _forward(*args,**kwargs)
      begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
      out=_forward(*args,**kwargs);end.record()
      vae_events.append((begin,end,{'module':_name,'type':type(_module).__name__,'input':list(args[0].shape),'output':list(out.shape),'weight':list(_module.weight.shape) if hasattr(_module,'weight') else None}))
      return out
     module.forward=measured
  model=phase('load_denoiser',lambda:nodes.UNETLoader().load_unet('fastwan22_5b_fullattn_comfy_bf16.safetensors' if a.fastwan else 'wan2.2_ti2v_5B_fp16.safetensors',a.weight_dtype)[0])
  model=ModelSamplingSD3().patch(model,8)[0]
  if a.paiton_artifact:
   sys.path.insert(0,str(a.paiton_runtime))
   from paiton_wan.fusions import install
   model=install(model,a.paiton_artifact)
  dm=model.model.diffusion_model;original=dm.forward;evals=[]
  if a.compile:
   for b in dm.blocks:b.forward=torch.compile(b.forward,fullgraph=False,dynamic=False)
  def forward(*args,**kwargs):
   prof=None
   if a.profile and state['run']==0 and len(evals)==2:
    prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True);prof.__enter__()
   t,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);t.record();out=original(*args,**kwargs);b.record();evals.append((t,b,list(args[0].shape) if args else []))
   if prof:
    prof.__exit__(None,None,None);prof.export_chrome_trace(str(a.output/'denoiser-trace.json'));(a.output/'denoiser-profile.txt').write_text(prof.key_averages(group_by_input_shape=True).table(sort_by='self_device_time_total',row_limit=60))
   return out
  dm.forward=forward
  sigmas=BasicScheduler.execute(model,'simple',a.steps,1.)[0]
  if a.fastwan:
   sys.path.insert(0,str(a.paiton_runtime))
   from paiton_wan.dmd import training_table,TIMESTEPS
   ts,ss=training_table();emit('schedule',dmd_timesteps=TIMESTEPS,training_shift=8,sigmas=[float(ss[(ts-t).abs().argmin()]) for t in TIMESTEPS])
  else:emit('schedule',sigmas=sigmas.tolist())
  img=torch.from_numpy(np.array(Image.open(a.image).convert('RGB')).astype(np.float32)/255.)[None] if a.image else None
  negative='Overexposed, static, blurry details, subtitles, painting, still image, gray cast, worst quality, low quality, JPEG artifacts, deformed, disfigured, extra limbs, fused fingers, cluttered background, three legs, backwards walking.'
  for run in range(a.runs):
   state['run']=run;evals.clear();phases={}
   if a.paiton_artifact:model.paiton_wan_fusions.calls=0
   if a.paiton_vae_artifact:vae.paiton_wan_vae.calls=0
   start=time.perf_counter()
   def cond():
    pos=nodes.CLIPTextEncode().encode(clip,a.prompt)[0];neg=nodes.CLIPTextEncode().encode(clip,negative)[0] if not a.fastwan else None
    latent=Wan22ImageToVideoLatent.execute(vae,a.width,a.height,a.frames,1,img)[0]
    return pos,neg,latent
   pos,neg,latent=phase('conditioning',cond)
   if a.fastwan:
    if a.image:raise ValueError('FastWan text-only gate does not accept an image')
    sys.path.insert(0,str(a.paiton_runtime))
    from paiton_wan.dmd import sample
    samples=phase('denoise',lambda:sample(model,pos,a.seed,a.width,a.height,a.frames))
   else:
    samples=phase('denoise',lambda:nodes.KSampler().sample(model,a.seed,a.steps,a.guidance,'uni_pc','simple',pos,neg,latent,denoise=1.)[0])
   pixels=phase('decode',lambda:vae.decode_tiled(samples['samples'],tile_x=32,tile_y=32,overlap=8) if a.tiled else vae.decode(samples['samples']))
   if a.vae_events and run==1:(a.output/'vae-events.json').write_text(json.dumps([dict(r,milliseconds=b.elapsed_time(e)) for b,e,r in vae_events],indent=2))
   if pixels.ndim==5: pixels=pixels.reshape(-1,*pixels.shape[-3:])
   pipeline=time.perf_counter()-start
   def encode():
    arr=(pixels.cpu().numpy()*255.).clip(0,255).astype(np.uint8);path=a.output/f'clip-{run}.mp4'
    with av.open(str(path),'w') as c:
     s=c.add_stream('libx264',rate=24);s.width=arr.shape[2];s.height=arr.shape[1];s.pix_fmt='yuv420p';s.options={'crf':'18','preset':'fast'}
     for frame in arr:
      for pkt in s.encode(av.VideoFrame.from_ndarray(frame,format='rgb24')):c.mux(pkt)
     for pkt in s.encode():c.mux(pkt)
    return str(path)
   path=phase('encoding',encode);total=time.perf_counter()-start
   row={'run':run,'warmup':run<a.warmups,'prompt':a.prompt,'seed':a.seed,'clip':path,'width':pixels.shape[2],'height':pixels.shape[1],'frames':pixels.shape[0],'fps':24,'duration':pixels.shape[0]/24,'pipeline_seconds':pipeline,'end_to_end_seconds':total,'clips_per_hour':3600/total,'video_seconds_per_wall_second':pixels.shape[0]/24/total,'phases':dict(phases),'paiton_vae_calls':vae.paiton_wan_vae.calls if a.paiton_vae_artifact else 0,'paiton_fusion_calls':model.paiton_wan_fusions.calls if a.paiton_artifact else 0,'evaluations':len(evals),'evaluation_ms':[t.elapsed_time(b) for t,b,_ in evals],'evaluation_shapes':[s for _,_,s in evals],'finite':bool(pixels.isfinite().all()),'peak_host_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
   results.append(row);(a.output/'results.json').write_text(json.dumps(results,indent=2));emit('clip_complete',**row)
   torch.save(samples['samples'].cpu(),a.output/f'latents-{run}.pt')
   del samples,pixels,pos,neg,latent
finally:stop.set();events.close()
