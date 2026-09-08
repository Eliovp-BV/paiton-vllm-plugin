"""Run the stock Comfy H3 nodes directly, with phase and device telemetry.

No cached conditioning is reused between runs. Model loading is reported
separately, and component movement during subsequent runs stays in the clock.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

ROOT = Path(os.environ.get("PAITON_H3_DATA", "/data")).resolve()
COMFY_ROOT = Path(os.environ.get("PAITON_COMFY_ROOT", "/opt/comfyui")).resolve()
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cases',type=Path)
p.add_argument('--prompt', default='A small red fox trots through a sunlit woodland clearing, pauses beside a shallow stream, and looks toward the camera. Natural realistic fur, continuous gentle camera movement, coherent anatomy. Audio: birds chirping, leaves rustling and quiet flowing water; no music and no speech.')
p.add_argument('--seed', type=int, default=771)
p.add_argument('--runs', type=int, default=1)
p.add_argument('--width', type=int, default=864)
p.add_argument('--height', type=int, default=480)
p.add_argument('--frames', type=int, default=124)
p.add_argument('--steps', type=int, default=20)
p.add_argument('--setup-only', action='store_true')
p.add_argument('--encoder-only', action='store_true')
p.add_argument('--disable-dynamic-vram', action='store_true')
p.add_argument('--tiled-vae', action='store_true')
p.add_argument('--enable-triton-backend', action='store_true')
p.add_argument('--attention',choices=['sdpa','ck'],default='sdpa')
p.add_argument('--fast-disk',action='store_true')
p.add_argument('--profile-step',type=int,default=-1)
p.add_argument('--smart-memory',action='store_true')
p.add_argument('--denoiser',default='minimax_h3_fl2va_pruned_int8_convrot.safetensors')
p.add_argument('--video-vae',default='minimax_h3_video_vae_fp16.safetensors')
p.add_argument('--lora')
p.add_argument('--lora-mode',choices=['bypass','merge'],default='bypass')
p.add_argument('--sampler',default='res_multistep')
p.add_argument('--shift-video',type=float)
p.add_argument('--shift-audio',type=float,default=3.0)
p.add_argument('--paiton-runtime',type=Path)
p.add_argument('--paiton-projections',type=Path)
cfg = p.parse_args()
cfg.output.mkdir(parents=True, exist_ok=True)
(cfg.output/'command.json').write_text(json.dumps({'argv':sys.argv,'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}},indent=2))
os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1')
os.environ.setdefault('MIOPEN_FIND_MODE', 'FAST')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
sys.path.insert(0, str(COMFY_ROOT))
sys.argv = ['main.py', '--base-directory', str(ROOT), '--disable-all-custom-nodes', '--disable-api-nodes', '--use-pytorch-cross-attention' if cfg.attention=='sdpa' else '--use-ck-attention', '--bf16-vae', '--reserve-vram', '2', '--disable-smart-memory', '--cache-none', '--preview-method', 'none']
if cfg.disable_dynamic_vram:
    sys.argv.append('--disable-dynamic-vram')
if cfg.enable_triton_backend:
    sys.argv.append('--enable-triton-backend')
if cfg.fast_disk:
    sys.argv.append('--fast-disk')
if cfg.smart_memory:
    sys.argv.remove('--disable-smart-memory')
startup = time.perf_counter()
import main as comfy_main
import torch
import nodes
import comfy.model_management as mm
import comfy.memory_management as memory
from comfy_extras import nodes_minimax_h3 as h3
from comfy_extras import nodes_custom_sampler as sampler_nodes
from comfy_extras import nodes_video, nodes_audio
from comfy_api.latest import Types
if cfg.paiton_projections:
    if cfg.paiton_runtime is None:
        raise ValueError('--paiton-runtime is required with an artifact')
    sys.path.insert(0,str(cfg.paiton_runtime))
    from paiton_video import projection as paiton_projection
    paiton_projection.install_projections(cfg.paiton_projections)

events_file = (cfg.output/'events.jsonl').open('a',buffering=1)
telemetry_file = (cfg.output/'telemetry.jsonl').open('a',buffering=1)
state = {'phase':'setup', 'run':-1}
stop = threading.Event()

def system_snapshot():
    status = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.split(':')[0] in ('VmRSS','VmHWM','RssAnon','RssFile','VmSwap','VmLck'):
            key,value=line.split(':',1); status[key+'_bytes']=int(value.split()[0])*1024
    for line in Path('/proc/self/io').read_text().splitlines():
        key,value=line.split(':',1)
        if key in ('read_bytes','write_bytes'):status[key]=int(value)
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        if key in ('MemAvailable','SwapFree'):status[key+'_bytes']=int(value.split()[0])*1024
    dev=next((card/'device' for card in Path('/sys/class/drm').glob('card[0-9]*') if (card/'device/mem_info_vram_total').exists() and int((card/'device/mem_info_vram_total').read_text()) >= 30*1024**3), Path('/sys/class/drm/card0/device'))
    for name in ('mem_info_vram_used','gpu_busy_percent','mem_busy_percent','power_dpm_force_performance_level','pp_dpm_sclk','pp_dpm_mclk'):
        try:status[name]=(dev/name).read_text().strip()
        except OSError:pass
    for hw in (dev/'hwmon').glob('hwmon*'):
        for name in ('temp1_input','temp2_input','temp3_input','power1_average','freq1_input','freq2_input'):
            try:status[name]=int((hw/name).read_text())
            except OSError:pass
    return status

def emit(event, **kwargs):
    row={'event':event,'time':time.time(),'monotonic':time.perf_counter(),**state,**kwargs}
    events_file.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def monitor():
    low_memory_samples=0
    while not stop.is_set():
        snapshot=system_snapshot()
        telemetry_file.write(json.dumps({'time':time.time(),**state,**snapshot})+'\n')
        low_memory_samples = low_memory_samples+1 if snapshot['MemAvailable_bytes']<400*1024**2 and snapshot['SwapFree_bytes']<512*1024**2 else 0
        if low_memory_samples>=4:
            # Stop only this qualification process before exhausting the host.
            os.kill(os.getpid(),2)
            return
        stop.wait(0.5)

monitor_thread=threading.Thread(target=monitor,daemon=True)
monitor_thread.start()

def phase(name, fn):
    state['phase']=name
    torch.cuda.synchronize()
    start=time.perf_counter()
    before=system_snapshot()
    torch.cuda.reset_peak_memory_stats()
    emit('phase_start')
    value=fn()
    torch.cuda.synchronize()
    after=system_snapshot()
    emit('phase_end',seconds=time.perf_counter()-start,
         gpu_allocated=torch.cuda.memory_allocated(),gpu_reserved=torch.cuda.memory_reserved(),
         gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved(),
         aimdo_vram=comfy_main.comfy_aimdo.control.get_total_vram_usage() if memory.aimdo_enabled else None,
         read_bytes=after['read_bytes']-before['read_bytes'],host=after)
    return value

try:
    emit('setup_complete',seconds=time.perf_counter()-startup,torch=torch.__version__,hip=torch.version.hip,dynamic_vram=memory.aimdo_enabled)
    if cfg.setup_only:
        sys.exit(0)
    with torch.inference_mode():
        clip=phase('load_encoder',lambda:nodes.CLIPLoader().load_clip('qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors','minimax','default')[0])
        video_vae=None
        if not cfg.encoder_only:
            def load_video_vae():
                sd,metadata=comfy_main.comfy.utils.load_torch_file(str(ROOT/'models/vae'/cfg.video_vae),return_metadata=True)
                return comfy_main.comfy.sd.VAE(sd=sd,metadata=metadata,dtype=torch.bfloat16)
            video_vae=phase('load_video_vae',load_video_vae)
            # Comfy's global --bf16-vae flag overrides even the audio VAE's
            # FP32-only dtype list. Use the stock VAE constructor's explicit
            # dtype argument to retain the required audio precision.
            audio_vae=phase('load_audio_vae',lambda:comfy_main.comfy.sd.VAE(
                sd=comfy_main.comfy.utils.load_torch_file(str(ROOT/'models/vae/minimax_h3_audio_vae_fp32.safetensors')),
                dtype=torch.float32))
            model=phase('load_denoiser',lambda:nodes.UNETLoader().load_unet(cfg.denoiser,'default')[0])
            if cfg.lora:
                from comfy_extras.nodes_lora_debug import LoraLoaderBypassModelOnly
                lora_loader=LoraLoaderBypassModelOnly if cfg.lora_mode=='bypass' else nodes.LoraLoaderModelOnly
                model=phase('load_lora',lambda:lora_loader().load_lora_model_only(model,cfg.lora,1.0)[0])
            if cfg.shift_video is not None:
                model=h3.MiniMaxH3SigmaShift.execute(model,cfg.shift_video,cfg.shift_audio)[0]
            # The pinned official template uses res_multistep + simple, 20 steps.
            sampler=sampler_nodes.KSamplerSelect.execute(cfg.sampler)[0]
            sigmas=sampler_nodes.BasicScheduler.execute(model,'simple',cfg.steps,1.0)[0]
            emit('schedule',video_sigmas=sigmas.tolist(),sampler=cfg.sampler,steps=cfg.steps,
                 model_sampling_class=type(model.get_model_object('model_sampling')).__name__)
            original_forward=model.model.diffusion_model.forward
            evals=[]
            def observe_forward(*args,**kwargs):
                profile=None
                if state['run']==0 and len(evals)==cfg.profile_step:
                    profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True)
                    profile.__enter__()
                start=torch.cuda.Event(enable_timing=True)
                end=torch.cuda.Event(enable_timing=True)
                start.record()
                if cfg.paiton_projections:
                    with paiton_projection.projection_scope():
                        value=original_forward(*args,**kwargs)
                else:
                    value=original_forward(*args,**kwargs)
                end.record()
                evals.append((start,end))
                if profile is not None:
                    profile.__exit__(None,None,None)
                    profile.export_chrome_trace(str(cfg.output/'denoiser-trace.json'))
                    (cfg.output/'denoiser-profile.txt').write_text(profile.key_averages(group_by_input_shape=True).table(sort_by='self_device_time_total',row_limit=70))
                return value
            model.model.diffusion_model.forward=observe_forward
        cases=json.loads(cfg.cases.read_text()) if cfg.cases else None
        for run in range(cfg.runs):
            if cases:
                case=cases[run % len(cases)]
                cfg.prompt=case['prompt'];cfg.seed=case['seed']
                state['case']=case['id']
            state['run']=run
            emit('prompt_start',prompt=cfg.prompt,seed=cfg.seed)
            start=time.perf_counter()
            cond,latent=phase('conditioning',lambda:h3.MiniMaxH3ImageToVideo.execute(clip,video_vae,cfg.prompt,cfg.width,cfg.height,cfg.frames))
            emit('conditioning_shape',shape=list(cond[0][0].shape),finite=bool(torch.isfinite(cond[0][0]).all()))
            if cfg.encoder_only:
                torch.save({'conditioning':cond,'prompt':cfg.prompt},cfg.output/f'conditioning-{run}.pt')
                continue
            evals.clear()
            guider=sampler_nodes.BasicGuider.execute(model,cond)[0]
            noise=sampler_nodes.RandomNoise.execute(cfg.seed)[0]
            samples=phase('sampling',lambda:sampler_nodes.SamplerCustomAdvanced.execute(noise,guider,sampler,sigmas,latent)[0])
            emit('denoiser_evaluations',count=len(evals),gpu_ms=[a.elapsed_time(b) for a,b in evals])
            if cfg.paiton_projections:
                emit('paiton_projection_calls',count=paiton_projection.calls)
                from paiton_video import fusions
                emit('paiton_fusion_calls',counts=dict(fusions.calls))
            if cfg.tiled_vae:
                images=phase('video_decode',lambda:nodes.VAEDecodeTiled().decode(video_vae,samples,512,64,64,8)[0])
            else:
                images=phase('video_decode',lambda:nodes.VAEDecode().decode(video_vae,samples)[0])
            audio=phase('audio_decode',lambda:nodes_audio.VAEDecodeAudio.execute(audio_vae,samples)[0])
            before_encoding=time.perf_counter()-start
            video=nodes_video.CreateVideo.execute(images,24,audio,8)[0]
            path=cfg.output/f'clip-{run}.mp4'
            phase('encode_mux',lambda:video.save_to(str(path),format=Types.VideoContainer('mp4'),codec=Types.VideoCodec('h264'),metadata={'prompt':cfg.prompt,'seed':cfg.seed,'denoiser':cfg.denoiser,'video_vae':cfg.video_vae,'lora':cfg.lora,'lora_mode':cfg.lora_mode,'sampler':cfg.sampler,'steps':cfg.steps,'shift_video':cfg.shift_video or 12,'shift_audio':cfg.shift_audio,'checkpoint_lock':json.loads((Path(__file__).resolve().parents[1]/'checkpoints.lock.json').read_text())}))
            seconds=time.perf_counter()-start
            emit('clip_complete',prompt=cfg.prompt,seed=cfg.seed,seconds=seconds,pipeline_before_encoding_seconds=before_encoding,
                 actual_frames=images.shape[0],width=images.shape[2],height=images.shape[1],fps=24,
                 audio_shape=list(audio['waveform'].shape),audio_sample_rate=audio['sample_rate'],
                 audio_finite=bool(torch.isfinite(audio['waveform']).all()),audio_abs_peak=float(audio['waveform'].abs().max()),
                 image_finite=bool(torch.isfinite(images).all()),path=str(path),
                 clips_per_hour=3600/seconds,video_seconds_per_wall_second=images.shape[0]/24/seconds)
            torch.save({'video':samples['samples'].tensors[0].cpu(),'audio':samples['samples'].tensors[1].cpu()},cfg.output/f'latents-{run}.pt')
            del images,audio,video,samples,cond,latent,guider,noise
            gc.collect()
except BaseException as exc:
    if not isinstance(exc,SystemExit):emit('failure',type=type(exc).__name__,message=str(exc))
    raise
finally:
    stop.set()
    monitor_thread.join(timeout=2)
    telemetry_file.close()
    events_file.close()
