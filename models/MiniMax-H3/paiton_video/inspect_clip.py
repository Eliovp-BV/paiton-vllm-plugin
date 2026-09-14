"""Retain decoded-file metadata, contact sheets and simple corruption checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import av
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

p=argparse.ArgumentParser()
p.add_argument('clip',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
if shutil.which('ffprobe'):
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(a.clip)]))
else:
    with av.open(str(a.clip)) as c:
        meta={'tool':'PyAV','version':av.__version__,'duration_seconds':c.duration/av.time_base,'format':c.format.name,'streams':[]}
        for s in c.streams:
            item={'type':s.type,'codec':s.codec_context.name,'frames':s.frames,'time_base':str(s.time_base),'duration_seconds':float(s.duration*s.time_base) if s.duration is not None else None}
            if s.type=='video':item.update(width=s.width,height=s.height,average_rate=str(s.average_rate),pixel_format=s.codec_context.format.name)
            elif s.type=='audio':item.update(sample_rate=s.codec_context.sample_rate,channels=s.codec_context.channels)
            meta['streams'].append(item)
(a.output/'media-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
frame_stats=[];seen={};duplicate=[];previous=None;thumbs=[];times=[]
with av.open(str(a.clip)) as container:
    for index,frame in enumerate(container.decode(video=0)):
        arr=frame.to_ndarray(format='rgb24');digest=hashlib.sha256(arr).hexdigest()
        if digest in seen:duplicate.append([seen[digest],index])
        seen[digest]=index
        small=np.asarray(Image.fromarray(arr).resize((216,120)))
        stat={'index':index,'time':float(frame.time),'mean':float(arr.mean()),'std':float(arr.std()),'sha256':digest}
        if previous is not None:stat['adjacent_mean_abs_difference']=float(np.abs(small.astype(np.float32)-previous).mean())
        previous=small.astype(np.float32)
        frame_stats.append(stat)
        if index%5==0:
            thumbs.append(Image.fromarray(small));times.append(frame.time)
        if index in (0,24,48,72,96,123):Image.fromarray(arr).save(a.output/f'frame-{index:03}.png')
rows=(len(thumbs)+4)//5
sheet=Image.new('RGB',(5*224,rows*146),(23,25,29));draw=ImageDraw.Draw(sheet)
for index,(thumb,stamp) in enumerate(zip(thumbs,times)):
    x=(index%5)*224+4;y=(index//5)*146+4
    sheet.paste(thumb,(x,y));draw.text((x,y+123),f'{stamp:.2f} s',fill='white')
sheet.save(a.output/'contact-sheet.jpg',quality=92)
wave=[];rates=[]
with av.open(str(a.clip)) as container:
    if container.streams.audio:
        for frame in container.decode(audio=0):wave.append(frame.to_ndarray());rates.append(frame.sample_rate)
audio=np.concatenate(wave,axis=-1) if wave else np.zeros((0,0),dtype=np.float32)
metrics={'video_frames':len(frame_stats),'duplicate_frame_pairs':duplicate,'audio_shape':list(audio.shape),'audio_sample_rates':sorted(set(rates)),
         'audio_finite':bool(np.isfinite(audio).all()),'audio_peak':float(np.abs(audio).max()) if audio.size else None,
         'audio_rms':float(np.sqrt(np.mean(audio.astype(np.float64)**2))) if audio.size else None,
         'audio_fraction_at_or_above_one':float((np.abs(audio)>=1).mean()) if audio.size else None,
         'frames':frame_stats}
(a.output/'quality-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
if wave:subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-y','-i',str(a.clip),'-vn','-c:a','pcm_s16le',str(a.output/'audio.wav')],check=True)
print(json.dumps({k:v for k,v in metrics.items() if k!='frames'},indent=2))
