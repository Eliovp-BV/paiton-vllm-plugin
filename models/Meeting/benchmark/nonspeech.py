"""Selected pipeline checks on newly generated silence, noise and tonal music."""
import argparse,hashlib,json,wave
from pathlib import Path
import numpy as np
from paiton_meeting.asr import ParakeetASR
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--models',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--artifact',type=Path);args=parser.parse_args()
base=args.output;base.mkdir(exist_ok=False)
rate=16000;t=np.arange(rate*30)/rate
silence=np.zeros(len(t),np.float32)
noise=np.random.default_rng(1201).normal(0,.02,len(t)).astype(np.float32)
music=np.zeros(len(t),np.float32)
for bar,chord in enumerate(((220,275,330),(247,311,370),(196,247,294),(220,277,330),(262,330,392),(196,245,294))):
 start=3+bar*4;mask=(t>=start)&(t<start+4);local=t[mask]-start;envelope=np.sin(np.pi*local/4)**2
 music[mask]=.07*envelope*sum(np.sin(2*np.pi*f*local)+.3*np.sin(4*np.pi*f*local) for f in chord)
model=ParakeetASR(args.models/'parakeet',artifact=args.artifact,vad_path=args.models/'silero/silero_vad.jit')
rows=[]
for name,audio in [('silence',silence),('white_noise',noise),('synthetic_tonal_music',music)]:
 path=base/(name+'.wav');payload=np.round(np.clip(audio,-1,1)*32767).astype('<i2').tobytes()
 with wave.open(str(path),'wb') as output:output.setnchannels(1);output.setsampwidth(2);output.setframerate(rate);output.writeframes(payload)
 result=model.transcribe(path)
 rows.append(dict(name=name,license='CC0-1.0',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),duration=30,segments=result['segments'],word_count=len(result['words']),seconds=result['asr_seconds'],chunks=result['chunk_timings']))
output=dict(note='Newly generated non-human test signals. Tonal music is not representative of all music or singing. Tests the actual selected VAD+ASR path, not raw ungated ASR.',results=rows)
(base/'nonspeech-qualification.json').write_text(json.dumps(output,indent=2));print(json.dumps(output),flush=True)
