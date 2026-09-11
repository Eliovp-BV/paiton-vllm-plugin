"""Compare legacy decoder-mode inference against resolved mode and CPU transfers.

Requires a long recording providing chunks 3, 12 and 70 (30 s, 2 s overlap).
Fixed generated tensors are replayed; this is not full-pipeline acceleration.
"""
import argparse,json,time,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--recording',required=True);p.add_argument('--artifact');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('Output exists')
 from paiton_meeting.asr import ParakeetASR
 from paiton_meeting.audio import chunks
 asr=ParakeetASR(a.model,artifact=a.artifact)
 from transformers import AutoProcessor
 # Recreate the checkpoint's original legacy processor, even when the pipeline
 # already resolves decoder mode once. Both modes consume identical outputs.
 asr.processor=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 torch=asr.torch;rows=[]
 for index,chunk in enumerate(chunks(a.recording)):
  if index not in [3,12,70]:continue
  inputs=asr.processor(audio=chunk.samples,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda',asr.dtype)
  with torch.inference_mode():result=asr.model.generate(**inputs,return_dict_in_generate=True,max_new_tokens=768)
  torch.cuda.synchronize();baseline=None;original_mode=asr.processor.decoder_type;inferred_mode=asr.processor._decoder_type
  for repeat in range(4):
   for mode in (['gpu-scalars','fixed-mode-cpu'] if repeat%2==0 else ['fixed-mode-cpu','gpu-scalars']):
    torch.cuda.synchronize();started=time.perf_counter()
    ids,durations=result.sequences,result.durations
    asr.processor.decoder_type=inferred_mode if mode=='fixed-mode-cpu' else original_mode
    if mode=='fixed-mode-cpu':ids,durations=ids.cpu(),durations.cpu()
    decoded=asr.processor.decode(ids,durations=durations,skip_special_tokens=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    if baseline is None:baseline=decoded
    assert decoded==baseline,'Timestamp/text behavior changed'
    rows.append(dict(chunk=index,start=chunk.start,mode=mode,repeat=repeat,cached=repeat>0,seconds=elapsed,sequence_shape=list(ids.shape),durations_dtype=str(durations.dtype),decoded_sha256=hashlib.sha256(json.dumps(decoded,sort_keys=True).encode()).hexdigest()))
  asr.processor.decoder_type=original_mode
  if index==70:break
 if len(rows)!=24:p.error('Recording did not provide all requested chunks')
 out=dict(original_decoder_type=original_mode,resolved_decoder_type=inferred_mode,compiler_enabled=bool(a.artifact),exact_decoded_equality=True,rows=rows,note='Same generated GPU sequences/durations replayed through existing processor; fixed-mode CPU mode preserves the inferred decoder type and includes both tensor copies. Isolated postprocessing only, not complete ASR or pipeline performance.')
 with a.output.open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps({'rows':len(rows),'exact_decoded_equality':True}),flush=True)
if __name__=='__main__':main()
