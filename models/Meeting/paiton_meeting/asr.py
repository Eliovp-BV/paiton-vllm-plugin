"""Qualified settings are locked separately; no implicit network downloads."""
import time
from pathlib import Path
from .audio import chunks
from .transcript import token_words,stitch_words,segments_from_words

class ParakeetASR:
    def __init__(self,directory,dtype='float16',artifact=None,vad_path=None):
        import torch
        from transformers import AutoProcessor,ParakeetForTDT
        self.torch=torch
        self.dtype=getattr(torch,dtype)
        started=time.perf_counter()
        self.processor=AutoProcessor.from_pretrained(directory,local_files_only=True)
        # Legacy checkpoints omit this setting. Resolve the unchanged tokenizer
        # once instead of enumerating its full vocabulary for every output token.
        self.processor.decoder_type=self.processor._decoder_type
        self.model=ParakeetForTDT.from_pretrained(directory,dtype=self.dtype,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
        if artifact:
            from .lstm import PaitonLSTM
            self.model.decoder.lstm=PaitonLSTM(self.model.decoder.lstm,artifact).eval()
        from .vad import SileroVAD
        self.vad=SileroVAD(vad_path) if vad_path else None
        torch.cuda.synchronize();self.loading_seconds=time.perf_counter()-started

    def transcribe(self,path,track=0,channel=None,progress=None,turns=None):
        words=[];timings=[];started=time.perf_counter()
        for chunk in chunks(path,track=track,channel=channel):
            if self.vad and not self.vad.has_speech(chunk.samples):
                timings.append(dict(start=chunk.start,end=chunk.end,seconds=0,vad_skipped=True))
                if progress:progress(dict(processed_seconds=chunk.core_end,words=[]))
                continue
            inputs=self.processor(audio=chunk.samples,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda',self.dtype)
            stage=time.perf_counter()
            with self.torch.inference_mode():
                output=self.model.generate(**inputs,return_dict_in_generate=True,max_new_tokens=768)
            self.torch.cuda.synchronize()
            if output.sequences.shape[-1]>=768:
                raise ValueError('A speech chunk reached its decoding limit; no truncated transcript was accepted.')
            _,timestamps=self.processor.decode(output.sequences.cpu(),durations=output.durations.cpu(),skip_special_tokens=True)
            accepted=stitch_words(token_words(timestamps[0]),chunk)
            words.extend(accepted)
            timings.append(dict(start=chunk.start,end=chunk.end,seconds=time.perf_counter()-stage))
            if progress:progress(dict(processed_seconds=chunk.core_end,words=accepted))
        return dict(segments=segments_from_words(words,turns),words=words,
                    asr_seconds=time.perf_counter()-started,chunk_timings=timings,
                    loading_seconds=self.loading_seconds,
                    speaker_attribution='diarization' if turns is not None else 'unavailable')
