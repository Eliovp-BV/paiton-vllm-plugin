"""Reproducible ASR-stage timing; not a complete-pipeline speedup claim."""
import json
import os
from pathlib import Path
import resource
import statistics
import threading
import time


def benchmark(recording, model, vad, output, artifact=None, repeats=3):
    if repeats < 3 or repeats > 30:
        raise ValueError('Use 3–30 warm repetitions.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    import torch
    from .asr import ParakeetASR
    from .audio import pcm_frames, RATE
    from .__main__ import save
    # Decode once to count real samples, including formats without duration metadata.
    duration = sum(len(frame) for frame in pcm_frames(recording))/RATE
    if duration <= 0:
        raise ValueError('Benchmark requires a nonempty recording.')
    stopping = threading.Event()
    def sample():
        paths=list(Path('/sys/class/drm').glob('card*/device/mem_info_vram_used'))
        descriptor=os.open(output/'telemetry.jsonl',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(descriptor,'w') as stream:
            while not stopping.wait(.25):
                row=dict(time=time.perf_counter(), allocated=torch.cuda.memory_allocated(),
                         reserved=torch.cuda.memory_reserved(),
                         driver_vram={p.parent.parent.name:int(p.read_text()) for p in paths})
                stream.write(json.dumps(row)+'\n');stream.flush()
    thread=threading.Thread(target=sample,daemon=True);thread.start();rows=[]
    try:
        asr=ParakeetASR(model,artifact=artifact,vad_path=vad)
        for repeat in range(repeats+1):
            torch.cuda.reset_peak_memory_stats()
            result=asr.transcribe(recording)
            save(output/f'transcript-{repeat}.json',result)
            seconds=result['asr_seconds']
            row=dict(repeat=repeat,warm=repeat>0,processing_seconds=seconds,
                     loading_seconds=asr.loading_seconds,audio_seconds=duration,
                     rtf=seconds/duration,audio_hours_per_wall_hour=duration/seconds,
                     peak_allocated=torch.cuda.max_memory_allocated(),
                     peak_reserved=torch.cuda.max_memory_reserved(),
                     peak_host_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            rows.append(row);print(json.dumps(row),flush=True)
    finally:
        stopping.set();thread.join();save(output/'timings.json',rows)
    warm=[r['processing_seconds'] for r in rows if r['warm']]
    report=dict(stage='asr-only',compiler=bool(artifact),sample_count=len(warm),
                median_seconds=statistics.median(warm),stddev_seconds=statistics.stdev(warm),
                minimum_seconds=min(warm),maximum_seconds=max(warm),
                rtf_definition='processing seconds / audio seconds',
                note='First processing pass and model loading are separate. This command does not measure diarization, summary or model switching.')
    save(output/'report.json',report)
    return report
