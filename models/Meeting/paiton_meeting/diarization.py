"""Offline anonymous diarization, preserving overlap and separate turns.

The checkpoint must already exist locally. Access approval and download are a
separate installation step. PCM is spooled to a private temporary file rather
than concatenated into host RAM; the original recording is never changed.
"""
import os
import tempfile
import time
from pathlib import Path

from .audio import RATE, pcm_frames


def diarize(path, directory, *, track=0, channel=None, scratch=None, progress=None):
    import numpy as np
    import torch
    from pyannote.audio import Pipeline

    directory = Path(directory)
    if not (directory / 'config.yaml').is_file():
        raise ValueError('Install the authorized local diarization checkpoint first.')
    # Must be set before Pipeline construction; inference never needs telemetry.
    os.environ['PYANNOTE_METRICS_ENABLED'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='meeting-pcm-', dir=scratch) as temporary:
        pcm_path = Path(temporary) / 'audio.f32'
        count = 0
        with pcm_path.open('wb') as output:
            for values in pcm_frames(path, track, channel):
                output.write(values.astype('<f4', copy=False).tobytes())
                count += len(values)
        decode_seconds = time.perf_counter() - started
        if not count:
            return dict(turns=[], exclusive_turns=[], seconds=0, loading_seconds=0,
                        decode_seconds=decode_seconds)
        waveform = np.memmap(pcm_path, dtype='<f4', mode='c', shape=(1, count))
        pipeline = None
        tensor = None
        try:
            loaded = time.perf_counter()
            pipeline = Pipeline.from_pretrained(directory).to(torch.device('cuda'))
            torch.cuda.synchronize()
            loading_seconds = time.perf_counter() - loaded
            processing = time.perf_counter()
            def hook(step_name, step_artifact, file=None, total=None, completed=None):
                if progress and (completed is None or completed == total):
                    progress(dict(stage=step_name, completed=completed, total=total))
            tensor = torch.from_numpy(waveform)
            result = pipeline({'waveform': tensor, 'sample_rate': RATE}, hook=hook)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - processing
            def turns(annotation):
                return [dict(start=segment.start, end=segment.end, speaker=speaker)
                        for segment, _, speaker in annotation.itertracks(yield_label=True)]
            return dict(turns=turns(result.speaker_diarization),
                        exclusive_turns=turns(result.exclusive_speaker_diarization),
                        seconds=seconds, loading_seconds=loading_seconds,
                        decode_seconds=decode_seconds)
        finally:
            del tensor, pipeline
            waveform._mmap.close()
            torch.cuda.empty_cache()
