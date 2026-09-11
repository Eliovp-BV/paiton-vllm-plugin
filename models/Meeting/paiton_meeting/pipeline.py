"""One-command recording import with isolated stage processes and local output."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def process(recording, models, output, artifact=None, track=0, channel=None,
            keep_intermediates=False, summary_backend='transformers'):
    recording = Path(recording).resolve(strict=True)
    models = Path(models).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    scratch = output / 'work'
    scratch.mkdir(mode=0o700)
    selection = ['--track', str(track)]
    if channel is not None:
        selection += ['--channel', str(channel)]
    timing = {}
    started = time.perf_counter()

    def stage(name, args):
        now = time.perf_counter()
        result = subprocess.run([sys.executable, '-m', 'paiton_meeting', *map(str, args)],
                                check=False, env={**os.environ, 'HF_HUB_OFFLINE': '1',
                                                 'HF_HUB_DISABLE_TELEMETRY': '1',
                                                 'PYANNOTE_METRICS_ENABLED': '0'})
        timing[name + '_seconds'] = time.perf_counter() - now
        if result.returncode:
            raise RuntimeError(f'{name} failed; the original recording was preserved.')

    try:
        asr = ['transcribe', recording, '--model', models/'parakeet', '--vad',
               models/'silero/silero_vad.jit', '--output', scratch/'asr.json', *selection]
        if artifact:
            asr += ['--artifact', Path(artifact).resolve(strict=True)]
        stage('asr_with_startup', asr)
        stage('diarization_with_startup', ['diarize', recording, '--model', models/'pyannote',
              '--output', scratch/'diarization.json', '--scratch', scratch, *selection])
        stage('attribution_with_startup', ['attribute', scratch/'asr.json',
              scratch/'diarization.json', '--output', scratch/'transcript.json'])
        stage('playback_with_startup', ['normalize', recording, '--output',
              output/'playback.wav', *selection])
        stage('summary_with_startup', ['summarize', scratch/'transcript.json',
              '--checkpoint', models/'granite-summary', '--summary-backend', summary_backend, '--output', scratch/'result.json'])
        result = json.loads((scratch/'result.json').read_text())
        timing['complete_pipeline_seconds'] = time.perf_counter() - started
        result['pipeline_timings'] = timing
        result['compiler_enabled'] = bool(artifact)
        from .__main__ import save
        save(output/'result.json', result)
        print(json.dumps(dict(status='complete', timings=timing)), flush=True)
    finally:
        # Only the exclusively created work directory belongs to this operation.
        # The input recording and previously existing output directories are untouched.
        if not keep_intermediates:
            shutil.rmtree(scratch)
    return result
