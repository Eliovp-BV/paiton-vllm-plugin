"""Bounded-memory decoding; original recordings remain untouched.

Tracks/channels are explicit. Stereo downmix is suitable for ordinary recordings;
separate participant/microphone feeds must use channel selection, not this default.
Only local files are accepted; FFmpeg network protocols are never used.
"""
from dataclasses import dataclass
from pathlib import Path
import av
import numpy as np

RATE = 16000
MAX_SECONDS = 8 * 3600

@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray
    start: float
    end: float
    core_start: float
    core_end: float


def inspect_audio(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Recording must be a local file.')
    with path.open('rb') as source, av.open(source) as container:
        tracks = [dict(index=s.index, codec=s.codec_context.name,
                       rate=s.codec_context.sample_rate,
                       channels=s.codec_context.channels,
                       duration=float(s.duration*s.time_base) if s.duration is not None else None)
                  for s in container.streams.audio]
        if not tracks:
            raise ValueError('The recording has no audio track.')
        return dict(tracks=tracks, duration=container.duration/av.time_base if container.duration else None)


def pcm_frames(path, track=0, channel=None):
    """Yield mono 16 kHz PCM with bounded decoder/resampler buffers.

Timestamp origin is decoded audio start. Container clock discontinuities >250 ms
fail explicitly instead of silently shifting transcript references.
"""
    with Path(path).open('rb') as source, av.open(source) as container:
        if track < 0 or track >= len(container.streams.audio):
            raise ValueError('Audio track does not exist.')
        stream = container.streams.audio[track]
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=RATE)
        elapsed = 0
        expected = None
        for frame in container.decode(stream):
            if frame.time is not None:
                if expected is not None and abs(frame.time-expected) > .25:
                    raise ValueError('Audio clock discontinuity; repair or split the recording before import.')
                expected = frame.time + frame.samples/frame.sample_rate
            if channel is not None:
                channels = len(frame.layout.channels)
                if channel < 0 or channel >= channels:
                    raise ValueError('Audio channel does not exist.')
                planar = av.AudioResampler(format='fltp',layout=frame.layout,rate=frame.sample_rate)
                converted = planar.resample(frame)[0]
                frame = av.AudioFrame.from_ndarray(converted.to_ndarray()[channel:channel+1].copy(),format='fltp',layout='mono')
                frame.sample_rate = converted.sample_rate
            for output in resampler.resample(frame):
                values = output.to_ndarray()[0].astype(np.float32, copy=False)
                elapsed += len(values)
                if elapsed > MAX_SECONDS*RATE:
                    raise ValueError('Recording exceeds the eight-hour import limit.')
                yield values
        for output in resampler.resample(None):
            yield output.to_ndarray()[0].astype(np.float32,copy=False)


def chunks(path, seconds=30, overlap=2, track=0, channel=None):
    if seconds <= 2*overlap or overlap < 0:
        raise ValueError('Chunk length must exceed twice the overlap.')
    width, margin = round(seconds*RATE), round(overlap*RATE)
    hop = width-2*margin
    buffer = np.empty(0,dtype=np.float32)
    offset = 0
    first = True
    for frame in pcm_frames(path,track,channel):
        buffer = np.concatenate((buffer,frame))
        while len(buffer) >= width:
            yield AudioChunk(buffer[:width].copy(),offset/RATE,(offset+width)/RATE,
                             offset/RATE if first else (offset+margin)/RATE,
                             (offset+width-margin)/RATE)
            first = False
            buffer = buffer[hop:]
            offset += hop
    # A retained overlap alone must not create an extra duplicated transcript.
    if len(buffer) > (margin if not first else 0):
        yield AudioChunk(buffer.copy(),offset/RATE,(offset+len(buffer))/RATE,
                         offset/RATE if first else (offset+margin)/RATE,
                         (offset+len(buffer))/RATE)


def normalize(path, output, track=0, channel=None):
    """Create a playback WAV on the exact decoded-audio transcript clock.

Original container tracks remain intact. No microphone/system tracks are combined
implicitly. The selected audio timeline starts at zero, even for video containers
whose first audio timestamp is nonzero.
"""
    import os
    import wave
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    count = 0
    try:
        with os.fdopen(descriptor, 'wb') as target, wave.open(target, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(RATE)
            for values in pcm_frames(path, track, channel):
                # Normalized playback only; inference can retain original FP32
                # decoding when evaluating lossy sources and quantization.
                pcm = np.clip(np.rint(values * 32768), -32768, 32767).astype('<i2')
                wav.writeframesraw(pcm.tobytes()); count += len(pcm)
        return dict(duration=count/RATE, rate=RATE, channels=1, time_origin='decoded-audio-start')
    except BaseException:
        Path(output).unlink(missing_ok=True)
        raise
