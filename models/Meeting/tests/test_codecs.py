"""Real codec round trips through the bundled FFmpeg libraries."""
import av
import numpy as np
import pytest
from paiton_meeting.audio import pcm_frames, inspect_audio

@pytest.mark.parametrize('suffix,codec', [('wav','pcm_s16le'),('flac','flac'),('mp3','libmp3lame'),('m4a','aac'),('mp4','aac'),('ogg','libopus'),('webm','libopus')])
def test_actual_recording_formats(tmp_path,suffix,codec):
    path=tmp_path/f'recording.{suffix}'
    rate=48000
    with av.open(str(path),'w') as output:
        stream=output.add_stream(codec,rate=rate);stream.layout='stereo'
        for offset in range(0,rate*2,960):
            t=(np.arange(960)+offset)/rate
            values=np.stack([np.sin(t*2*np.pi*f)*.2 for f in (300,700)]).astype('float32')
            frame=av.AudioFrame.from_ndarray(values,format='fltp',layout='stereo');frame.sample_rate=rate
            for packet in stream.encode(frame):output.mux(packet)
        for packet in stream.encode():output.mux(packet)
    assert inspect_audio(path)['tracks'][0]['channels']==2
    pcm=np.concatenate(list(pcm_frames(path)))
    assert abs(len(pcm)/16000-2)<.1
    assert np.sqrt(np.mean(pcm**2))>.03
    left=np.concatenate(list(pcm_frames(path,channel=0)))
    right=np.concatenate(list(pcm_frames(path,channel=1)))
    assert not np.allclose(left,right,atol=.01)
