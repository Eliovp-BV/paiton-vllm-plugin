import wave
import numpy as np
import pytest
from paiton_meeting.audio import chunks, inspect_audio, pcm_frames


def recording(tmp_path,seconds=65,channels=2,rate=48000):
    path=tmp_path/'input.wav'
    t=np.arange(round(seconds*rate))/rate
    samples=np.stack([np.sin(2*np.pi*(300+i*400)*t)*12000 for i in range(channels)],axis=1).astype('<i2')
    with wave.open(str(path),'wb') as f:
        f.setnchannels(channels);f.setsampwidth(2);f.setframerate(rate);f.writeframes(samples.tobytes())
    return path


def test_resampling_and_chunk_coverage(tmp_path):
    path=recording(tmp_path)
    assert inspect_audio(path)['tracks'][0]['channels']==2
    result=list(chunks(path))
    assert result[0].core_start==0
    assert result[-1].core_end==pytest.approx(65,abs=1/16000)
    for a,b in zip(result,result[1:]):
        assert a.core_end==b.core_start
        assert a.end>b.start
    assert max(len(c.samples) for c in result)<=30*16000


def test_channel_preservation_and_invalid_selection(tmp_path):
    path=recording(tmp_path,seconds=1)
    left=np.concatenate(list(pcm_frames(path,channel=0)))
    right=np.concatenate(list(pcm_frames(path,channel=1)))
    assert len(left)==len(right)==16000
    assert not np.allclose(left,right)
    with pytest.raises(ValueError):list(pcm_frames(path,channel=2))
    with pytest.raises(ValueError):list(pcm_frames(path,track=-1))


def test_no_audio_and_empty_tail(tmp_path):
    path=recording(tmp_path,seconds=30,channels=1,rate=16000)
    result=list(chunks(path))
    assert result[-1].core_end==30
    assert result[0].core_end==result[-1].core_start
    with pytest.raises(ValueError):list(chunks(path,seconds=2,overlap=2))

def test_normalized_playback_and_original_preservation(tmp_path):
    from paiton_meeting.audio import normalize
    import hashlib
    source=recording(tmp_path,seconds=2)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    output=tmp_path/'playback.wav'
    info=normalize(source,output,channel=1)
    assert info['duration']==2 and info['time_origin']=='decoded-audio-start'
    assert inspect_audio(output)['tracks'][0]['channels']==1
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    with pytest.raises(FileExistsError):normalize(source,output)
    assert output.exists()
