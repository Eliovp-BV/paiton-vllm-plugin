from types import SimpleNamespace
import pytest
from paiton_meeting.pipeline import process


def test_existing_output_is_never_reused(tmp_path, monkeypatch):
    recording=tmp_path/'meeting.wav';recording.write_bytes(b'user recording')
    models=tmp_path/'models';models.mkdir()
    output=tmp_path/'existing';output.mkdir();(output/'result.json').write_text('user output')
    monkeypatch.setattr('subprocess.run',lambda *a,**k:pytest.fail('inference should not start'))
    with pytest.raises(FileExistsError):process(recording,models,output)
    assert (output/'result.json').read_text()=='user output'
    assert recording.read_bytes()==b'user recording'


def test_failed_stage_removes_only_owned_scratch(tmp_path, monkeypatch):
    recording=tmp_path/'meeting.wav';recording.write_bytes(b'user recording')
    models=tmp_path/'models';models.mkdir()
    output=tmp_path/'new'
    monkeypatch.setattr('subprocess.run',lambda *a,**k:SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError,match='original recording was preserved'):
        process(recording,models,output)
    assert recording.read_bytes()==b'user recording'
    assert not (output/'work').exists()
    assert not (output/'result.json').exists()
