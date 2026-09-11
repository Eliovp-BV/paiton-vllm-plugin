import json
import os
from pathlib import Path
import subprocess
import pytest


@pytest.mark.parametrize('option,compiled',[(None,True),('--stock',False),('--paiton',True)])
def test_launcher_selects_explicit_backend_and_preserves_input(tmp_path,option,compiled):
    binary=tmp_path/'bin';binary.mkdir()
    scripts={
        'docker':'#!/usr/bin/env python3\nimport json,os,sys\nopen(os.environ["MEETING_TEST_CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n',
        'flock':'#!/bin/sh\nshift\nexec "$@"\n',
        'stat':'#!/bin/sh\necho 993\n'}
    for name,source in scripts.items():
        path=binary/name;path.write_text(source);path.chmod(0o755)
    recording=tmp_path/'meeting with spaces.wav';recording.write_bytes(b'preserved fixture')
    cache=tmp_path/'cache';(cache/'models').mkdir(parents=True)
    capture=tmp_path/'arguments.json'
    env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],
             PAITON_MEETING_CACHE=str(cache),PAITON_MEETING_IMAGE='local-test-image',
             MEETING_TEST_CAPTURE=str(capture))
    launcher=Path(__file__).resolve().parents[1]/'run-docker.sh'
    command=[str(launcher)]+([option] if option else [])+[str(recording),str(tmp_path/'new-output'),'--summary-backend','vllm']
    subprocess.run(command,env=env,check=True)
    args=json.loads(capture.read_text())
    assert ('--artifact' in args) is compiled
    assert f'{recording}:/recording/input:ro' in args
    assert args[-2:]==['--summary-backend','vllm']
    assert recording.read_bytes()==b'preserved fixture'
