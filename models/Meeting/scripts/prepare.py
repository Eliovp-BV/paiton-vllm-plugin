#!/usr/bin/env python3
"""Download pinned public checkpoints into a persistent, content-addressed cache.

Use `hf auth login` separately after accepting the gated model's conditions.
Credentials are discovered by huggingface_hub and are never printed here.
Existing role directories are never overwritten. No audio is accessed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    args=parser.parse_args();root=args.models.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=json.loads((Path(__file__).resolve().parents[1]/'models.lock.json').read_text())
    from huggingface_hub import snapshot_download
    for role,target in [('asr','parakeet'),('diarization','pyannote'),('summary','granite-summary')]:
        model=lock['models'][role];link=root/target
        # Avoid replacing a user's existing checkpoint, including a broken link.
        if link.exists() or link.is_symlink():
            expected=root/'hub'/('models--'+model['repository'].replace('/','--'))/'snapshots'/model['revision']
            if link.resolve()!=expected.resolve():
                raise ValueError(f'{target}: existing checkpoint preserved; choose a new cache directory for pinned preparation.')
            print(f'{target}: reusing the pinned cache snapshot.')
            continue
        snapshot=Path(snapshot_download(model['repository'],revision=model['revision'],
                      cache_dir=root/'hub',max_workers=2,
                      allow_patterns=model.get('download_patterns')))
        link.symlink_to(os.path.relpath(snapshot,root),target_is_directory=True)
        print(f'{target}: pinned snapshot cached.')
    vad=lock['models']['vad'];directory=root/'silero';directory.mkdir(exist_ok=True)
    path=directory/'silero_vad.jit'
    if not path.exists():
        request=urllib.request.Request(vad['url'])
        with urllib.request.urlopen(request,timeout=120) as response:
            data=response.read(16*1024*1024+1)
        if len(data)>16*1024*1024 or hashlib.sha256(data).hexdigest()!=vad['sha256']:
            raise ValueError('VAD download does not match the pinned artifact.')
        # Exclusive creation preserves a file installed concurrently.
        with path.open('xb') as output:output.write(data)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=vad['sha256']:
        raise ValueError('Existing VAD checkpoint differs; it was not overwritten.')
    from provenance import record
    record(root)
    print('Pinned speech, diarization and compact summary checkpoints ready.')

if __name__=='__main__':main()
