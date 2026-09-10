"""Record hashes only after validating pinned Hub snapshot provenance."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def record(root):
    root=Path(root)
    lock=json.loads((Path(__file__).resolve().parents[1]/'models.lock.json').read_text())
    receipt={}
    for role,name,config in [('asr','parakeet','config.json'),('diarization','pyannote','config.yaml'),('summary','granite-summary','config.json')]:
        expected=lock['models'][role]['revision'];directory=root/name
        metadata=directory/'.cache/huggingface/download'/f'{config}.metadata'
        provenance=metadata.read_text().splitlines()[0] if metadata.is_file() else directory.resolve().name
        if provenance!=expected:raise ValueError(f'{name}: could not establish the pinned download revision; files were preserved.')
        files={}
        for path in sorted(directory.rglob('*')):
            relative=path.relative_to(directory)
            if any(part.startswith('.') for part in relative.parts) or not path.is_file():continue
            files[str(relative)]=dict(bytes=path.stat().st_size,sha256=digest(path))
        receipt[name]=dict(revision=expected,files=files)
    vad=root/'silero/silero_vad.jit'
    if digest(vad)!=lock['models']['vad']['sha256']:raise ValueError('VAD hash mismatch.')
    receipt['silero']=dict(revision=lock['models']['vad']['revision'],files={'silero_vad.jit':dict(bytes=vad.stat().st_size,sha256=digest(vad))})
    path=root/'model-provenance.json'
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(receipt,indent=2)+'\n');temporary.replace(path)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',required=True);a=p.parse_args();record(a.models)
    print('Pinned model provenance and file hashes recorded.')
