"""Verify that a container includes every existing qualified runtime artifact."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'artifacts'
COMPONENTS=('', 'bf16-regions', 'attention', 'attention-w64', 'attention-qk8',
            'attention-full8', 'normfuse', 'gemm-fp8', 'gemm-int8', 'gemm-w64',
            'quantize', 'unpack-fp8')


def main():
    for component in COMPONENTS:
        root=ROOT/component
        row=json.loads((root/'manifest.json').read_text())
        if row['architecture']!='gfx1201' or Path(row['file']).name!=row['file']:
            raise ValueError(f'Invalid artifact manifest: {component}')
        with (root/row['file']).open('rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=row['sha256']:
            raise ValueError(f'Artifact SHA-256 mismatch: {component}')
    print(f'Verified {len(COMPONENTS)} native runtime artifacts; no GPU initialization')


if __name__=='__main__':
    main()
