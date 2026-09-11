"""Check the release C ABI and dependencies without importing Torch."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact',type=Path)
    args=parser.parse_args();path=args.artifact.resolve(strict=True)
    manifest=json.loads(path.with_suffix('.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:
        raise ValueError('Artifact checksum mismatch.')
    if manifest['abi']!='PaitonMeetingLstm640F16V1' or manifest['gpu_arch']!='gfx1201':
        raise ValueError('Unexpected release ABI or GPU target.')
    dependencies=subprocess.run(['ldd',str(path)],check=True,capture_output=True,text=True).stdout
    if any(name in dependencies.lower() for name in ('libtorch','libc10','libaten','not found')):
        raise ValueError('Artifact has missing or framework-specific dependencies.')
    library=ctypes.CDLL(str(path))
    fn=getattr(library,manifest['abi']);fn.argtypes=[ctypes.c_void_p]*11;fn.restype=ctypes.c_int
    if fn(*([None]*11))==0:
        raise ValueError('The ABI accepted null tensor pointers.')
    if any(name=='torch' or name.startswith('torch.') for name in sys.modules):
        raise ValueError('Standalone check imported Torch.')
    print(json.dumps({'status':'pass','sha256':manifest['sha256'],'torch_imported':False}))
    print(dependencies)


if __name__=='__main__':main()
