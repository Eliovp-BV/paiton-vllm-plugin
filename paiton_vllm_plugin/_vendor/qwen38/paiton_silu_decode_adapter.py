"""Experimental boundary for existing vLLM; native compiler/artifact is HIP-only."""
import ctypes as C
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import threading
import time
from paiton_silu_decode_policy import Policy, FLAG

TARGET='radiance_mxfp4_fp8'
ROOT=Path('/opt/paiton/runtime/silu-decode')
LIB=AUDIT=POLICY=None
CALLS=FALLBACKS=0
SHADOW=os.environ.get('PAITON_TARGET_SILU_DECODE_SHADOW','0')=='1'

def check(rc):
    if rc:raise RuntimeError(f'Native SiLU HIP status {rc}')

def load():
    global LIB,AUDIT,POLICY
    import torch  # Tensor ownership/device identity in existing external vLLM only.
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('Initialize native SiLU before capture')
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx1201':
        raise RuntimeError('Native SiLU requires one gfx1201 device')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    path=(ROOT/manifest['file']).resolve(strict=True)
    if path.parent!=ROOT or path.suffix!='.so':raise ValueError('Expected sibling native SiLU artifact')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('Native SiLU artifact hash mismatch')
    if hashlib.sha256(Path(__import__('paiton_silu_decode_policy').__file__).read_bytes()).hexdigest()!=manifest['policy_sha256']:
        raise ValueError('Native SiLU policy hash mismatch')
    POLICY=Policy.from_environment(STARTUP_FLAGS,manifest)
    LIB=C.CDLL(str(path),mode=C.RTLD_LOCAL)
    LIB.paiton_silu_decode_arch.restype=C.c_char_p
    if LIB.paiton_silu_decode_abi_version()!=1 or LIB.paiton_silu_decode_arch()!=b'gfx1201':raise ValueError('Native SiLU ABI mismatch')
    LIB.paiton_silu_decode.argtypes=[C.c_void_p]*3+[C.c_int]*3+[C.c_void_p]
    LIB.paiton_silu_decode.restype=C.c_int
    if SHADOW:
        audit_manifest=json.loads((ROOT/'audit-manifest.json').read_text())
        audit_path=ROOT/'audit.so'
        if hashlib.sha256(audit_path.read_bytes()).hexdigest()!=audit_manifest['sha256']:raise ValueError('Audit artifact mismatch')
        AUDIT=C.CDLL(str(audit_path),mode=C.RTLD_LOCAL)
        AUDIT.paiton_silu_decode_audit_compare.argtypes=[C.c_void_p]*4+[C.c_int]*2+[C.c_void_p]
        AUDIT.paiton_silu_decode_audit_report.argtypes=[C.POINTER(C.c_ulonglong)]
        check(AUDIT.paiton_silu_decode_audit_init())
        directory=Path('/evidence')
        def report():
            deadline=time.monotonic()+1200
            for stage in ('warmup','final'):
                while time.monotonic()<deadline and not (directory/(stage+'-audit-request')).exists():time.sleep(.2)
                if not (directory/(stage+'-audit-request')).exists():return
                counts=(C.c_ulonglong*6)();rc=AUDIT.paiton_silu_decode_audit_report(counts)
                value={'hip_status':rc,'completed_comparisons':counts[0],'fp8_different_bytes':counts[1],
                       'scale_different_values':counts[2],'nonfinite_scales':counts[3],
                       'm8_comparisons':counts[4],'m64_comparisons':counts[5],
                       'python_launches':CALLS,'counter_semantics':'GPU comparisons include graph replays; host calls count capture/dispatch only','fallbacks':FALLBACKS,'policy':POLICY.snapshot()}
                path=directory/(stage+'-audit.json');tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
        threading.Thread(target=report,daemon=True).start()
    print('[paiton.silu_decode] '+json.dumps({'policy':POLICY.snapshot(),'shadow':SHADOW}),file=sys.stderr,flush=True)

def wrap(module):
    if getattr(module,'_paiton_silu_decode_wrapped',False):return
    original=module.launch_silu_mul_quant
    def launch(gu,q,scale,M,N,stream,tiled=0):
        global CALLS,FALLBACKS
        if M not in (8,64) or N!=17408 or tiled!=0:
            FALLBACKS+=1
            return original(gu,q,scale,M,N,stream,tiled)
        if LIB is None:load()
        if not POLICY.allows(M,N,arch='gfx1201',tiled=tiled):
            FALLBACKS+=1
            return original(gu,q,scale,M,N,stream,tiled)
        output,scale_output=q,scale
        if SHADOW:
            import torch
            if torch.cuda.current_stream().cuda_stream!=stream:raise RuntimeError('Shadow allocation stream mismatch')
            temporary=torch.empty((M,N),device='cuda',dtype=torch.uint8)
            temporary_scale=torch.empty(M,device='cuda',dtype=torch.float32)
            output,scale_output=temporary.data_ptr(),temporary_scale.data_ptr()
            original(gu,q,scale,M,N,stream,tiled)
        check(LIB.paiton_silu_decode(gu,output,scale_output,M,N,0,stream))
        CALLS+=1
        if SHADOW:check(AUDIT.paiton_silu_decode_audit_compare(q,output,scale,scale_output,M,N,stream))
        if CALLS in (1,64,256,512,4096):print('[paiton.silu_decode] native launches='+str(CALLS),file=sys.stderr,flush=True)
        return None
    module.launch_silu_mul_quant=launch
    module._paiton_silu_decode_wrapped=True

class Loader:
    def __init__(self,original):self.original=original
    def __getattr__(self,name):return getattr(self.original,name)
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):self.original.exec_module(module);wrap(module)

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!=TARGET:return None
        after=False
        for finder in list(sys.meta_path):
            if finder is self:after=True;continue
            if not after:continue
            spec=finder.find_spec(fullname,path,target)
            if spec is not None:
                if spec.loader:spec.loader=Loader(spec.loader)
                return spec
        return None

requested=os.environ.get(FLAG,'0')
STARTUP_FLAGS={FLAG:requested}
if requested not in ('0','1','off','on','false','true'):raise ValueError('Invalid decode enable flag')
if requested in ('1','on','true'):
    if TARGET in sys.modules:wrap(sys.modules[TARGET])
    else:sys.meta_path.insert(0,Finder())
