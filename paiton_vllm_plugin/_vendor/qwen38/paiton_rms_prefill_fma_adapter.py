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
from paiton_rms_prefill_fma_policy import Policy, FLAG

TARGET='radiance_mxfp4_fp8'
ROOT=Path('/opt/paiton/runtime/rms-prefill')
LIB=AUDIT=POLICY=None
CALLS=FALLBACKS=0
SHADOW=os.environ.get('PAITON_TARGET_RMS_PREFILL_SHADOW','0')=='1'

def check(rc):
    if rc:raise RuntimeError(f'Native RMS HIP status {rc}')

def load():
    global LIB,AUDIT,POLICY
    import torch  # Tensor ownership/device identity in existing external vLLM only.
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('Initialize native RMS before capture')
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx1201':
        raise RuntimeError('Native RMS requires one gfx1201 device')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    path=(ROOT/manifest['file']).resolve(strict=True)
    if path.parent!=ROOT or path.suffix!='.so':raise ValueError('Expected sibling native RMS artifact')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('Native RMS artifact hash mismatch')
    if hashlib.sha256(Path(__import__('paiton_rms_prefill_fma_policy').__file__).read_bytes()).hexdigest()!=manifest['policy_sha256']:
        raise ValueError('Native RMS policy hash mismatch')
    POLICY=Policy.from_environment(STARTUP_FLAGS,manifest)
    LIB=C.CDLL(str(path),mode=C.RTLD_LOCAL)
    LIB.paiton_rms_prefill_arch.restype=C.c_char_p
    if LIB.paiton_rms_prefill_abi_version()!=1 or LIB.paiton_rms_prefill_arch()!=b'gfx1201':raise ValueError('Native RMS ABI mismatch')
    LIB.paiton_rms_prefill.argtypes=[C.c_void_p]*6+[C.c_int]*2+[C.c_float,C.c_void_p]
    LIB.paiton_rms_prefill.restype=C.c_int
    if SHADOW:
        audit_manifest=json.loads((ROOT/'audit-manifest.json').read_text())
        audit_path=ROOT/'audit.so'
        if hashlib.sha256(audit_path.read_bytes()).hexdigest()!=audit_manifest['sha256']:raise ValueError('Audit artifact mismatch')
        AUDIT=C.CDLL(str(audit_path),mode=C.RTLD_LOCAL)
        AUDIT.paiton_rms_audit_compare.argtypes=[C.c_void_p]*6+[C.c_int]*2+[C.c_void_p]
        AUDIT.paiton_rms_audit_report.argtypes=[C.POINTER(C.c_ulonglong)]
        check(AUDIT.paiton_rms_audit_init())
        directory=Path('/evidence')
        def report():
            deadline=time.monotonic()+1200
            for stage in ('warmup','final'):
                while time.monotonic()<deadline and not (directory/(stage+'-audit-request')).exists():time.sleep(.2)
                if not (directory/(stage+'-audit-request')).exists():return
                counts=(C.c_ulonglong*6)();rc=AUDIT.paiton_rms_audit_report(counts)
                value={'hip_status':rc,'completed_comparisons':counts[0],'fp8_different_bytes':counts[1],
                       'scale_different_values':counts[2],'nonfinite_scales':counts[3],
                       'residual_different_values':counts[4],'nonfinite_residuals':counts[5],'python_launches':CALLS,'fallbacks':FALLBACKS,'policy':POLICY.snapshot()}
                path=directory/(stage+'-audit.json');tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
        threading.Thread(target=report,daemon=True).start()
    print('[paiton.rms] '+json.dumps({'policy':POLICY.snapshot(),'shadow':SHADOW}),file=sys.stderr,flush=True)

def wrap(module):
    if getattr(module,'_paiton_rms_wrapped',False):return
    original=module.launch_add_rms_quant
    def launch(y,residual,weight,q,scale,residual_out,M,K,eps,stream,tiled=0):
        global CALLS,FALLBACKS
        args=(y,residual,weight,q,scale,residual_out,M,K,eps,stream,tiled)
        if (M,K,tiled)!=(4089,5120,1) or eps not in (1.e-6,1.e-5):
            FALLBACKS+=1
            return original(*args)
        if LIB is None:load()
        if not POLICY.allows(M,K,arch='gfx1201',tiled=tiled,eps=eps):
            FALLBACKS+=1
            return original(*args)
        output,scale_output,res_output=q,scale,residual_out
        if SHADOW:
            import torch
            if torch.cuda.current_stream().cuda_stream!=stream:raise RuntimeError('Shadow allocation stream mismatch')
            temporary=torch.empty(((M+15)//16*16,K),device='cuda',dtype=torch.uint8)
            temporary_scale=torch.empty(M,device='cuda',dtype=torch.float32)
            temporary_residual=torch.empty((M,K),device='cuda',dtype=torch.bfloat16)
            output,scale_output,res_output=temporary.data_ptr(),temporary_scale.data_ptr(),temporary_residual.data_ptr()
            original(*args)
        check(LIB.paiton_rms_prefill(y,residual,weight,output,scale_output,res_output,M,K,eps,stream))
        CALLS+=1
        if SHADOW:check(AUDIT.paiton_rms_audit_compare(q,output,scale,scale_output,residual_out,res_output,M,K,stream))
        if CALLS in (1,64,256,512,4096):print('[paiton.rms] native launches='+str(CALLS),file=sys.stderr,flush=True)
        return None
    module.launch_add_rms_quant=launch
    module._paiton_rms_wrapped=True

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
if requested not in ('0','1','off','on','false','true'):raise ValueError('Invalid prefill enable flag')
if requested in ('1','on','true'):
    if TARGET in sys.modules:wrap(sys.modules[TARGET])
    else:sys.meta_path.insert(0,Finder())
