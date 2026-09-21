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
from paiton_gdn_tiled_policy import Policy, FLAG

TARGET='radiance_arnq'
SOURCE_SHA='f76b0fc5819926d609e8f9665aab09ff02ea4da327017d90a6cb8feca2aeb4a7'
CONSUMER_SHA='461cc4096740897c2da6bb81b04e31020084bf5f6671c63152e67be8944f0610'
ROOT=Path('/opt/paiton/runtime/gdn-tiled')
LIB=AUDIT=POLICY=None
CALLS=FALLBACKS=0
PENDING={}
OBSERVED=set()
SHADOW=os.environ.get('PAITON_TARGET_GDN_TILED_SHADOW','0')=='1'

def check(rc):
    if rc:raise RuntimeError(f'Native GDN HIP status {rc}')

def load():
    global LIB,AUDIT,POLICY
    import torch  # Tensor ownership/device identity in existing external vLLM only.
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('Initialize native GDN before capture')
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx1201':
        raise RuntimeError('Native GDN requires one gfx1201 device')
    import radiance_mxfp4 as consumer
    if hashlib.sha256(Path(consumer.__file__).read_bytes()).hexdigest()!=CONSUMER_SHA:
        raise ImportError('Unreviewed tiled consumer source')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    path=(ROOT/manifest['file']).resolve(strict=True)
    if path.parent!=ROOT or path.suffix!='.so':raise ValueError('Expected sibling native GDN artifact')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('Native GDN artifact hash mismatch')
    if hashlib.sha256(Path(__import__('paiton_gdn_tiled_policy').__file__).read_bytes()).hexdigest()!=manifest['policy_sha256']:
        raise ValueError('Native GDN policy hash mismatch')
    POLICY=Policy.from_environment(STARTUP_FLAGS,manifest)
    LIB=C.CDLL(str(path),mode=C.RTLD_LOCAL)
    LIB.paiton_gdn_tiled_arch.restype=C.c_char_p
    if LIB.paiton_gdn_tiled_abi_version()!=1 or LIB.paiton_gdn_tiled_arch()!=b'gfx1201':raise ValueError('Native GDN ABI mismatch')
    LIB.paiton_gdn_tiled.argtypes=[C.c_void_p,C.c_void_p,C.c_long,C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_float,C.c_size_t,C.c_void_p]
    LIB.paiton_gdn_tiled.restype=C.c_int
    if SHADOW:
        audit_manifest=json.loads((ROOT/'audit-manifest.json').read_text())
        audit_path=ROOT/'audit.so'
        if hashlib.sha256(audit_path.read_bytes()).hexdigest()!=audit_manifest['sha256']:raise ValueError('Audit artifact mismatch')
        AUDIT=C.CDLL(str(audit_path),mode=C.RTLD_LOCAL)
        AUDIT.paiton_gdn_tiled_audit_compare.argtypes=[C.c_void_p]*7+[C.c_int]*2+[C.c_long,C.c_void_p]
        AUDIT.paiton_gdn_tiled_audit_report.argtypes=[C.POINTER(C.c_ulonglong)]
        AUDIT.paiton_gdn_tiled_audit_projection.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_void_p]
        check(AUDIT.paiton_gdn_tiled_audit_init())
        install_projection_shadow(consumer)
        directory=Path('/evidence')
        def report():
            deadline=time.monotonic()+1200
            for stage in ('warmup','final'):
                while time.monotonic()<deadline and not (directory/(stage+'-audit-request')).exists():time.sleep(.2)
                if not (directory/(stage+'-audit-request')).exists():return
                counts=(C.c_ulonglong*13)();rc=AUDIT.paiton_gdn_tiled_audit_report(counts)
                value={'hip_status':rc,'completed_comparisons':counts[0],'fp8_different_bytes':counts[1],
                       'scale_different_values':counts[2],'nonfinite_scales':counts[3],
                       'm64_comparisons':counts[4],'m4089_comparisons':counts[5],
                       'nonfinite_x':counts[6],'nonfinite_z':counts[7],'nonfinite_weights':counts[8],
                       'nonzero_padding':counts[9],
                       'projection_comparisons':counts[10],'projection_different_values':counts[11],
                       'projection_nonfinite_values':counts[12],'pending_shadow_outputs':len(PENDING),
                       'python_launches':CALLS,'fallbacks':FALLBACKS,'policy':POLICY.snapshot()}
                path=directory/(stage+'-audit.json');tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
        threading.Thread(target=report,daemon=True).start()
    print('[paiton.gdn_tiled] '+json.dumps({'policy':POLICY.snapshot(),'shadow':SHADOW}),file=sys.stderr,flush=True)

def install_projection_shadow(consumer):
    """Compare the complete row/fragment consumer on real weights and inputs."""
    op=consumer.mxfp4_linear_pq
    original=op._backend_fns.get('cuda',op._backend_fns.get(None))
    if original is None:raise ImportError('Missing existing projection backend')
    if getattr(consumer,'_paiton_gdn_tiled_shadow',False):raise RuntimeError('Duplicate projection shadow')
    def project(q,scale,weight,wscale,wref):
        marked=PENDING.pop(q.data_ptr(),None)
        result=original(q,scale,weight,wscale,wref)
        if marked is None:return result
        native_q,native_scale,m,k,stream=marked
        import torch  # Existing external allocation/stream boundary only.
        n=weight.shape[0]
        if (m,n,k)!=(4089,5120,6144) or torch.cuda.current_stream(q.device).cuda_stream!=stream:
            raise RuntimeError('Unexpected marked GDN projection shape or stream')
        alternative=torch.empty((m,n),dtype=torch.bfloat16,device=q.device)
        consumer._ext.launch_at(native_q.data_ptr(),weight.data_ptr(),wscale.data_ptr(),
            wref.data_ptr(),native_scale.data_ptr(),alternative.data_ptr(),m,n,k,stream)
        check(AUDIT.paiton_gdn_tiled_audit_projection(result.data_ptr(),alternative.data_ptr(),m,n,stream))
        return result
    op.register_kernel('cuda',project)
    consumer._paiton_gdn_tiled_shadow=True

def wrap(module):
    if getattr(module,'_paiton_gdn_tiled_wrapped',False):return
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ImportError('Unreviewed GDN custom-op source')
    op=module.gdn_norm_quant
    original=op._backend_fns.get('cuda',op._backend_fns.get(None))
    if original is None:raise ImportError('Missing existing GDN backend')

    def launch(x,z,weight,eps):
        global CALLS,FALLBACKS
        import torch  # Existing external allocator/view/stream boundary only.
        import radiance_mxfp4 as consumer
        # Initialize diagnostics on the first real GPU call, even a fallback shape.
        # M4089 need not occur during startup; a zero-count startup audit is valid.
        if LIB is None:load()
        M,N=x.shape
        allowed=(POLICY.allows(M,N,arch='gfx1201',z_stride=z.stride(0),eps=eps)
            and consumer.a_tiled_wanted(M)
            and x.is_cuda and z.is_cuda and weight.is_cuda
            and x.device==z.device==weight.device
            and x.dtype==z.dtype==weight.dtype==torch.bfloat16
            and x.is_contiguous() and tuple(z.shape)==(M,N) and z.stride(-1)==1
            and weight.is_contiguous() and weight.numel()==128)
        if not allowed:
            FALLBACKS+=1
            return original(x,z,weight,eps)
        # Reuse the pinned ar_add_rms_quant allocation/view pattern. No tensor math.
        backing=torch.empty(((M+15)//16*16,N),dtype=torch.float8_e4m3fn,device=x.device)
        q=backing[:M]
        scale=torch.empty((M,),dtype=torch.float32,device=x.device)
        stream=torch.cuda.current_stream(x.device).cuda_stream
        reference=original(x,z,weight,eps) if SHADOW else None
        check(LIB.paiton_gdn_tiled(x.data_ptr(),z.data_ptr(),z.stride(0),weight.data_ptr(),
            q.data_ptr(),scale.data_ptr(),M,N,eps,backing.numel()*backing.element_size(),stream))
        CALLS+=1
        if SHADOW:
            rq,rs=reference
            check(AUDIT.paiton_gdn_tiled_audit_compare(rq.data_ptr(),q.data_ptr(),rs.data_ptr(),
                scale.data_ptr(),x.data_ptr(),z.data_ptr(),weight.data_ptr(),M,N,z.stride(0),stream))
            if rq.data_ptr() in PENDING or len(PENDING)>=16:raise RuntimeError('Unconsumed GDN shadow output')
            PENDING[rq.data_ptr()]=(q,scale,M,N,stream)
            result=reference
        else:
            consumer.a_tiled_register(q,M,N)
            result=(q,scale)
        if CALLS in (1,64,256,512,4096):
            print('[paiton.gdn_tiled] native launches='+str(CALLS),file=sys.stderr,flush=True)
        return result
    op.register_kernel('cuda',launch)
    module._paiton_gdn_tiled_wrapped=True

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
if requested not in ('0','1','off','on','false','true'):raise ValueError('Invalid GDN decode enable flag')
if requested in ('1','on','true'):
    if TARGET in sys.modules:wrap(sys.modules[TARGET])
    else:sys.meta_path.insert(0,Finder())
