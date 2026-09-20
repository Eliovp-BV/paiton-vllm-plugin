"""External vLLM ownership/dispatch adapter; all new arithmetic is native HIP."""
import ctypes as C
import functools
import hashlib
import importlib.abc
import inspect
import json
import linecache
import os
from pathlib import Path
import sys
import threading
import time
from .context_policy import Policy, FLAG

TARGET = 'vllm.model_executor.models.qwen3_dflash'
ROOT = Path('/opt/paiton/runtime/dflash-context')
LIB = AUDIT = POLICY = None
CALLS = FALLBACKS = 0
ROWS = {}
LOCAL = threading.local()
SHADOW = os.environ.get('PAITON_DFLASH_CONTEXT_NORM_ROPE_SHADOW', '0') == '1'
STARTUP_FLAGS = {FLAG: os.environ.get(FLAG, '0')}

def check(rc):
    if rc: raise RuntimeError(f'Native context HIP status {rc}')

def load():
    global LIB, AUDIT, POLICY
    import torch  # Existing external runtime: device/stream/ownership only.
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Initialize native context before graph capture')
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx1201':
        raise RuntimeError('Native context requires one gfx1201 device')
    manifest = json.loads((ROOT/'manifest.json').read_text())
    path = (ROOT/manifest['file']).resolve(strict=True)
    if path.parent != ROOT or path.suffix != '.so': raise ValueError('Expected sibling native context artifact')
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Native context artifact mismatch')
    if hashlib.sha256(Path(sys.modules[__package__+'.context_policy'].__file__).read_bytes()).hexdigest() != manifest['policy_sha256']:
        raise ValueError('Native context policy mismatch')
    POLICY = Policy.from_environment(STARTUP_FLAGS, manifest)
    LIB = C.CDLL(str(path), mode=C.RTLD_LOCAL)
    LIB.paiton_context_arch.restype = C.c_char_p
    if LIB.paiton_context_abi() != 1 or LIB.paiton_context_arch() != b'gfx1201':
        raise ValueError('Native context ABI mismatch')
    LIB.paiton_context_norm_rope.argtypes = [C.c_void_p]*5+[C.c_int,C.c_int,C.c_float,C.c_void_p]
    LIB.paiton_context_norm_rope.restype = C.c_int
    if SHADOW:
        audit_path = ROOT/'audit.so'
        expected = json.loads((ROOT/'audit-manifest.json').read_text())['sha256']
        if hashlib.sha256(audit_path.read_bytes()).hexdigest() != expected: raise ValueError('Audit artifact mismatch')
        AUDIT = C.CDLL(str(audit_path), mode=C.RTLD_LOCAL)
        AUDIT.paiton_draft_audit_compare.argtypes = [C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]
        AUDIT.paiton_draft_audit_report.argtypes = [C.POINTER(C.c_ulonglong)]
        check(AUDIT.paiton_draft_audit_init())
        def report():
            directory = Path('/evidence'); deadline = time.monotonic()+1200
            for stage in ('warmup','final'):
                while time.monotonic()<deadline and not (directory/(stage+'-audit-request')).exists(): time.sleep(.2)
                if not (directory/(stage+'-audit-request')).exists(): return
                counts = (C.c_ulonglong*4)(); rc = AUDIT.paiton_draft_audit_report(counts)
                value = dict(hip_status=rc, completed_comparisons=counts[0], mismatched_elements=counts[1],
                    reference_nonfinite=counts[2], candidate_nonfinite=counts[3], python_launches=CALLS,
                    fallbacks=FALLBACKS, row_counts=dict(ROWS), policy=POLICY.snapshot())
                path=directory/(stage+'-audit.json'); tmp=path.with_suffix('.tmp')
                tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(path)
        threading.Thread(target=report,daemon=True).start()
    print('[paiton.dflash_context] '+json.dumps({'policy':POLICY.snapshot(),'shadow':SHADOW}),flush=True)

def compatible(model, states, positions):
    m = states.shape[0]
    if not POLICY.allows(m, arch='gfx1201', layers=model._num_attn_layers,
            heads=model._num_kv_heads, dim=model._head_dim, eps=model._rms_norm_eps,
            neox=model._rope_is_neox, shadow=SHADOW): return False
    w, cs = model._k_norm_weights, model._rope_cos_sin_cache
    tensors = (states, positions, w, cs)
    return (model._kv_size == 1024 and model._rope_head_size == 128
            and tuple(w.shape) == (5,128) and len(cs.shape) == 2 and cs.shape[1] == 128
            and 1 <= cs.shape[0] <= 262144 and tuple(positions.shape) == (m,)
            and all(t.is_cuda and t.device == states.device and t.is_contiguous() for t in tensors)
            and str(positions.dtype) == 'torch.int64'
            and all(str(t.dtype) == 'torch.bfloat16' for t in (states,w,cs)))

def launch(model, k, positions):
    global CALLS
    import torch
    m = k.shape[1]
    if tuple(k.shape) != (5,m,8,128) or str(k.dtype) != 'torch.bfloat16' or not k.is_contiguous():
        raise RuntimeError('Pinned context projection layout changed')
    if k.device != positions.device: raise RuntimeError('Context tensor device mismatch')
    out = torch.empty_like(k)
    stream = torch.cuda.current_stream(k.device).cuda_stream
    cs = model._rope_cos_sin_cache
    check(LIB.paiton_context_norm_rope(k.data_ptr(), model._k_norm_weights.data_ptr(),
        positions.data_ptr(), cs.data_ptr(), out.data_ptr(), m, cs.shape[0], model._rms_norm_eps, stream))
    CALLS += 1; ROWS[m] = ROWS.get(m,0)+1
    if CALLS in (1,64,256,1024): print('[paiton.dflash_context] native launches='+str(CALLS)+' m='+str(m),flush=True)
    return out, stream

def wrap(module):
    cls = module.DFlashQwen3Model
    if getattr(cls,'_paiton_context_wrapped',False): return
    original = cls.precompute_and_store_context_kv
    norm = cls._normalize_context_k
    build = cls._build_fused_kv_buffers
    if list(inspect.signature(original).parameters) != ['self','context_states','context_positions','context_slot_mapping']:
        raise RuntimeError('Context method signature changed')
    if hasattr(original,'__wrapped__'): raise RuntimeError('Context method already wrapped')

    @functools.wraps(build)
    def initialize(self,*args,**kwargs):
        result=build(self,*args,**kwargs)
        if LIB is None: load()
        return result

    @functools.wraps(norm)
    def observe_norm(self,all_k):
        result=norm(self,all_k)
        state=getattr(LOCAL,'active',None)
        if state is not None and state['model'] is self:
            if 'k' in state: raise RuntimeError('Context normalization count changed')
            state.update(k=all_k,reference=result)
        return result

    @functools.wraps(original)
    def context(self,context_states,context_positions,context_slot_mapping=None):
        global FALLBACKS
        if not hasattr(self,'_num_attn_layers'): self._build_fused_kv_buffers()
        if LIB is None: load()
        if not compatible(self,context_states,context_positions):
            FALLBACKS+=1
            return original(self,context_states,context_positions,context_slot_mapping)
        if SHADOW:
            if getattr(LOCAL,'active',None) is not None: raise RuntimeError('Nested context shadow')
            state={'model':self}; LOCAL.active=state
            try:
                result=original(self,context_states,context_positions,context_slot_mapping)
            finally:
                LOCAL.active=None
            if 'k' not in state: raise RuntimeError('Original context normalization not observed')
            out,stream=launch(self,state['k'],context_positions)
            check(AUDIT.paiton_draft_audit_compare(state['reference'].data_ptr(),out.data_ptr(),out.numel(),stream))
            return result
        m=context_states.shape[0]
        k,v=self._project_context_kv(context_states,m,5,8,128)
        out,_=launch(self,k,context_positions)
        if context_slot_mapping is None: return None
        per_layer=isinstance(context_slot_mapping,(list,tuple))
        for i,attn in enumerate(self._attn_layers):
            slots=context_slot_mapping[i] if per_layer else context_slot_mapping
            if slots is not None:
                attn.impl.do_kv_cache_update(attn,out[i],v[i],attn.kv_cache,slots)
        return None

    cls._build_fused_kv_buffers=initialize
    cls._normalize_context_k=observe_norm
    cls.precompute_and_store_context_kv=context
    cls._paiton_context_wrapped=True

def verify_source(module):
    expected=json.loads((ROOT/'external-source-receipts.json').read_text())
    original=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    effective=hashlib.sha256(''.join(linecache.getlines(module.__file__)).encode()).hexdigest()
    if original != expected['original_sha256']:
        raise RuntimeError('Context original or active overlay source changed')
    if effective != expected['overlay_sha256']:
        # The native MLP hook has one pinned source transformation. Verify that
        # exact composition, including its input, rather than relaxing the guard.
        from .native_hooks import MODEL, enabled, transform
        if not enabled(os.environ) or module.__name__ != MODEL:
            raise RuntimeError('Context original or active overlay source changed')
        loader = module.__spec__.loader
        data = loader.get_data(loader.path)
        if hashlib.sha256(data).hexdigest() != expected['overlay_sha256']:
            raise RuntimeError('Context native integration input changed')
        transformed = transform(MODEL, data)
        if effective != hashlib.sha256(transformed.encode()).hexdigest():
            raise RuntimeError('Context native integration source changed')

class Loader:
    def __init__(self,original): self.original=original
    def __getattr__(self,name): return getattr(self.original,name)
    def create_module(self,spec): return self.original.create_module(spec)
    def exec_module(self,module): self.original.exec_module(module); verify_source(module); wrap(module)

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname != TARGET: return None
        after=False
        for finder in list(sys.meta_path):
            if finder is self: after=True; continue
            if not after: continue
            spec=finder.find_spec(fullname,path,target)
            if spec is not None:
                if spec.loader is None: raise ImportError('Missing context model loader')
                spec.loader=Loader(spec.loader); return spec
        raise ImportError('Missing pinned context model')

if STARTUP_FLAGS[FLAG] not in ('0','1','off','on','false','true'): raise ValueError('Invalid context enable flag')
if STARTUP_FLAGS[FLAG] in ('1','on','true'):
    if TARGET in sys.modules: raise RuntimeError('Install context adapter before model import')
    sys.meta_path.insert(0,Finder())
