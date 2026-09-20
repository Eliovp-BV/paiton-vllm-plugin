"""Experimental plugin scheduling hook over the unchanged external vLLM operators."""
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
from .context_graph_runtime import ContextGraph, slot_list, model_key
from .context_graph_guards import capture_allowed, configuration_allowed

TARGET='vllm.model_executor.models.qwen3_dflash'
SPECULATOR='vllm.v1.worker.gpu.spec_decode.dflash.speculator'
ROOT=Path('/opt/paiton/runtime/dflash-context-graph')
ENABLED=os.environ.get('PAITON_DFLASH_CONTEXT_GRAPH','0') in ('1','on','true')
SHADOW=os.environ.get('PAITON_DFLASH_CONTEXT_GRAPH_SHADOW','0')=='1'
AUDIT=None
RUNTIME=None
QUALIFIED=False

def check(rc):
    if rc:raise RuntimeError('Context graph audit HIP status '+str(rc))

def initialize_audit():
    global AUDIT
    if AUDIT is not None:return
    path=ROOT/'audit.so';manifest=json.loads((ROOT/'audit-manifest.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:raise RuntimeError('Audit artifact changed')
    AUDIT=C.CDLL(str(path))
    AUDIT.audit_cache.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int64,C.c_int64,C.c_int64,C.c_int,C.c_int,C.c_void_p]
    AUDIT.audit_report.argtypes=[C.POINTER(C.c_ulonglong)]
    check(AUDIT.audit_init())
    def report():
        deadline=time.monotonic()+1200
        for stage in ('warmup','final'):
            request=Path('/evidence')/(stage+'-audit-request')
            while time.monotonic()<deadline and not request.exists():time.sleep(.2)
            if not request.exists():return
            counters=(C.c_ulonglong*6)();rc=AUDIT.audit_report(counters)
            result=dict(zip(('completed_layer_comparisons','mismatched_elements','reference_nonfinite','candidate_nonfinite','bad_slots','compared_elements'),counters))
            result.update(hip_status=rc,captures=RUNTIME.captures,replays=RUNTIME.replays,fallbacks=RUNTIME.fallbacks,configuration_qualified=QUALIFIED)
            path=request.with_name(stage+'-audit.json');tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    threading.Thread(target=report,daemon=True).start()

def cache_audit(model,slots,compare):
    import torch
    device=model._attn_layers[0].kv_cache.device
    maps=slot_list(slots,device)
    if maps is None:raise RuntimeError('Audited graph replay lacks valid slot metadata')
    stream=torch.cuda.current_stream(device).cuda_stream
    for i,(a,mapping) in enumerate(zip(model._attn_layers,maps)):
        cache=a.kv_cache;s=cache.stride()
        check(AUDIT.audit_cache(cache.data_ptr(),mapping.data_ptr(),cache.shape[0],cache.shape[2],cache.element_size(),s[0],s[1],s[2],i,int(compare),stream))

def verify_source(module):
    if module.__name__==SPECULATOR:
        source=Path(module.__file__).read_bytes()
        effective=''.join(linecache.getlines(module.__file__)).encode() or source
        expected='4a6b86232fa98eeb62143c469f2f5789c94d3af0f80851f623ddaa8c78e23f7b'
        if hashlib.sha256(source).hexdigest()!=expected or hashlib.sha256(effective).hexdigest()!=expected:raise RuntimeError('Speculator capture source changed')
        return
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!='51d2b55f883d34393e9b732952fc866ae0308b2a39cfcc58243e0990802afcfd':raise RuntimeError('Original context source changed')
    if hashlib.sha256(''.join(linecache.getlines(module.__file__)).encode()).hexdigest()!='4e98732973aed543d605ce128ece361f35518ec82ac7809614ef5f915da634d2':raise RuntimeError('Effective context source changed')

def wrap(module):
    if module.__name__==SPECULATOR:
        cls=module.DFlashSpeculator;capture=cls.capture
        if list(inspect.signature(capture).parameters)!=['self'] or hasattr(capture,'__wrapped__'):raise RuntimeError('Unexpected speculator capture method')
        @functools.wraps(capture)
        def capture_context(self):
            result=capture(self)
            from vllm.compilation import monitor
            import torch
            if not QUALIFIED:return result
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('Nested context startup capture')
            monitor.validate_cudagraph_capturing_enabled()
            # Cache allocation and original query captures are complete. At most
            # two inert calls allow original lazy projection initialization.
            for _ in range(2):
                self.model.precompute_and_store_context_kv(self.hidden_states[:8],self.context_positions[:8],None)
                if RUNTIME.captures:break
            if RUNTIME.captures!=1:raise RuntimeError('No eligible explicit startup context capture; see metadata diagnostic')
            return result
        cls.capture=capture_context
        return
    cls=module.DFlashQwen3Model;original=cls.precompute_and_store_context_kv;build=cls._build_fused_kv_buffers
    if list(inspect.signature(original).parameters)!=['self','context_states','context_positions','context_slot_mapping'] or hasattr(original,'__wrapped__'):
        raise RuntimeError('Unexpected or already wrapped context method')
    @functools.wraps(build)
    def initialize(self,*args,**kwargs):
        global QUALIFIED,RUNTIME
        result=build(self,*args,**kwargs)
        if RUNTIME is not None:return result
        import torch
        from vllm.config import get_current_vllm_config
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Initialize context graph adapter before capture')
        QUALIFIED=configuration_allowed(get_current_vllm_config())
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx1201':QUALIFIED=False
        RUNTIME=ContextGraph(original)
        if SHADOW:initialize_audit()
        print('[PAITON_DFLASH_CONTEXT_GRAPH] '+json.dumps({'event':'initialized','configuration_qualified':QUALIFIED,'shadow':SHADOW}),flush=True)
        return result
    @functools.wraps(original)
    def context(self,context_states,context_positions,context_slot_mapping=None):
        if not hasattr(self,'_num_attn_layers'):self._build_fused_kv_buffers()
        if RUNTIME is None:raise RuntimeError('Context graph initialization not observed')
        if not QUALIFIED:return original(self,context_states,context_positions,context_slot_mapping)
        from vllm.compilation import monitor
        if context_states.shape[0]==8 and context_slot_mapping is None:
            import torch
            metadata={'event':'startup_m8_metadata','states_shape':list(context_states.shape),'states_dtype':str(context_states.dtype),
                'key_eligible':model_key(self,context_states,context_positions) is not None,
                'capture_enabled':monitor.cudagraph_capturing_enabled,'nested_capture':torch.cuda.is_current_stream_capturing(),
                'cache_shapes':[list(a.kv_cache.shape) for a in self._attn_layers],
                'cache_strides':[list(a.kv_cache.stride()) for a in self._attn_layers],
                'cache_dtypes':[str(a.kv_cache.dtype) for a in self._attn_layers],
                'weight_shapes':{k:list(getattr(self,k).shape) if getattr(self,k,None) is not None else None for k in ('_hidden_norm_weight','_fused_kv_weight','_k_norm_weights','_rope_cos_sin_cache')}}
            print('[PAITON_DFLASH_CONTEXT_GRAPH] '+json.dumps(metadata),flush=True)
        before=RUNTIME.replays
        result=RUNTIME(self,context_states,context_positions,context_slot_mapping,
            capture_allowed=capture_allowed(context_slot_mapping,monitor))
        if SHADOW and RUNTIME.replays!=before:
            cache_audit(self,context_slot_mapping,False)
            # Restore/use original eager cache output in the model shadow.
            original(self,context_states,context_positions,context_slot_mapping)
            cache_audit(self,context_slot_mapping,True)
        return result
    cls._build_fused_kv_buffers=initialize
    cls.precompute_and_store_context_kv=context

class Loader:
    def __init__(self,original):self.original=original
    def __getattr__(self,name):return getattr(self.original,name)
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):self.original.exec_module(module);verify_source(module);wrap(module)
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in (TARGET,SPECULATOR):return None
        after=False
        for finder in list(sys.meta_path):
            if finder is self:after=True;continue
            if not after:continue
            spec=finder.find_spec(fullname,path,target)
            if spec is not None:
                if spec.loader is None:raise ImportError('Missing context graph loader')
                spec.loader=Loader(spec.loader);return spec
        raise ImportError('Missing context model')
if ENABLED:
    if any(name in sys.modules for name in (TARGET,SPECULATOR)):raise RuntimeError('Install context graph adapter before model/speculator import')
    if os.environ.get('PAITON_DFLASH_CONTEXT_NORM_ROPE','0') not in ('0','off','false'):raise RuntimeError('Context graph round excludes separate native fusion')
    sys.meta_path.insert(0,Finder())
