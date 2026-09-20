"""External vLLM graph scheduling prototype. No new tensor arithmetic.

Uses its existing Torch graph owner solely to retain original operator allocations.
This is not a native conversion of the operators being replayed.
"""
import ctypes as C
import json
import threading
import types

def tensor_key(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))

def model_key(model, states, positions):
    """Return a complete persistent-storage identity, or no eligible identity."""
    if (tuple(states.shape) != (8,5120) or tuple(positions.shape) != (8,)
            or str(states.dtype) != 'torch.bfloat16' or str(positions.dtype) != 'torch.int64'
            or not states.is_cuda or not positions.is_cuda
            or not states.is_contiguous() or not positions.is_contiguous()
            or states.device != positions.device): return None
    if (getattr(model,'_num_attn_layers',None),getattr(model,'_num_kv_heads',None),
            getattr(model,'_head_dim',None),getattr(model,'_kv_size',None)) != (5,8,128,1024): return None
    if (getattr(model,'_rope_head_size',None)!=128 or getattr(model,'_rope_is_neox',None) is not True
            or getattr(model,'_rms_norm_eps',None) not in (1e-6,1e-5)): return None
    tensors=[states,positions]
    for name,shape in (('_hidden_norm_weight',(5120,)),('_fused_kv_weight',(10240,5120)),('_k_norm_weights',(5,128)),('_rope_cos_sin_cache',None)):
        t=getattr(model,name,None)
        if (t is None or not t.is_cuda or t.device != states.device
                or str(t.dtype)!='torch.bfloat16' or not t.is_contiguous()): return None
        if shape is not None and tuple(t.shape)!=shape:return None
        if shape is None and (len(t.shape)!=2 or t.shape[1]!=128 or not 1<=t.shape[0]<=262144):return None
        tensors.append(t)
    bias=getattr(model,'_fused_kv_bias',None)
    if bias is not None:
        if (not bias.is_cuda or bias.device != states.device or tuple(bias.shape)!=(10240,)
                or str(bias.dtype)!='torch.bfloat16' or not bias.is_contiguous()): return None
        tensors.append(bias)
    if len(model._attn_layers)!=5: return None
    for attn in model._attn_layers:
        if attn.impl._is_per_token_head_quant: return None
        cache=attn.kv_cache
        if not hasattr(cache,'shape') or len(cache.shape)!=4 or tuple(cache.shape[1:])!=(8,880,256): return None
        if not cache.is_cuda or cache.device!=states.device or cache.stride()[-1]!=1: return None
        if str(cache.dtype) not in ('torch.uint8','torch.float8_e4m3fn','torch.bfloat16'): return None
        for t in (cache,attn._k_scale,attn._v_scale):
            if not t.is_cuda or t.device!=states.device: return None
            tensors.append(t)
        if any(str(t.dtype)!='torch.float32' or tuple(t.shape) not in ((),(1,)) for t in (attn._k_scale,attn._v_scale)):return None
    metadata=(model._rms_norm_eps,model._rope_head_size,model._rope_is_neox,
              tuple((type(a.impl).__module__,type(a.impl).__qualname__,a.impl.kv_cache_dtype,
                     a.impl.head_size,a.impl._is_per_token_head_quant) for a in model._attn_layers),
              bias is not None)
    return (tuple(tensor_key(t) for t in tensors),metadata),tuple(tensors)

def slot_list(slots, device):
    values=list(slots) if isinstance(slots,(list,tuple)) else [slots]*5
    if len(values)!=5: return None
    if any(t is None or tuple(t.shape)!=(8,) or str(t.dtype)!='torch.int64'
           or not t.is_cuda or t.device!=device or not t.is_contiguous() for t in values): return None
    return values

class CapturedTail:
    """At most one startup capture; changed storage falls back, never recaptures."""
    def __init__(self,original,torch_module=None,hip=None):
        self.original=original; self.torch=torch_module; self.hip=hip
        self.key=None; self.owners=(); self.slots=(); self.graph=None; self.stream=None
        self.replay_stream=None
        self.captures=self.replays=self.fallbacks=0
        self.lock=threading.Lock()

    def _libraries(self):
        if self.torch is None:
            import torch
            self.torch=torch
        if self.hip is None:
            self.hip=C.CDLL('/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/lib/libamdhip64.so.7')
            self.hip.hipMemsetAsync.argtypes=[C.c_void_p,C.c_int,C.c_size_t,C.c_void_p]
            self.hip.hipMemcpyAsync.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_void_p]

    @staticmethod
    def _check(rc):
        if rc: raise RuntimeError('Context graph HIP status '+str(rc))

    def __call__(self,model,states,positions,slots=None,*,capture_allowed=False):
        info=model_key(model,states,positions)
        if info is None:
            self.fallbacks+=1; return self.original(model,states,positions,slots)
        self._libraries(); torch=self.torch
        if torch.cuda.is_current_stream_capturing():
            self.fallbacks+=1; return self.original(model,states,positions,slots)
        identity,owners=info
        if self.graph is None:
            # No capture on a live request. Startup dummy calls retain their
            # original no-cache-write behavior; graph slots are inert (-1).
            result=self.original(model,states,positions,slots)
            if slots is not None or not capture_allowed: self.fallbacks+=1; return result
            if not self.lock.acquire(blocking=False): raise RuntimeError('Concurrent context capture')
            try:
                current=torch.cuda.current_stream(states.device)
                self.slots=tuple(torch.empty((8,),dtype=torch.int64,device=states.device) for _ in range(5))
                for t in self.slots:self._check(self.hip.hipMemsetAsync(t.data_ptr(),255,64,current.cuda_stream))
                # Warm the unchanged cache writers with only negative slots.
                self.original(model,states,positions,list(self.slots))
                self.stream=torch.cuda.Stream(device=states.device)
                self.stream.wait_stream(current)
                graph=torch.cuda.CUDAGraph()
                before=torch.cuda.memory_reserved(states.device)
                with torch.cuda.graph(graph,stream=self.stream):
                    self.original(model,states,positions,list(self.slots))
                current.wait_stream(self.stream)
                extra=torch.cuda.memory_reserved(states.device)-before
                if extra+163840+320>64*1024*1024: raise RuntimeError('Context graph exceeds 64MiB additional graph storage bound: '+str(extra))
                self.key=identity; self.owners=owners; self.graph=graph; self.captures+=1
                self.replay_stream=current.cuda_stream
                print('[PAITON_DFLASH_CONTEXT_GRAPH] '+json.dumps({'event':'captured','rows':8,'captures':self.captures,'extra_reserved_bytes':extra,'slots_inert':True}),flush=True)
            finally:self.lock.release()
            return result
        values=slot_list(slots,states.device) if slots is not None else None
        current=torch.cuda.current_stream(states.device)
        if identity!=self.key or values is None or current.cuda_stream!=self.replay_stream:
            self.fallbacks+=1; return self.original(model,states,positions,slots)
        if not self.lock.acquire(blocking=False): raise RuntimeError('Concurrent context graph replay')
        try:
            # Copy current device values; no slot index is a captured host constant.
            for dst,src in zip(self.slots,values):
                self._check(self.hip.hipMemcpyAsync(dst.data_ptr(),src.data_ptr(),64,3,current.cuda_stream))
            self.graph.replay()
            self.replays+=1
            if self.replays in (1,64,256,1024):
                print('[PAITON_DFLASH_CONTEXT_GRAPH] '+json.dumps({'event':'replayed','rows':8,'replays':self.replays}),flush=True)
        finally:self.lock.release()
        return None

class ContextGraph:
    """Bounded tail graph; unchanged projection executes eagerly every time.

    Two owned HIP-copy destinations keep captured K/V addresses stable. Only
    original normalization, position expansion, RoPE and cache writers replay.
    """
    def __init__(self,original,torch_module=None,hip=None):
        self.original=original
        self.inner=CapturedTail(self._tail,torch_module,hip)
        self.k=self.v=None
        self.outer_fallbacks=0
        self.lock=threading.Lock()
    @property
    def captures(self):return self.inner.captures
    @property
    def replays(self):return self.inner.replays
    @property
    def fallbacks(self):return self.outer_fallbacks+self.inner.fallbacks
    def _tail(self,model,states,positions,slots):
        names=('_num_attn_layers','_kv_size','_head_dim','_num_kv_heads','_rms_norm_eps',
               '_rope_head_size','_rope_cos_sin_cache','_rope_is_neox','_attn_layers')
        proxy=types.SimpleNamespace(**{name:getattr(model,name) for name in names})
        proxy._project_context_kv=lambda *args:(self.k,self.v)
        proxy._normalize_context_k=model._normalize_context_k
        return self.original(proxy,states,positions,slots)
    def __call__(self,model,states,positions,slots=None,*,capture_allowed=False):
        def fallback():
            self.outer_fallbacks+=1
            return self.original(model,states,positions,slots)
        info=model_key(model,states,positions)
        if info is None:return fallback()
        self.inner._libraries();torch=self.inner.torch
        if torch.cuda.is_current_stream_capturing():return fallback()
        current=torch.cuda.current_stream(states.device)
        if self.inner.graph is None:
            if slots is not None or not capture_allowed:return fallback()
        elif (info[0]!=self.inner.key or slot_list(slots,states.device) is None
              or current.cuda_stream!=self.inner.replay_stream):return fallback()
        if not self.lock.acquire(blocking=False):raise RuntimeError('Concurrent context tail call')
        try:
            if self.k is None:
                self.k=torch.empty((5,8,8,128),dtype=torch.bfloat16,device=states.device)
                self.v=torch.empty_like(self.k)
            k,v=model._project_context_kv(states,8,5,8,128)
            for src,dst in ((k,self.k),(v,self.v)):
                if (tuple(src.shape)!=(5,8,8,128) or str(src.dtype)!='torch.bfloat16'
                        or src.device!=states.device or not src.is_contiguous()):raise RuntimeError('Original projection layout changed')
                self.inner._check(self.inner.hip.hipMemcpyAsync(dst.data_ptr(),src.data_ptr(),81920,3,current.cuda_stream))
            return self.inner(model,states,positions,slots,capture_allowed=capture_allowed)
        finally:self.lock.release()
