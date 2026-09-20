"""Reuse qualified native DFlash capsules through scoped external-runtime hooks.

No target source/ABI or global runtime namespace changes. Only changed draft
compilation receives a new AOT qualname and backend tag. Native arithmetic lives
in the existing HIP artifacts; this module adds no tensor arithmetic.
"""
import functools
import hashlib
import importlib.abc
import importlib.util
import json
import linecache
import os
from pathlib import Path
import sys

MASTER='PAITON_DFLASH_NATIVE'
MODEL='vllm.model_executor.models.qwen3_dflash'
UTILS='vllm.v1.worker.gpu.spec_decode.dflash.utils'
DISPATCH='radiance_kernels'
SUFFIX='_paiton_native_plugin_v1'
HASHES={MODEL:'4e98732973aed543d605ce128ece361f35518ec82ac7809614ef5f915da634d2',
        UTILS:'0ff13c1c2b01e015ef976304d4d7c5675303ddcb70ef4912424eb31abaafaa53',
        DISPATCH:'7b7fa7047d4922d4fd1750a09beb8dc392978296a8f654a3d7ae05735ece6548'}

def enabled(env):
    value=env.get(MASTER,'0')
    if value not in ('0','1','off','on','false','true'):raise ValueError('Invalid native DFlash master flag')
    return value in ('1','on','true')

def transform(name,data):
    if hashlib.sha256(data).hexdigest()!=HASHES[name]:raise ImportError('Pinned DFlash integration source changed: '+name)
    source=importlib.util.decode_source(data)
    if name==MODEL:
        boundary='        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)'
        if source.count(boundary)!=1:raise ImportError('DFlash MLP construction boundary not unique')
        source=source.replace(boundary,'        from paiton_vllm_plugin.dflash.native_mlp import configure_mlp\n        configure_mlp(self.mlp, config)\n'+boundary)
    elif name==UTILS:
        old='set_model_tag("dflash_head")'
        if source.count(old)!=1:raise ImportError('Draft compilation tag not unique')
        source=source.replace(old,'set_model_tag("dflash_head'+SUFFIX+'")')
    else:raise ValueError('No shared dispatcher source transformation is permitted')
    return source

def bind_projection(module):
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=HASHES[DISPATCH]:raise ImportError('Shared dispatcher source changed')
    if getattr(module,'_paiton_native_scoped',False):raise RuntimeError('Duplicate native projection binding')
    ps=module._PS
    if ps is None:raise RuntimeError('Missing existing preshuffle provider')
    original=ps.gemm_a8w8_blockscale_preshuffle
    if getattr(original,'_paiton_native_scoped',False):raise RuntimeError('Existing native projection binding')
    @functools.wraps(original)
    def projected(A,B,As,Bs,dtype,*extra,**kwargs):
        used_fallback=False
        def fallback():
            nonlocal used_fallback
            used_fallback=True
            return original(A,B,As,Bs,dtype,*extra,**kwargs)
        # Keep all unsupported calls, transpose contracts and output types exact.
        if extra or dtype!=module.torch.bfloat16 or kwargs.get('is_x_scale_tranposed',True) is not False or set(kwargs)-{'config','is_x_scale_tranposed'}:
            return fallback()
        if len(A.shape)!=2 or len(B.shape)!=2:return fallback()
        from .native_runtime import run
        result=run(A,B,As,Bs,B.shape[0]*16,A.shape[1],fallback)
        if not used_fallback and not getattr(module,'_paiton_scoped_down_observed',False):
            module._paiton_scoped_down_observed=True
            print('[PAITON_DFLASH_SCOPED_DOWN_EXECUTED] native eligible dispatch; host capture observation',flush=True)
        return result
    projected._paiton_native_scoped=True
    ps.gemm_a8w8_blockscale_preshuffle=projected
    module._paiton_native_scoped=True
    print('[PAITON_DFLASH_SCOPED_PROJECTION] existing custom-op ABI unchanged',flush=True)

def after_load(module):
    if module.__name__==MODEL:
        forward=module.DFlashQwen3Model.forward
        if forward.__qualname__!='DFlashQwen3Model.forward':raise RuntimeError('Unexpected draft forward identity')
        forward.__qualname__+=SUFFIX
        print('[PAITON_DFLASH_SCOPED_MODEL] '+json.dumps({'aot_qualname':forward.__qualname__,'draft_backend_tag':'dflash_head'+SUFFIX,'target_graph_unchanged':True}),flush=True)
    elif module.__name__==DISPATCH:bind_projection(module)

class Loader:
    def __init__(self,original,name):self.original,self.name=original,name
    def __getattr__(self,name):return getattr(self.original,name)
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        if self.name==DISPATCH:
            self.original.exec_module(module)
        else:
            data=self.original.get_data(self.original.path)
            source=transform(self.name,data)
            filename=getattr(self.original,'original',self.original.path)
            linecache.cache[filename]=(len(source.encode()),None,source.splitlines(True),filename)
            exec(compile(source,filename,'exec'),module.__dict__)
        after_load(module)

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in HASHES:return None
        after=False
        for finder in list(sys.meta_path):
            if finder is self:after=True;continue
            if not after:continue
            spec=finder.find_spec(fullname,path,target)
            if spec is not None:
                if spec.loader:spec.loader=Loader(spec.loader,fullname)
                return spec
        return None

def install():
    global SUFFIX
    if enabled(os.environ):
        from .native_runtime import POLICY
        # Producer on/off changes the captured graph. Artifact/feature identity
        # must not alias when the external AOT loader skips shape guards.
        SUFFIX = '_paiton_native_plugin_v1_' + POLICY.cache_key
        for name in (MODEL,UTILS):
            if name in sys.modules:raise RuntimeError('Native DFlash must bind before draft model import')
        if DISPATCH in sys.modules:bind_projection(sys.modules[DISPATCH])
        sys.meta_path.insert(0,Finder())
