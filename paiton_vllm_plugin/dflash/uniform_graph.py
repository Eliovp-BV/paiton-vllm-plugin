"""Opt-in uniform target graph coverage using original vLLM graph ownership.

The external framework owns its existing graph pool and model operators. This
module adds only capture descriptors and guarded dispatch, with no new arithmetic
or framework dependency in the Paiton compiler.
"""
import functools
import hashlib
import importlib.abc
import json
import linecache
from pathlib import Path
import sys
from .context_graph_guards import configuration_allowed
from .uniform_graph_guards import runtime_scope, validate_runtime_sources
from .uniform_graph_policy import CASES, MAX_EXTRA_BYTES, eligible, extend_descriptors

TARGET = 'vllm.v1.worker.gpu.cudagraph_utils'
SOURCE_SHA = '024d0570b1808c2baa47ac0ebeb5407eeaf8dcaa847f55e844769038cc48caad'


def validate_source(module):
    source = Path(module.__file__).read_bytes()
    effective = ''.join(linecache.getlines(module.__file__)).encode() or source
    if any(hashlib.sha256(data).hexdigest() != SOURCE_SHA for data in (source, effective)):
        raise RuntimeError('Uniform target graphs require the pinned vLLM graph implementation')
    # Decode-only classification and active state layout are part of the contract.
    validate_runtime_sources(Path(module.__file__).resolve().parents[4])


def bind(module):
    validate_source(module)
    cls = module.ModelCudaGraphManager
    if getattr(cls, '_paiton_uniform_graphs', False):
        raise RuntimeError('Uniform target graph hook already installed')
    init = cls._init_candidates
    capture = cls.capture
    dispatch = cls.dispatch
    full = module.CUDAGraphMode.FULL
    piecewise = module.CUDAGraphMode.PIECEWISE

    @functools.wraps(init)
    def initialize(self):
        init(self)
        config = self.vllm_config
        if not configuration_allowed(config) or not runtime_scope(config) or self.max_num_reqs != 8:
            raise RuntimeError('Uniform target graph configuration is outside the qualified scope')
        if self.varlen_decode or self.decode_query_len != 8 or self.lora_capture_cases != [0]:
            raise RuntimeError('Uniform target graphs require the original fixed DFlash configuration')
        if full not in self._capture_descs or piecewise not in self._capture_descs:
            raise RuntimeError('Uniform target graphs require original FULL and PIECEWISE support')
        if str(config.model_config.dtype) != 'torch.bfloat16' or getattr(config.cache_config, 'mamba_ssm_cache_dtype', None) != 'float16':
            raise RuntimeError('Uniform target graphs require the pinned BF16/FP16-state model')
        text = config.model_config.hf_text_config
        if any(getattr(text, k, None) != v for k, v in {'model_type': 'qwen3_5_text', 'hidden_size': 5120, 'num_hidden_layers': 64, 'num_attention_heads': 24, 'num_key_value_heads': 4, 'head_dim': 256}.items()):
            raise RuntimeError('Uniform target graphs require the qualified Qwen target architecture')
        if config.cache_config.cache_dtype != 'fp8' or config.compilation_config.max_cudagraph_capture_size < 40:
            raise RuntimeError('Uniform target graph cache or capture bounds changed')
        torch = module.torch
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_properties(self.device).gcnArchName.split(':')[0] != 'gfx1201':
            raise RuntimeError('Uniform target graphs require the qualified single gfx1201 device')
        self._paiton_uniform_original_descs = self._capture_descs
        self._capture_descs, self._paiton_uniform_extra = extend_descriptors(self._capture_descs, module.BatchExecutionDescriptor, full)
        self._paiton_uniform_ready = False
        self._paiton_uniform_hits = {q: 0 for _, q in CASES}

    @functools.wraps(capture)
    def capture_uniform(self, *args, **kwargs):
        if self._max_full_descs_to_capture is not None:
            # Original memory estimator sees the full descriptor count and samples
            # its original largest graphs. Profiling graphs are never deployed.
            return capture(self, *args, **kwargs)
        all_descs = self._capture_descs
        try:
            self._capture_descs = self._paiton_uniform_original_descs
            result = capture(self, *args, **kwargs)
            torch = module.torch
            torch.cuda.synchronize(self.device)
            before_reserved = torch.cuda.memory_reserved(self.device)
            before_free = torch.cuda.mem_get_info(self.device)[0]
            self._capture_descs = {full: sorted(self._paiton_uniform_extra, key=lambda d: d.num_tokens, reverse=True)}
            capture(self, *args, **kwargs)
            torch.cuda.synchronize(self.device)
            reserved_delta = torch.cuda.memory_reserved(self.device) - before_reserved
            device_delta = before_free - torch.cuda.mem_get_info(self.device)[0]
            if max(reserved_delta, device_delta) > MAX_EXTRA_BYTES:
                raise RuntimeError('Uniform target graphs exceeded the 64 MiB additional capture budget')
            if not all(desc in self.graphs for desc in self._paiton_uniform_extra):
                raise RuntimeError('Uniform target graphs were not captured')
            self._paiton_uniform_ready = True
            print('[PAITON_DFLASH_UNIFORM_GRAPHS] '+json.dumps({'event': 'captured', 'cases': CASES,
                'extra_reserved_bytes': reserved_delta, 'extra_device_bytes': device_delta}), flush=True)
            return result
        finally:
            self._capture_descs = all_descs

    @functools.wraps(dispatch)
    def dispatch_uniform(self, num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=None):
        if self._paiton_uniform_ready and eligible(num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len):
            desc = next(d for d in self._paiton_uniform_extra if d.uniform_token_count == uniform_token_count)
            if desc not in self.graphs:
                raise RuntimeError('Qualified uniform target graph disappeared')
            self._paiton_uniform_hits[uniform_token_count] += 1
            hits = self._paiton_uniform_hits[uniform_token_count]
            if hits in (1, 64, 256):
                print('[PAITON_DFLASH_UNIFORM_GRAPHS] '+json.dumps({'event': 'dispatch', 'query_length': uniform_token_count, 'hits': hits}), flush=True)
            return desc
        return dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len)

    cls._init_candidates = initialize
    cls.capture = capture_uniform
    cls.dispatch = dispatch_uniform
    cls._paiton_uniform_graphs = True


class Loader:
    def __init__(self, original): self.original = original
    def __getattr__(self, name): return getattr(self.original, name)
    def create_module(self, spec): return self.original.create_module(spec)
    def exec_module(self, module): self.original.exec_module(module); bind(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET: return None
        after = False
        for finder in list(sys.meta_path):
            if finder is self: after = True; continue
            if not after: continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is None: raise ImportError('Missing uniform target graph loader')
                spec.loader = Loader(spec.loader)
                return spec
        raise ImportError('Missing pinned vLLM graph manager')


def install():
    if TARGET in sys.modules: raise RuntimeError('Install uniform target graphs before vLLM imports')
    sys.meta_path.insert(0, Finder())
