"""Startup-only host configuration; no model, GPU, network or framework imports."""
from dataclasses import dataclass
from typing import Mapping

FLAGS = ('PAITON_DFLASH_NATIVE', 'PAITON_DFLASH_DOWN_PROJECTION',
         'PAITON_DFLASH_SILU_QUANT', 'PAITON_DFLASH_GATE_PROJECTION',
         'PAITON_DFLASH_CONTEXT_KV', 'PAITON_DFLASH_NATIVE_BUNDLE',
         'PAITON_DFLASH_RERANK', 'PAITON_DFLASH_DRAFT_SAMPLE_METHOD',
         'PAITON_DFLASH_BLOCK_CANDIDATES', 'PAITON_DFLASH_VERIFY_CAP', 'PAITON_DFLASH_VERIFY_TRACE', 'PAITON_DFLASH_CONTEXT_NORM_ROPE', 'PAITON_DFLASH_CONTEXT_GRAPH', 'PAITON_DFLASH_UNIFORM_GRAPHS')


def boolean(value, name):
    if value in ('1', 'on', 'true'): return True
    if value in ('0', 'off', 'false'): return False
    raise ValueError(name + ' must be 0/1, off/on or false/true')


@dataclass(frozen=True)
class Settings:
    native: bool
    rerank: int
    sample_method: str
    block_candidates: int
    verify_cap: int
    verify_trace: bool
    context_norm_rope: bool
    context_graph: bool
    uniform_graphs: bool
    requested: tuple

    @classmethod
    def from_environment(cls, env: Mapping[str, str]):
        native = boolean(env.get('PAITON_DFLASH_NATIVE', '0'), 'PAITON_DFLASH_NATIVE')
        for name in FLAGS[1:5]:
            if name in env:
                requested = boolean(env[name], name)
                if native and requested and name in ('PAITON_DFLASH_GATE_PROJECTION', 'PAITON_DFLASH_CONTEXT_KV'):
                    raise ValueError(name + ' is not available in this native bundle')
        pool = env.get('PAITON_DFLASH_RERANK', '0')
        if pool not in ('0', '128', '256'):
            raise ValueError('PAITON_DFLASH_RERANK supports 0, experimental 128 or unqualified 256')
        sample = env.get('PAITON_DFLASH_DRAFT_SAMPLE_METHOD', 'inherit')
        if sample not in ('inherit', 'greedy', 'probabilistic'):
            raise ValueError('PAITON_DFLASH_DRAFT_SAMPLE_METHOD must be inherit, greedy or probabilistic')
        block = env.get('PAITON_DFLASH_BLOCK_CANDIDATES', '0')
        if block not in ('0', '16'):
            raise ValueError('PAITON_DFLASH_BLOCK_CANDIDATES supports 0 (inherit eight) or experimental 16')
        if block != '0' and (pool != '128' or sample != 'probabilistic'):
            raise ValueError('Sixteen block candidates require the scoped R128 probabilistic proposal policy')
        cap = env.get('PAITON_DFLASH_VERIFY_CAP', '0')
        if cap not in ('0', '3', '5'):
            raise ValueError('PAITON_DFLASH_VERIFY_CAP supports 0, experimental 3 or 5')
        if cap != '0' and (pool != '128' or sample != 'probabilistic' or block != '0'):
            raise ValueError('Verifier caps require the scoped R128 probabilistic policy with original block retention')
        trace = boolean(env.get('PAITON_DFLASH_VERIFY_TRACE', '0'), 'PAITON_DFLASH_VERIFY_TRACE')
        if trace and (pool != '128' or sample != 'probabilistic' or block != '0'):
            raise ValueError('Verifier tracing requires the scoped R128 probabilistic policy')
        context = boolean(env.get('PAITON_DFLASH_CONTEXT_NORM_ROPE', '0'), 'PAITON_DFLASH_CONTEXT_NORM_ROPE')
        graph = boolean(env.get('PAITON_DFLASH_CONTEXT_GRAPH', '0'), 'PAITON_DFLASH_CONTEXT_GRAPH')
        uniform = boolean(env.get('PAITON_DFLASH_UNIFORM_GRAPHS', '0'), 'PAITON_DFLASH_UNIFORM_GRAPHS')
        if uniform and (native or pool != '0' or sample != 'inherit' or block != '0' or cap != '0' or trace or context or graph):
            raise ValueError('Uniform target graphs are initially qualified only with the original DFlash policy')
        if graph and context:
            raise ValueError('Context graph and native norm/RoPE are separate experimental policies')
        return cls(native, int(pool), sample, int(block), int(cap), trace, context, graph, uniform, tuple((k, env[k]) for k in FLAGS if k in env))

    def configure_speculation(self, spec):
        result = dict(spec)
        if self.native or self.rerank or self.sample_method != 'inherit' or self.verify_cap or self.context_norm_rope or self.context_graph or self.uniform_graphs:
            if result.get('method') != 'dflash' or result.get('num_speculative_tokens') != 7:
                raise ValueError('Experimental integration requires the pinned DFlash seven-token proposal')
            if result.get('draft_tensor_parallel_size', 1) != 1:
                raise ValueError('Experimental DFlash integration requires draft TP1')
        if self.sample_method != 'inherit':
            result['draft_sample_method'] = self.sample_method
        if self.rerank and result.get('draft_sample_method') != 'probabilistic':
            raise ValueError('Wider pools are currently screened only with probabilistic draft proposals')
        return result
