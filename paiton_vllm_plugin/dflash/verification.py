"""Experimental host-only verifier prefix cap; the drafter still produces seven tokens."""
import functools
import hashlib
import importlib.abc
import inspect
import json
import linecache
from collections import Counter
from pathlib import Path
import sys

TARGET = 'vllm.v1.worker.gpu.model_runner'
SOURCES = {'worker/gpu/model_runner.py': 'd7bae5b05a2a9f02c1ba29f0b4b32434cee43c3b369d3985505d3f8ed89b461b', 'worker/gpu/spec_decode/dflash2/speculator.py': '9ae6a9e27e8777d9590914cbc925d9cb3b66a3031e830abb468c7c4cb2295382', 'worker/gpu/spec_decode/dflash/speculator.py': '4a6b86232fa98eeb62143c469f2f5789c94d3af0f80851f623ddaa8c78e23f7b', 'worker/gpu/async_utils.py': '36bd876f1c2cf648ed9436c66205db39df7bd4f9b0da90f507fed8b3159840c5', 'worker/gpu/spec_decode/speculator.py': '9587daa4a84743e051b4bca586767b1c4d6ee9c53850db98d5727591316bea7f', 'worker/gpu/spec_decode/utils.py': '39ebdfdc8de50d7fddc324aa011275dccd38f2dcc32c4e3268dbbf3ea915fe49', 'worker/gpu/spec_decode/rejection_sampler.py': 'e20adc9b6c8a62a5be232bae15a1e361e898142966a68ea3ef244b39ffc5ce44', 'worker/gpu/spec_decode/rejection_sampler_utils.py': '20ca2e5ac34e9ef93dca388bed72a00d4ff67d569ec6ec2393b21a92d74a1fae', 'outputs.py': '9e3be802fd0107ace7f77806357e3fcd58670c1f705c8395053c751b2ad35841', 'core/sched/scheduler.py': 'abca7134821e2fb5cc8572df5c4a0b570ecf702646254327bc786fc452074983'}


OVERLAYS = {
    'worker/gpu/model_runner.py': 'ad78894ad3e74d10cea04aa9d8f942f5e8c61006738a83c76600469cffa1155c',
    'worker/gpu/spec_decode/dflash2/speculator.py': '40d0011daa0dda6aefe56bad41b73f400113305b9a0a8fb265490df5ae964192',
    'worker/gpu/async_utils.py': '7cc402cacac25137c38edafd415d66c508298f5fc1926f75cf297ec5f50b8484',
    'core/sched/scheduler.py': '4762cc252945ebfa812da34398cd114ec8538dabb68b15e9ea4455e7d961edc9',
}

def validate_active_source(module, root):
    runtime = root.parents[1]/'paiton_runtime_compat'
    entries = json.loads((runtime/'compat_manifest.json').read_text())['files']
    for relative in SOURCES:
        entry = entries.get(str(root/relative))
        if relative not in OVERLAYS:
            if entry is not None: raise RuntimeError('Unexpected verifier overlay: '+relative)
            continue
        if entry is None or entry['original_sha256'] != SOURCES[relative] or entry['overlay_sha256'] != OVERLAYS[relative]:
            raise RuntimeError('Verifier active overlay metadata changed: '+relative)
        path = (runtime/entry['overlay']).resolve(strict=True)
        if not path.is_relative_to(runtime) or hashlib.sha256(path.read_bytes()).hexdigest() != OVERLAYS[relative]:
            raise RuntimeError('Verifier active overlay content changed: '+relative)
    effective = hashlib.sha256(''.join(linecache.getlines(module.__file__)).encode()).hexdigest()
    if effective != OVERLAYS['worker/gpu/model_runner.py']:
        raise RuntimeError('Verifier active runner source changed')

def prefix_for_eight(request_ids, token_ids, cap):
    """Return a fresh prefix list only for the frozen eight-request/seven-placeholder domain; structured/token-bearing lists fall back."""
    if cap not in (3, 5): raise ValueError('Unsupported verifier cap')
    if len(request_ids) != len(token_ids) or len(set(request_ids)) != len(request_ids):
        raise ValueError('Draft request identities do not resolve uniquely')
    if len(request_ids) != 8: return token_ids
    if any(not isinstance(row, list) or len(row) != 7 for row in token_ids): return token_ids
    if any(type(token) is not int or token != -1 for row in token_ids for token in row):
        return token_ids
    return [row[:cap] for row in token_ids]

def validate_source(module):
    root = Path(module.__file__).resolve().parents[2]
    for relative, expected in SOURCES.items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Verifier cap requires the pinned vLLM source: '+relative)
    validate_active_source(module, root)

def bind(module, cap, trace=False):
    if cap not in ((0, 3, 5) if trace else (3, 5)): raise ValueError('Unsupported verifier cap')
    validate_source(module)
    cls = module.GPUModelRunner
    if getattr(cls, '_paiton_verifier_cap', None) is not None:
        raise RuntimeError('Verifier cap already installed')
    original = cls.take_draft_token_ids
    if list(inspect.signature(original).parameters) != ['self'] or hasattr(original, '__wrapped__'):
        raise RuntimeError('Verifier return boundary changed')
    prepare = cls.prepare_inputs
    if list(inspect.signature(prepare).parameters) != ['self', 'scheduler_output', 'batch_req_state', 'batch_desc'] or hasattr(prepare, '__wrapped__'):
        raise RuntimeError('Scheduled input boundary changed')
    counts = {'calls': 0, 'capped': 0, 'scheduled_capped': 0}
    seen = set()
    def observation(stage, payload):
        if not trace: return
        key = (stage, json.dumps(payload, sort_keys=True))
        if key not in seen and len(seen) < 64:
            seen.add(key)
            print('[PAITON_DFLASH_VERIFY_TRACE] '+json.dumps(dict(payload, stage=stage, call=counts['calls'], cap=cap)), flush=True)

    @functools.wraps(original)
    def take(self):
        config = self.speculative_config
        if self.scheduler_config.async_scheduling:
            raise RuntimeError('Verifier cap is qualified only for synchronous scheduling')
        if (self.num_speculative_steps != 7 or config.method != 'dflash'
                or config.num_speculative_tokens != 7 or config.draft_sample_method != 'probabilistic'
                or self.speculator.num_speculative_steps != 7 or self.speculator.num_query_per_req != 8
                or self.adaptive_verification is not None):
            raise RuntimeError('Verifier cap requires unchanged seven-token probabilistic DFlash without adaptive verification')
        result = original(self)
        counts['calls'] += 1
        if result is None:
            observation('returned', {'result': 'none'})
            return None
        observation('returned', {'requests': len(result.req_ids),
            'length_histogram': dict(Counter(len(row) for row in result.draft_token_ids)),
            'structured': self.draft_tokens_handler.draft_tokens_np is not None,
            'all_placeholders': all(type(token) is int and token == -1 for row in result.draft_token_ids for token in row)})
        if cap == 0: return result
        # The new runner normally returns CPU placeholders. Actual draft tokens
        # and pre-temperature logits remain in persistent GPU request-state rows.
        # Structured output lists retain the original full verification length.
        if self.draft_tokens_handler.draft_tokens_np is not None: return result
        clipped = prefix_for_eight(result.req_ids, result.draft_token_ids, cap)
        if clipped is result.draft_token_ids: return result
        mapping = self.req_states.req_id_to_index
        indices = [mapping.get(req_id, -1) for req_id in result.req_ids]
        logits = self.speculator.draft_logits
        if (logits is None or len(logits.shape) != 3 or logits.shape[1] != 7
                or len(set(indices)) != len(indices)
                or any(index < 0 or index >= logits.shape[0] for index in indices)):
            raise RuntimeError('Capped proposals lack matching persistent probability rows')
        counts['capped'] += 1
        if counts['capped'] in (1, 64, 256, 1024):
            print('[PAITON_DFLASH_VERIFY_CAP] '+json.dumps({
                'policy': 'verifier-prefix-r3', 'requests': 8, 'generated_per_request': 7,
                'returned_per_request': cap, 'capped_calls': counts['capped'],
                'draft_block_unchanged': True, 'cpu_values': 'placeholders',
                'persistent_gpu_draft_and_logits_unchanged': True}), flush=True)
        return type(result)(req_ids=list(result.req_ids), draft_token_ids=clipped)

    take._paiton_counts = counts

    @functools.wraps(prepare)
    def observe(self, scheduler_output, batch_req_state, batch_desc):
        result = prepare(self, scheduler_output, batch_req_state, batch_desc)
        lengths = result.num_draft_tokens_per_req
        if trace:
            observation('scheduled', {'requests': len(lengths) if lengths is not None else None,
                'length_histogram': dict(Counter(int(n) for n in lengths)) if lengths is not None else {},
                'logical_rows': int(result.num_tokens), 'padded_rows': int(result.num_tokens_after_padding)})
        if cap and lengths is not None and len(lengths) == 8 and all(n == cap for n in lengths):
            counts['scheduled_capped'] += 1
            if counts['scheduled_capped'] in (1, 64, 256, 1024):
                print('[PAITON_DFLASH_VERIFY_SCHEDULED] '+json.dumps({
                    'requests': 8, 'scheduled_drafts_per_request': cap,
                    'logical_target_rows': int(result.num_tokens),
                    'padded_gpu_rows': int(result.num_tokens_after_padding),
                    'scheduled_calls': counts['scheduled_capped']}), flush=True)
        return result

    cls.take_draft_token_ids = take
    cls.prepare_inputs = observe
    cls._paiton_verifier_cap = cap
    print('[PAITON_DFLASH_VERIFY_CAP] installed cap='+str(cap)+' exact_batch=8 sync_only', flush=True)

class Loader:
    def __init__(self, original, cap, trace): self.original, self.cap, self.trace = original, cap, trace
    def __getattr__(self, name): return getattr(self.original, name)
    def create_module(self, spec): return self.original.create_module(spec)
    def exec_module(self, module): self.original.exec_module(module); bind(module, self.cap, self.trace)

class Finder(importlib.abc.MetaPathFinder):
    def __init__(self, cap, trace): self.cap, self.trace = cap, trace
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET: return None
        after = False
        for finder in list(sys.meta_path):
            if finder is self: after = True; continue
            if not after: continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is None: raise ImportError('Missing verifier boundary loader')
                spec.loader = Loader(spec.loader, self.cap, self.trace); return spec
        raise ImportError('Missing pinned GPU model runner')

def install(cap, trace=False):
    if cap == 0 and not trace: return
    if cap not in ((0, 3, 5) if trace else (3, 5)): raise ValueError('Unsupported verifier cap')
    if TARGET in sys.modules: bind(sys.modules[TARGET], cap, trace)
    else: sys.meta_path.insert(0, Finder(cap, trace))
