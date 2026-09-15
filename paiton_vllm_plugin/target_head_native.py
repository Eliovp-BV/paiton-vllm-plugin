"""External adapter for the approximate native greedy target head."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch


class NativeTargetHead:
    def __init__(self, model, shared_head):
        owners = [module for module in model.modules()
                  if hasattr(module, 'lm_head') and hasattr(module, 'logits_processor')]
        if len(owners) != 1:
            raise ValueError('Expected one target vocabulary-head owner')
        self.owner = owners[0]
        self.head = shared_head
        processor = self.owner.logits_processor
        if (self.owner.lm_head.weight is not shared_head.weight or processor.scale != 1.0
            or processor.soft_cap is not None or shared_head.max_rows != 16):
            raise ValueError('Expected shared target/draft BF16 vocabulary head with scale1')
        path = Path(os.environ['PAITON_TARGET_HEAD_RUNTIME_MANIFEST']).resolve(strict=True)
        spec = json.loads(path.read_text())
        if (spec.get('abi'), spec.get('arch')) != (1, 'gfx1201'):
            raise ValueError('Unsupported native target-head manifest')
        binary = (path.parent/spec['file']).resolve(strict=True)
        if (binary.parent != path.parent or binary.suffix != '.so'
            or hashlib.sha256(binary.read_bytes()).hexdigest() != spec['sha256']):
            raise ValueError('Native target-head artifact path/hash mismatch')
        self.lib = ctypes.CDLL(str(binary), mode=ctypes.RTLD_LOCAL)
        self.scatter = self.lib.paiton_target_logits
        self.scatter.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_void_p]
        self.scatter.restype = ctypes.c_int
        self.audit = dict(artifact_sha256=spec['sha256'], native_rows=set(),
                          fallback_rows=set(), policy='greedy_no_full_vocabulary_features',
                          arithmetic='approximate_int2_r128_bf16_rerank_top16')

    def __call__(self, hidden_states):
        rows = hidden_states.shape[0]
        if not 1 <= rows <= 16:
            self.audit['fallback_rows'].add(rows)
            return None
        result = self.head(hidden_states, self.owner.lm_head.weight)
        if result is None:
            return None
        ids, values = result
        logits = torch.empty((rows, self.head.n), dtype=torch.float32, device=hidden_states.device)
        status = self.scatter(ids.data_ptr(), values.data_ptr(), logits.data_ptr(),
                              self.head.scope.error.data_ptr(), rows, self.head.n,
                              torch.cuda.current_stream(hidden_states.device).cuda_stream)
        if status:
            raise RuntimeError(f'Native target logits HIP status {status}')
        self.audit['native_rows'].add(rows)
        return logits
