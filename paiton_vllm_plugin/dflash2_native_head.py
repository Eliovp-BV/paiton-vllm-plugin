"""External vLLM tensor adapter for the experimental standalone HIP draft head.

The target head, selector and verification stay upstream. Compiler/native code
has no Torch dependency; tensors and model lifecycle belong to this adapter.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch


def _check(status):
    if status:
        raise RuntimeError(f"Native DFlash2 head HIP status {status}")


class NativeDraftHead:
    def __init__(self, draft, scope):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare native draft head before graph capture")
        path = Path(os.environ["PAITON_DFLASH_HEAD_RUNTIME_MANIFEST"]).resolve(strict=True)
        manifest = json.loads(path.read_text())
        if (manifest.get("abi"), manifest.get("arch"), manifest.get("arithmetic")) != (
            1, "gfx1201", "int2_group128_coarse_r128_bf16_rerank_fp32_top16"
        ):
            raise ValueError("Unsupported native draft-head contract")
        binary = (path.parent / manifest["file"]).resolve(strict=True)
        if binary.parent != path.parent or binary.suffix != ".so":
            raise ValueError("Expected sibling native draft-head library")
        if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("Native draft-head artifact hash mismatch")
        weight = draft.lm_head.weight
        processor = draft.candidate_logits_processor
        if (tuple(weight.shape) != (248320, 5120) or weight.dtype != torch.bfloat16
            or not weight.is_cuda or not weight.is_contiguous()
            or weight.device != scope.error.device
            or draft.model.candidate_selector.top_k != 16
            or processor.scale != 1.0 or processor.soft_cap is not None):
            raise ValueError("Native draft head requires the pinned TP1 BF16 head and scale1/top16 contract")
        self.lib = ctypes.CDLL(str(binary), mode=ctypes.RTLD_LOCAL)
        self.lib.paiton_draft_head_arch.restype = ctypes.c_char_p
        if self.lib.paiton_draft_head_abi_version() != 1 or self.lib.paiton_draft_head_arch() != b"gfx1201":
            raise ValueError("Native draft-head binary ABI mismatch")
        size = self.lib.paiton_draft_head_scratch_bytes
        size.argtypes = [ctypes.c_int]*3
        size.restype = ctypes.c_size_t
        self.pack = self.lib.paiton_draft_head_pack
        self.pack.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*2 + [ctypes.c_void_p]
        self.pack.restype = ctypes.c_int
        self.run = self.lib.paiton_draft_head_run
        self.run.argtypes = [ctypes.c_void_p]*8 + [ctypes.c_int]*3 + [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        self.run.restype = ctypes.c_int
        self.n, self.k = weight.shape
        self.weight = weight
        self.scope = scope
        self.max_rows = int(manifest["max_rows"])
        if self.max_rows not in (16, 64):
            raise ValueError("Unsupported draft-head dispatch row limit")
        device = weight.device
        self.packed = torch.empty(self.n*self.k//4, dtype=torch.uint8, device=device)
        self.scales = torch.empty((self.n, self.k//128), dtype=torch.bfloat16, device=device)
        self.zeros = torch.empty_like(self.scales)
        self.scratch = torch.empty(size(self.max_rows, self.n, self.k), dtype=torch.uint8, device=device)
        stream = torch.cuda.current_stream(device).cuda_stream
        # Check pack independently even while synthetic warmup checking is unarmed.
        error = torch.empty(1, dtype=torch.int32, device=device)
        scope.runtime.clear_error(error, stream)
        _check(self.pack(weight.data_ptr(), self.packed.data_ptr(), self.scales.data_ptr(),
                        self.zeros.data_ptr(), error.data_ptr(), self.n, self.k, stream))
        scope.runtime.check_error(error, stream)
        self.audit = dict(artifact_sha256=manifest["sha256"], arithmetic=manifest["arithmetic"],
                          max_rows=self.max_rows, native_rows=set(), fallback_rows=set(),
                          target_verification="upstream_full_head")

    def __call__(self, hidden_states, current_weight):
        if current_weight is not self.weight:
            raise RuntimeError("Draft head changed after native weight preparation")
        if (hidden_states.ndim != 2 or hidden_states.shape[1] != self.k
            or hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous()
            or hidden_states.device != self.weight.device):
            raise ValueError("Native draft head requires contiguous device BF16 rows")
        rows = hidden_states.shape[0]
        if not 1 <= rows <= self.max_rows:
            self.audit["fallback_rows"].add(rows)
            return None
        ids = torch.empty((rows, 16), dtype=torch.int64, device=hidden_states.device)
        values = torch.empty((rows, 16), dtype=torch.float32, device=hidden_states.device)
        stream = torch.cuda.current_stream(hidden_states.device).cuda_stream
        _check(self.run(hidden_states.data_ptr(), self.weight.data_ptr(), self.packed.data_ptr(),
                        self.scales.data_ptr(), self.zeros.data_ptr(), ids.data_ptr(), values.data_ptr(),
                        self.scope.error.data_ptr(), rows, self.n, self.k,
                        self.scratch.data_ptr(), self.scratch.numel(), stream))
        self.audit["native_rows"].add(rows)
        return ids, values
