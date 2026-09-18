"""External vLLM adapter for the independently built native HIP Q/K producer.

No compiler or kernel source is shipped here. Unsupported rows retain the
original norm operations, and RoPE always stays with the original vLLM layer.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path

_installed = False


def install():
    global _installed
    if _installed:
        return
    import torch
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    path = Path(os.environ['PAITON_RUNTIME_COMPAT_QK_MANIFEST']).resolve(strict=True)
    manifest = json.loads(path.read_text())
    if (manifest.get('abi') != 1 or manifest.get('arch') != 'gfx1201'
            or manifest.get('arithmetic') != 'gemma_fp32_gamma_plus_one_rms256_bf16_qk_raw_gate'
            or manifest.get('geometry') != [24, 4, 256, 14336]):
        raise ValueError('Unsupported native QK contract')
    rows = frozenset(manifest['qualified_rows'])
    if not rows or not rows <= set(range(8, 65, 8)):
        raise ValueError('Unqualified native QK row shape')
    artifact = manifest['artifact']
    binary = (path.parent / artifact['file']).resolve(strict=True)
    if binary.parent != path.parent or hashlib.sha256(binary.read_bytes()).hexdigest() != artifact['sha256']:
        raise ValueError('Native QK artifact hash/path mismatch')
    library = ctypes.CDLL(str(binary), mode=ctypes.RTLD_LOCAL)
    library.paiton_qk_norm_split_abi_version.restype = ctypes.c_int
    library.paiton_qk_norm_split_arch.restype = ctypes.c_char_p
    if library.paiton_qk_norm_split_abi_version() != 1 or library.paiton_qk_norm_split_arch() != b'gfx1201':
        raise ValueError('Native QK binary ABI mismatch')
    launch = library.paiton_qk_norm_split
    launch.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int, ctypes.c_int64, ctypes.c_float, ctypes.c_void_p]
    launch.restype = ctypes.c_int
    layers = {}
    state = {'error': None, 'selected_rows': set()}

    @torch.library.custom_op('paiton_runtime::qk_norm_split', mutates_args=('error',))
    def produce(qkv: torch.Tensor, qw: torch.Tensor, kw: torch.Tensor,
                error: torch.Tensor, layer_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(layers) != 16:
            raise RuntimeError('Incomplete native QK layer binding')
        layer = layers[layer_name]
        m = qkv.shape[0]
        eligible = (m in rows and qkv.shape[1] == 14336
                    and qkv.dtype == qw.dtype == kw.dtype == torch.bfloat16
                    and qkv.device == qw.device == kw.device == error.device
                    and qkv.stride(1) == 1 and 14336 <= qkv.stride(0) <= 65536
                    and qw.is_contiguous() and kw.is_contiguous())
        if eligible:
            q = torch.empty((m, 6144), device=qkv.device, dtype=qkv.dtype)
            k = torch.empty((m, 1024), device=qkv.device, dtype=qkv.dtype)
            gate = torch.empty_like(q)
            rc = launch(qkv.data_ptr(), qw.data_ptr(), kw.data_ptr(),
                        q.data_ptr(), k.data_ptr(), gate.data_ptr(), 0,
                        error.data_ptr(), m, qkv.stride(0), layer.q_norm.variance_epsilon,
                        torch.cuda.current_stream(qkv.device).cuda_stream)
            if rc:
                raise RuntimeError(f'Native QK launch failed: {rc}')
            if m not in state['selected_rows']:
                state['selected_rows'].add(m)
                print(f'[paiton.runtime_qk] native rows={m}; unchanged RoPE', flush=True)
        else:
            # Identical original split/norm arithmetic for unqualified rows.
            qg, k, _ = qkv.split([12288, 1024, 1024], dim=-1)
            q, gate = torch.chunk(qg.view(m, 24, 512), 2, dim=-1)
            q = layer.q_norm(q.reshape(m, 24, 256)).reshape(m, 6144)
            k = layer.k_norm(k.reshape(m, 4, 256)).reshape(m, 1024)
            gate = gate.reshape(m, 6144).clone()
        # One asynchronous sticky-error check after the final target attention
        # layer, on the same stream. No per-layer host synchronization.
        if layer._paiton_runtime_qk_last:
            torch._assert_async(error == 0, 'Paiton native QK nonfinite/arithmetic error')
        return q, k, gate

    @produce.register_fake
    def produce_fake(qkv, qw, kw, error, layer_name):
        m = qkv.shape[0]
        return qkv.new_empty((m, 6144)), qkv.new_empty((m, 1024)), qkv.new_empty((m, 6144))

    original_init = Qwen3NextAttention.__init__
    original_project = Qwen3NextAttention._project_qkv_gate

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if (self.num_heads, self.num_kv_heads, self.head_dim) != (24, 4, 256):
            self._paiton_runtime_qk_enabled = False
            return
        if (not self.attn_output_gate or self.use_fused_qk_norm_rope_gate
                or not isinstance(self.q_norm, GemmaRMSNorm)
                or not isinstance(self.k_norm, GemmaRMSNorm)
                or self.q_norm.variance_epsilon != self.k_norm.variance_epsilon):
            raise ValueError('Unexpected target QK arithmetic contract')
        name = f'target_attention_{len(layers)}'
        if len(layers) >= 16:
            raise ValueError('Expected exactly 16 target attention layers')
        if state['error'] is None:
            state['error'] = torch.zeros(1, dtype=torch.int32, device=self.q_norm.weight.device)
        self.register_buffer('_paiton_runtime_qk_error', state['error'], persistent=False)
        self._paiton_runtime_qk_name = name
        self._paiton_runtime_qk_last = len(layers) == 15
        self._paiton_runtime_qk_enabled = True
        layers[name] = self
        if self._paiton_runtime_qk_last:
            print('[paiton.runtime_qk] bound 16 target attention layers', flush=True)

    def project(self, qkv, positions):
        if not self._paiton_runtime_qk_enabled:
            return original_project(self, qkv, positions)
        q, k, gate = produce(qkv, self.q_norm.weight, self.k_norm.weight,
                            self._paiton_runtime_qk_error, self._paiton_runtime_qk_name)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, qkv[..., 13312:14336], gate

    Qwen3NextAttention.__init__ = init
    Qwen3NextAttention._project_qkv_gate = project
    _installed = True
