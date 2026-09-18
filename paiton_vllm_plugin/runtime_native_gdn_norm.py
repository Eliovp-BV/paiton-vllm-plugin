"""External external runtime adapter: native HIP GDN normalization for qualified small rows.

The compiler and generated HIP source remain private. Large/unqualified shapes
use the original external runtime custom op. This module owns only serving integration.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import types

_installed = False
SMALL_ROWS = frozenset((8, 16, 24, 32))


def _install_output_error_guard(error):
    """Check native sticky status at vLLM's existing output synchronization.

    This is external serving glue, not a compiler/runtime dependency. The copy
    is outside graph replay and precedes AsyncOutput's existing completion event.
    Each output owns its host snapshot; later steps cannot overwrite it.
    """
    import functools
    import inspect
    import torch
    from vllm.v1.worker.gpu.async_utils import AsyncOutput

    expected = ('self', 'model_runner_output', 'sampler_output',
                'num_sampled_tokens', 'main_stream', 'copy_stream',
                'check_ep_fault', 'routed_experts')
    original_init = AsyncOutput.__init__
    original_get = AsyncOutput.get_output
    if tuple(inspect.signature(original_init).parameters) != expected:
        raise RuntimeError('Unsupported vLLM AsyncOutput error-check contract')
    if getattr(AsyncOutput, '_paiton_gdn_guard', False):
        raise RuntimeError('GDN output error guard already installed')

    @functools.wraps(original_init)
    def guarded_init(self, model_runner_output, sampler_output,
                     num_sampled_tokens, main_stream, copy_stream,
                     check_ep_fault=False, routed_experts=None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('GDN output validation must run outside graph capture')
        self._paiton_gdn_status = torch.empty(1, dtype=torch.int32,
                                             device='cpu', pin_memory=True)
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_stream(main_stream)
            self._paiton_gdn_status.copy_(error, non_blocking=True)
        # Do not race the readback against writes from subsequent model steps.
        # This waits only for the four-byte copy, not all token-output transfers.
        main_stream.wait_stream(copy_stream)
        original_init(self, model_runner_output, sampler_output,
                      num_sampled_tokens, main_stream, copy_stream,
                      check_ep_fault, routed_experts)

    @functools.wraps(original_get)
    def guarded_get(self):
        self.copy_event.synchronize()
        status = self._paiton_gdn_status.item()
        if status:
            raise RuntimeError(f'Paiton GDN norm nonfinite/arithmetic error: {status}')
        return original_get(self)

    AsyncOutput.__init__ = guarded_init
    AsyncOutput.get_output = guarded_get
    AsyncOutput._paiton_gdn_guard = True


def install():
    global _installed
    if _installed:
        return
    import torch
    import radiance_arnq as reference

    path = Path(os.environ['PAITON_RUNTIME_COMPAT_GDN_NORM_MANIFEST']).resolve(strict=True)
    manifest = json.loads(path.read_text())
    if (manifest.get('abi') != 1 or manifest.get('arch') != 'gfx1201'
            or manifest.get('arithmetic') != 'rms128_silu_fp32_native_bf16_round_row_e4m3fn'
            or frozenset(manifest.get('qualified_rows', ())) != SMALL_ROWS
            or manifest.get('threads') != 768):
        raise ValueError('Unqualified GDN norm runtime contract')
    artifact = manifest['artifact']
    binary = (path.parent / artifact['file']).resolve(strict=True)
    if binary.parent != path.parent or hashlib.sha256(binary.read_bytes()).hexdigest() != artifact['sha256']:
        raise ValueError('GDN norm artifact path/hash mismatch')
    library = ctypes.CDLL(str(binary), mode=ctypes.RTLD_LOCAL)
    library.paiton_gdn_norm_fp8_abi_version.restype = ctypes.c_int
    library.paiton_gdn_norm_fp8_arch.restype = ctypes.c_char_p
    if library.paiton_gdn_norm_fp8_abi_version() != 1 or library.paiton_gdn_norm_fp8_arch() != b'gfx1201':
        raise ValueError('GDN norm binary ABI mismatch')
    launch = library.paiton_gdn_norm_fp8
    launch.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int, ctypes.c_int64, ctypes.c_float, ctypes.c_void_p]
    launch.restype = ctypes.c_int
    selected = set()

    @torch.library.custom_op('paiton_runtime::gdn_norm_quant', mutates_args=('error',))
    def produce(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor,
                error: torch.Tensor, epsilon: float, last: bool) -> tuple[torch.Tensor, torch.Tensor]:
        m, n = x.shape
        eligible = (m in SMALL_ROWS and n == 6144 and z.shape == x.shape
                    and x.dtype == z.dtype == w.dtype == torch.bfloat16
                    and x.device == z.device == w.device == error.device
                    and x.is_contiguous() and w.is_contiguous() and w.shape == (128,)
                    and z.stride(1) == 1 and 6144 <= z.stride(0) <= 65536)
        if eligible:
            rounded = torch.empty_like(x)
            q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            scale = torch.empty(m, device=x.device, dtype=torch.float32)
            rc = launch(x.data_ptr(), z.data_ptr(), w.data_ptr(), rounded.data_ptr(),
                        q.data_ptr(), scale.data_ptr(), 0, error.data_ptr(), m,
                        z.stride(0), epsilon, torch.cuda.current_stream(x.device).cuda_stream)
            if rc:
                raise RuntimeError(f'Native GDN norm launch failed: {rc}')
            if m not in selected:
                selected.add(m)
                print(f'[paiton.runtime_gdn_norm] native rows={m}; threads=768', flush=True)
        else:
            q, scale = torch.ops.radiance.gdn_norm_quant(x, z, w, epsilon)
        return q, scale

    @produce.register_fake
    def fake(x, z, w, error, epsilon, last):
        return torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty(x.shape[0], device=x.device, dtype=torch.float32)

    def projection(self, core, z):
        m = core.shape[0]
        q, scale = produce(core.reshape(m, -1), z.reshape(m, -1), self.norm.weight,
                           self._paiton_gdn_norm_error, float(self.norm.eps), self._paiton_gdn_norm_last)
        output, _ = self.out_proj((q, scale))
        return output

    original_install = reference.install

    def bind(model):
        original_install(model)
        layers = [m for m in model.modules()
                  if getattr(getattr(m, '_output_projection', None), '__func__', None) is reference._gdn_output_projection]
        if not layers:
            return
        if len(layers) != 48:
            raise ValueError(f'Expected 48 external runtime GDN norm layers, got {len(layers)}')
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        if (not config.use_v2_model_runner
                or config.parallel_config.tensor_parallel_size != 1
                or config.parallel_config.pipeline_parallel_size != 1):
            raise ValueError('Native GDN requires the qualified V2 TP1/PP1 output path')
        error = torch.zeros(1, dtype=torch.int32, device=layers[0].norm.weight.device)
        _install_output_error_guard(error)
        for index, layer in enumerate(layers):
            if (reference._gnq_ok(layer) is not None
                    or layer.out_proj.input_size_per_partition != 6144):
                raise ValueError('Unexpected GDN projection contract')
            layer.register_buffer('_paiton_gdn_norm_error', error, persistent=False)
            layer._paiton_gdn_norm_last = index == len(layers) - 1
            layer._output_projection = types.MethodType(projection, layer)
        print('[paiton.runtime_gdn_norm] bound 48 layers; native M8/16/24/32, external runtime fallback otherwise', flush=True)

    reference.install = bind
    _installed = True
