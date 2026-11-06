"""ctypes binding of the native Idol denoiser (artifacts/libidol_denoiser_gfx1201.so).

The library reads the stock fp32 denoiser.safetensors itself (fp16 copies made at load) and runs the whole 50-step
guided sampler on the GPU. PyTorch is used here only to hand it device pointers and the current stream.
"""
import ctypes
import os

import torch
from tensordict import TensorDict

LIB = os.environ.get('PAITON_IDOL_LIB', os.path.join(os.path.dirname(__file__), '..', 'artifacts',
                                                     'libidol_denoiser_gfx1201.so'))
P = ctypes.c_void_p
_lib = ctypes.CDLL(LIB)
_lib.IdolCreate.restype = P
_lib.IdolCreate.argtypes = [ctypes.c_char_p]
_lib.IdolDestroy.argtypes = [P]
_lib.IdolSetCondition.argtypes = [P] * 4
_lib.IdolStepEx.argtypes = [P, P, P, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, P]


class NativeSampler:
    """Drop-in for idol.sampler.sample_flow. fast_from: first step that runs int8 QK + fp8 PV attention (None = all
    steps fp16, the exact path); cfg_until: steps from this index on run one conditional pass (None = guidance on every
    step, as stock)."""

    def __init__(self, denoiser_path, fast_from=None, cfg_until=None):
        self.h = _lib.IdolCreate(denoiser_path.encode())
        if not self.h:
            raise RuntimeError(f'native denoiser could not load {denoiser_path}')
        self.fast_from, self.cfg_until = fast_from, cfg_until

    def close(self):
        if self.h:
            _lib.IdolDestroy(self.h)
            self.h = None

    @torch.no_grad()
    def __call__(self, model, noise, cond, neg_cond, steps=50, rescale_t=3.0, cfg_strength=3.0, verbose=False,
                 batch_cfg=False):
        # neg_cond is always the all-zero condition in IdolPipeline; the library builds the unconditional branch itself
        s = torch.cuda.current_stream().cuda_stream
        f1, f2 = cond['feature1'][0].float().contiguous(), cond['feature2'][0].float().contiguous()
        if f1.shape != (4101, 1280) or f2.shape != (4101, 128):
            raise ValueError(f'unexpected condition shapes {tuple(f1.shape)} {tuple(f2.shape)}')
        if _lib.IdolSetCondition(self.h, f1.data_ptr(), f2.data_ptr(), s):
            raise RuntimeError('IdolSetCondition failed')
        z = noise['latent'][0].float().contiguous().clone()
        cam = noise['camera'][0, 0].float().contiguous().clone()
        fast = steps if self.fast_from is None else self.fast_from
        until = steps if self.cfg_until is None else self.cfg_until
        g = lambda t: rescale_t * t / (1 + (rescale_t - 1) * t)   # the stock sampler's shifted grid
        for i in range(steps):
            t, tp = g(1.0 - i / steps), g(1.0 - (i + 1) / steps)
            w = cfg_strength if i < until else 0.0
            if _lib.IdolStepEx(self.h, z.data_ptr(), cam.data_ptr(), 1000.0 * t, w, t - tp, 2 if i >= fast else 0, s):
                raise RuntimeError(f'IdolStepEx failed at step {i}')
        torch.cuda.current_stream().synchronize()   # f1/f2 must outlive the kernels that read them
        return TensorDict({'latent': z[None], 'camera': cam[None, None]}, batch_size=noise.batch_size, device=z.device)
