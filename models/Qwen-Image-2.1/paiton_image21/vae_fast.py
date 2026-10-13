"""VAE decoder regions for single-frame decoding.

- RMS normalization + SiLU of every decoder residual block and of the output head through the native vae-norm companion
  (bf16 in/out, fp32 statistics in registers; the framework upcasts whole activations to fp32, which dominates the
  2048x2048 decode peak). Only the fp32 summation order of the channel reduction differs from the framework.
- Zero padding of the single-frame convolutions passed to the convolution instead of a padded copy of the input.
- Nearest-neighbour upsampling in bf16 (the framework round-trips through fp32; identical values).
Every region keeps the original module path for inputs outside its contract.
"""
import ctypes as C
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .native_regions import _load_library


class NormSilu(nn.Module):
    """norm (QwenImage21RMS_norm, channel first) followed by SiLU, in one native pass."""
    def __init__(self, norm, kernel, counts):
        super().__init__()
        self.norm = norm
        self.kernel = kernel
        self.counts = counts

    def forward(self, x):
        n = self.norm
        if (self.kernel is None or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous() or x.ndim != 5
                or x.shape[0] != 1 or x.shape[2] != 1 or (x.shape[3] * x.shape[4]) % 4 or x.data_ptr() % 8
                or not n.channel_first or not isinstance(n.bias, float) or n.bias != 0.0
                or n.gamma.dtype != torch.bfloat16 or not n.gamma.is_contiguous() or n.gamma.numel() != x.shape[1]):
            self.counts["fallback"] += 1
            return F.silu(n(x))
        out = torch.empty_like(x)
        rc = self.kernel(x.data_ptr(), n.gamma.data_ptr(), out.data_ptr(), x.shape[1], x.shape[3] * x.shape[4],
                         float(n.scale), 1e-12, 1, torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"Image21 VAE norm failed with HIP status {rc}")
        self.counts["norm_silu"] += 1
        return out


def _conv_forward_padded(self, x, cache_x=None):
    if cache_x is not None:
        raise ValueError("single-frame convolution takes no feature cache")
    w, _, h, _ = self._padding
    return F.conv2d(x.squeeze(2), self.weight, self.bias, self.stride, (h, w), self.dilation, self.groups).unsqueeze(2)


def _upsample_bf16(self, x):
    return nn.Upsample.forward(self, x)


def install(vae, norm_directory=None):
    """Patch the decoder of a single-frame AutoencoderKLQwenImage21 in place; returns the region counters."""
    from diffusers.models.autoencoders import autoencoder_kl_qwenimage21 as m
    counts = dict(norm_silu=0, fallback=0, convs=0, upsamplers=0, norm_library=None)
    kernel = None
    if norm_directory is not None:
        library, manifest = _load_library(norm_directory, 1, "PaitonImage21VaeNormGetAbiVersion",
                                          "PaitonImage21VaeNormGetTargetArch", "PaitonImage21VaeNormInitialize")
        kernel = library.PaitonImage21VaeRmsNorm
        kernel.argtypes = [C.c_void_p] * 3 + [C.c_int, C.c_long, C.c_float, C.c_float, C.c_int, C.c_void_p]
        kernel.restype = C.c_int
        counts["norm_library"] = manifest["sha256"]
        vae._paiton_vae_norm_library = library   # keep the loaded companion alive with the module
    decoder = vae.decoder
    if kernel is not None:
        silu = type(decoder.nonlinearity)
        for block in decoder.modules():
            if isinstance(block, m.QwenImage21ResidualBlock) and isinstance(block.nonlinearity, silu):
                block.norm1 = NormSilu(block.norm1, kernel, counts)
                block.norm2 = NormSilu(block.norm2, kernel, counts)
                block.nonlinearity = nn.Identity()
        decoder.norm_out = NormSilu(decoder.norm_out, kernel, counts)
        decoder.nonlinearity = nn.Identity()
    for module in [*decoder.modules(), vae.post_quant_conv]:
        if isinstance(module, m.QwenImage21CausalConv3d) and module.padding_mode == "zeros":
            module.forward = types.MethodType(_conv_forward_padded, module)
            counts["convs"] += 1
        elif isinstance(module, m.QwenImage21Upsample) and module.mode in ("nearest", "nearest-exact"):
            module.forward = types.MethodType(_upsample_bf16, module)
            counts["upsamplers"] += 1
    return counts
