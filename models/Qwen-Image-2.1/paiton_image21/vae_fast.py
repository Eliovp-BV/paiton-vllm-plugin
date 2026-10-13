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


# channels-last convolution for the narrow, high-resolution layers (<= NHWC_MAX_CHANNELS channels in and out, at least
# NHWC_MIN_PIXELS output pixels), where the NCHW solver choice runs at a fraction of the matrix rate; 0 disables
NHWC_MIN_PIXELS = int(__import__("os").environ.get("PAITON_IMAGE21_VAE_NHWC_MIN_PIXELS", "0"))
NHWC_MAX_CHANNELS = 288


# native 3x3 convolution (vae-conv companion): "0" off, "narrow" for <= NATIVE_MAX_CHANNELS channels in and out at
# >= NATIVE_MIN_PIXELS output pixels, "all" for every qualifying convolution
NATIVE_CONV = __import__("os").environ.get("PAITON_IMAGE21_VAE_CONV", "0")
NATIVE_MAX_CHANNELS = 288
NATIVE_MIN_PIXELS = 1 << 20
_NATIVE = dict(kernel=None, counts=None)


def _native_conv(x, weight, bias, stride, padding, dilation, groups, module):
    kernel = _NATIVE["kernel"]
    if kernel is None or NATIVE_CONV == "0" or module is None:
        return None
    ci, h, w = x.shape[1], x.shape[2], x.shape[3]
    co = weight.shape[0]
    if (tuple(weight.shape[2:]) != (3, 3) or tuple(stride) != (1, 1) or tuple(padding) != (1, 1) or tuple(dilation) != (1, 1)
            or groups != 1 or x.shape[0] != 1 or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or ci % 16 or co % 48 or w % 64 or h % 4 or not x.is_cuda or x.data_ptr() % 16
            or (bias is not None and bias.dtype != torch.bfloat16)):
        return None
    if NATIVE_CONV == "narrow" and (ci > NATIVE_MAX_CHANNELS or co > NATIVE_MAX_CHANNELS or h * w < NATIVE_MIN_PIXELS):
        return None
    arranged = module.__dict__.get("_paiton_wt")
    if arranged is None or arranged.shape != (3, 3, co, ci):
        arranged = weight.permute(2, 3, 0, 1).contiguous()   # [ky][kx][co][ci], once per module
        module.__dict__["_paiton_wt"] = arranged
    x = x.contiguous()
    out = torch.empty((1, co, h, w), dtype=torch.bfloat16, device=x.device)
    rc = kernel(x.data_ptr(), arranged.data_ptr(), 0 if bias is None else bias.contiguous().data_ptr(), out.data_ptr(), ci, co, h, w,
                torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"Image21 VAE convolution failed with HIP status {rc}")
    _NATIVE["counts"]["native_conv"] += 1
    return out


def _conv2d(x, weight, bias, stride, padding, dilation, groups, module=None):
    out = _native_conv(x, weight, bias, stride, padding, dilation, groups, module)
    if out is not None:
        return out
    if (NHWC_MIN_PIXELS and x.shape[1] <= NHWC_MAX_CHANNELS and weight.shape[0] <= NHWC_MAX_CHANNELS
            and x.shape[-1] * x.shape[-2] >= NHWC_MIN_PIXELS):
        y = F.conv2d(x.contiguous(memory_format=torch.channels_last), weight.contiguous(memory_format=torch.channels_last),
                     bias, stride, padding, dilation, groups)
        return y.contiguous()
    return F.conv2d(x, weight, bias, stride, padding, dilation, groups)


def _conv_forward_padded(self, x, cache_x=None):
    if cache_x is not None:
        raise ValueError("single-frame convolution takes no feature cache")
    w, _, h, _ = self._padding
    return _conv2d(x.squeeze(2), self.weight, self.bias, self.stride, (h, w), self.dilation, self.groups, self).unsqueeze(2)


def _conv2d_forward(self, x):
    return _conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self)


def _upsample_bf16(self, x):
    return nn.Upsample.forward(self, x)


def install(vae, norm_directory=None, conv_directory=None):
    """Patch the decoder of a single-frame AutoencoderKLQwenImage21 in place; returns the region counters."""
    from diffusers.models.autoencoders import autoencoder_kl_qwenimage21 as m
    counts = dict(norm_silu=0, fallback=0, convs=0, upsamplers=0, native_conv=0, norm_library=None, conv_library=None,
                  native_conv_mode=NATIVE_CONV)
    _NATIVE["counts"] = counts
    if conv_directory is not None and NATIVE_CONV != "0":
        conv_library, conv_manifest = _load_library(conv_directory, 1, "PaitonImage21VaeConvGetAbiVersion",
                                                    "PaitonImage21VaeConvGetTargetArch", "PaitonImage21VaeConvInitialize")
        conv = conv_library.PaitonImage21VaeConv3x3
        conv.argtypes = [C.c_void_p] * 4 + [C.c_int] * 4 + [C.c_void_p]
        conv.restype = C.c_int
        _NATIVE["kernel"] = conv
        counts["conv_library"] = conv_manifest["sha256"]
        vae._paiton_vae_conv_library = conv_library
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
        elif type(module) is nn.Conv2d and module.padding_mode == "zeros" and isinstance(module.padding, tuple):
            module.forward = types.MethodType(_conv2d_forward, module)   # the upsamplers' convolutions
            counts["convs"] += 1
        elif isinstance(module, m.QwenImage21Upsample) and module.mode in ("nearest", "nearest-exact"):
            module.forward = types.MethodType(_upsample_bf16, module)
            counts["upsamplers"] += 1
    return counts
