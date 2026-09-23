"""Candidate low-precision GEMM path for the seven image-stream projections of every transformer block (NOT
qualified; enabled only by an explicit flag). Formats: 'fp8-row' (fp8 e4m3fn activations with one scale per row),
'int8-row', 'int8-32', 'int8-64', 'int8-128' and 'int8-256' (int8 activations, one scale per row or per K block). The MXFP4 weights are
reconstructed by native kernels into fp8 codes (exact: every value representable, verified by a counter) or int8
codes (exact for channels whose block exponents span <= 3 binades, the rest rounded and counted); the activations are
quantized by native kernels (optionally fused with the gated SiLU product); the GEMM is the native wave32 or wave64
WMMA kernel with fp32 accumulation and the scales and bias in its epilogue. The path can be kept exact for the cached
text-prefix pass and for the first forwards of a request (precision schedule). No framework arithmetic is involved.
"""
import ctypes as C

import torch

from .native_regions import _load_library, HIDDEN

FORMATS = {'fp8-row': (False, 0), 'int8-row': (True, 0), 'int8-32': (True, 32), 'int8-64': (True, 64), 'int8-128': (True, 128), 'int8-256': (True, 256)}


class Fp8Regions:
    def __init__(self, gemm_directory, quantize_directory, unpack_directory, format='fp8-row', gemm_int8_directory=None,
                 gemm_w64_directory=None, wave64=False, from_step=0, exact_prefix=True):
        if format not in FORMATS:
            raise ValueError(f'unknown low-precision format {format}')
        self.format = format
        self.int8, self.block = FORMATS[format]
        self.wave64 = bool(wave64)
        self.from_step = int(from_step)
        self.exact_prefix = bool(exact_prefix)
        self.gemm_manifest = self.gemm_int8_manifest = self.gemm_w64_manifest = None
        if self.wave64:
            self.gemm_w64_library, self.gemm_w64_manifest = _load_library(
                gemm_w64_directory, 1, 'PaitonImage21GemmW64GetAbiVersion', 'PaitonImage21GemmW64GetTargetArch', 'PaitonImage21GemmW64Initialize')
            if self.int8:
                self.gemm_kernel = self.gemm_w64_library.PaitonImage21GemmInt8W64
                self.gemm_kernel.argtypes = [C.c_void_p]*6 + [C.c_int]*6 + [C.c_void_p]
            else:
                self.gemm_kernel = self.gemm_w64_library.PaitonImage21GemmFp8W64
                self.gemm_kernel.argtypes = [C.c_void_p]*6 + [C.c_int]*5 + [C.c_void_p]
        elif self.int8:
            self.gemm_int8_library, self.gemm_int8_manifest = _load_library(
                gemm_int8_directory, 1, 'PaitonImage21GemmInt8GetAbiVersion', 'PaitonImage21GemmInt8GetTargetArch', 'PaitonImage21GemmInt8Initialize')
            self.gemm_kernel = self.gemm_int8_library.PaitonImage21GemmInt8
            self.gemm_kernel.argtypes = [C.c_void_p]*6 + [C.c_int]*6 + [C.c_void_p]
        else:
            self.gemm_library, self.gemm_manifest = _load_library(
                gemm_directory, 1, 'PaitonImage21GemmFp8GetAbiVersion', 'PaitonImage21GemmFp8GetTargetArch', 'PaitonImage21GemmFp8Initialize')
            self.gemm_kernel = self.gemm_library.PaitonImage21GemmFp8
            self.gemm_kernel.argtypes = [C.c_void_p]*6 + [C.c_int]*5 + [C.c_void_p]
        self.gemm_kernel.restype = C.c_int
        self.quantize_library, self.quantize_manifest = _load_library(
            quantize_directory, 1, 'PaitonImage21QuantizeGetAbiVersion', 'PaitonImage21QuantizeGetTargetArch', 'PaitonImage21QuantizeInitialize')
        self.quantize_kernel = self.quantize_library.PaitonImage21QuantizeRowwise
        self.quantize_kernel.argtypes = [C.c_void_p]*4 + [C.c_int]*4 + [C.c_void_p]
        self.quantize_kernel.restype = C.c_int
        self.unpack_library, self.unpack_manifest = _load_library(
            unpack_directory, 1, 'PaitonMXFP4Fp8GetAbiVersion', 'PaitonMXFP4Fp8GetTargetArch', 'PaitonMXFP4Fp8Initialize')
        self.unpack_kernel = self.unpack_library.paiton_mxfp4_unpack_int8 if self.int8 else self.unpack_library.paiton_mxfp4_unpack_fp8
        self.unpack_kernel.argtypes = [C.c_void_p]*4 + [C.c_int]*2 + [C.c_void_p]*2
        self.unpack_kernel.restype = C.c_int
        self.inexact = torch.zeros((), dtype=torch.int32, device='cuda')
        self.variant = 0
        self.group_m = 4
        self.counts = dict(quantize=0, gemm=0, unpack=0, exact_forwards=0, low_precision_forwards=0)
        self.forwards = 0        # transformer forwards of the current request (reset by the runtime per request)
        self.schedule_active = False   # set per transformer forward: past the exact prefix of the schedule
        self.active = False      # set per block forward: low precision for this forward or the exact framework path

    def call(self, function, *args):
        rc = function(*args, torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f'Image21 low-precision region failed with HIP status {rc}')

    def begin_transformer_forward(self):
        """Called once per transformer forward (runtime pre-hook): advance the precision schedule."""
        self.forwards += 1
        self.schedule_active = self.forwards > self.from_step

    def begin_forward(self, kv_cache_mode):
        """Called per block forward: low precision only past the schedule's exact prefix and never on the cached
        text-prefix pass (kv_cache_mode 'extract') when exact_prefix is set."""
        self.active = self.schedule_active and not (self.exact_prefix and kv_cache_mode == 'extract')
        self.counts['low_precision_forwards' if self.active else 'exact_forwards'] += 1
        return self.active

    def weight_fp8(self, module):
        """Reconstruction of a packed MXFP4 linear weight into fp8 (exact) or int8 codes and one 2^k scale per row."""
        packed, scale = module._packed_weight, module._packed_scale
        rows, padded = packed.shape[0], packed.shape[1]*2
        if padded != module._original_weight_shape[1] or not packed.is_contiguous() or not scale.is_contiguous():
            raise RuntimeError('Image21 low-precision path requires unpadded contiguous packed weights')
        codes = torch.empty((rows, padded), dtype=torch.uint8, device=packed.device)
        row_scale = torch.empty(rows, dtype=torch.float32, device=packed.device)
        self.call(self.unpack_kernel, packed.data_ptr(), scale.data_ptr(), codes.data_ptr(), row_scale.data_ptr(),
                  rows, padded, self.inexact.data_ptr())
        self.counts['unpack'] += 1
        return codes, row_scale

    def quantize(self, x, y=None, activation=0, int8=None, block=None):
        """Codes and fp32 scales of x per row (block 0) or per (row, block of K columns); activation 1: GELU(tanh) of
        x; 2: silu(x) * y. int8/block default to the configured format."""
        int8 = self.int8 if int8 is None else int8
        block = self.block if block is None else block
        cols = x.shape[-1]
        x2 = x.reshape(-1, cols)
        if not x2.is_contiguous() or x2.dtype != torch.bfloat16 or (y is not None and (y.shape != x.shape or not y.is_contiguous() or y.dtype != torch.bfloat16)):
            raise RuntimeError('Image21 low-precision quantizer requires contiguous bf16 inputs')
        rows = x2.shape[0]
        codes = torch.empty((rows, cols), dtype=torch.uint8, device=x.device)
        scale = torch.empty((rows, cols // block) if block else (rows,), dtype=torch.float32, device=x.device)
        self.call(self.quantize_kernel, x2.data_ptr(), y.data_ptr() if y is not None else 0, codes.data_ptr(), scale.data_ptr(),
                  rows, cols, (activation << 1) | int(int8), block)
        self.counts['quantize'] += 1
        return codes, scale

    def linear(self, module, codes, scale):
        """bf16 [rows][N] = dequant(codes, scale) @ W^T + bias with the reconstructed low-precision weights."""
        weight, weight_scale = self.weight_fp8(module)
        rows, k = codes.shape
        n = weight.shape[0]
        if weight.shape[1] != k:
            raise RuntimeError('Image21 low-precision linear: input width mismatch')
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=codes.device)
        bias = module.bias
        if bias is not None and (bias.dtype != torch.bfloat16 or not bias.is_contiguous()):
            raise RuntimeError('Image21 low-precision linear: bias must be contiguous bf16')
        bias_ptr = bias.data_ptr() if bias is not None else 0
        if self.int8:
            self.call(self.gemm_kernel, codes.data_ptr(), scale.data_ptr(), weight.data_ptr(), weight_scale.data_ptr(), bias_ptr, out.data_ptr(),
                      rows, n, k, self.block, self.variant, self.group_m)
        else:
            self.call(self.gemm_kernel, codes.data_ptr(), scale.data_ptr(), weight.data_ptr(), weight_scale.data_ptr(), bias_ptr, out.data_ptr(),
                      rows, n, k, self.variant, self.group_m)
        self.counts['gemm'] += 1
        return out

    def check_exact(self):
        """Weight values that could not be reconstructed exactly since the last check (fp8: must be 0)."""
        value = int(self.inexact.item())
        self.inexact.zero_()
        return value

    def receipt(self):
        manifests = dict(quantize_sha256=self.quantize_manifest['sha256'], unpack_sha256=self.unpack_manifest['sha256'])
        for name, manifest in (('gemm', self.gemm_manifest), ('gemm_int8', self.gemm_int8_manifest), ('gemm_w64', self.gemm_w64_manifest)):
            if manifest is not None:
                manifests[f'{name}_sha256'] = manifest['sha256']
        return dict(format=self.format, wave64=self.wave64, from_step=self.from_step, exact_prefix=self.exact_prefix,
                    variant=self.variant, group_m=self.group_m, counts=dict(self.counts), qualification='precision schedule; graded profiles in measurements/low-precision-r9700.json', **manifests)
