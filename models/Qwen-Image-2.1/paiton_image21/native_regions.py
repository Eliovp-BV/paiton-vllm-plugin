"""External framework adapter for qualified BF16 regions; no compiler dependency.

Attention masks/cache sequence follows pinned Diffusers QwenImage21AttnProcessor
(Apache-2.0). GEMMs, checkpoint arithmetic and scheduler remain the existing
implementations. Every intermediate BF16 rounding is explicit in the independently
built native libraries. Two optional companions extend the qualified region: an exact
native attention (dense text-to-image and cached decode steps, no masks) and a fused
LayerNorm(+gated residual)+modulation; when either artifact is absent, the original
framework path is used unchanged.
"""
import ctypes as C
import hashlib
import json
import inspect
import math
from pathlib import Path
import types

import torch
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.normalization import RMSNorm


HEADS, HEAD_DIM, HIDDEN = 32, 128, 4096
MAX_TOKENS = 40000
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)   # framework default scale for head_dim 128 (rounded to float at the boundary)


def _file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_library(directory, abi_version, abi_symbol, arch_symbol, init_symbol):
    """Validate an artifact directory (manifest identity, architecture, ABI) before dlopen."""
    directory = Path(directory)
    meta = json.loads((directory/'manifest.json').read_text())
    binary = directory/meta['file']
    if (meta['architecture'] != 'gfx1201' or meta['abi_version'] != abi_version
            or _file_sha256(binary) != meta['sha256']):
        raise RuntimeError('Image21 fusion artifact identity mismatch')
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx1201'):
        raise RuntimeError('Image21 native fusions require gfx1201')
    library = C.CDLL(str(binary.resolve()))
    abi = getattr(library, abi_symbol)
    abi.restype = C.c_uint
    if abi() != abi_version:
        raise RuntimeError('Image21 fusion ABI mismatch')
    arch = getattr(library, arch_symbol)
    arch.restype = C.c_char_p
    if arch() != b"gfx1201":
        raise RuntimeError("Image21 fusion target mismatch")
    init = getattr(library, init_symbol)
    init.restype = C.c_int
    if init():
        raise RuntimeError('Image21 fusion initialization failed')
    return library, meta


class NativeRegions:
    def __init__(self, directory, attention_directory=None, normfuse_directory=None, qk8_directory=None):
        self.fp8 = None   # candidate fp8 GEMM path (Fp8Regions), attached by install() only on explicit opt-in
        self.pending_quantized = None   # (codes, scale) of the fused LayerNorm quantizer for the next attention projection
        self.norm_modulate_quant_kernel = None
        self.qk8_library = self.qk8_manifest = None
        if qk8_directory is not None:   # candidate int8-QK^T attention, only for the low-precision forwards of the schedule
            self.qk8_library, self.qk8_manifest = _load_library(
                qk8_directory, 3, 'PaitonImage21AttentionQk8GetAbiVersion', 'PaitonImage21AttentionQk8GetTargetArch', 'PaitonImage21AttentionQk8Initialize')
            self.qk8_pack = self.qk8_library.PaitonImage21AttentionPackKV
            self.qk8_pack.argtypes = [C.c_void_p]*5 + [C.c_int]*3 + [C.c_long]*4 + [C.c_void_p]
            self.qk8_pack.restype = C.c_int
            self.qk8_kernel = self.qk8_library.PaitonImage21AttentionQk8
            self.qk8_kernel.argtypes = [C.c_void_p]*3 + [C.c_int]*4 + [C.c_long]*4 + [C.c_float, C.c_void_p]
            self.qk8_kernel.restype = C.c_int
            self.qk8_elements = self.qk8_library.PaitonImage21AttentionQk8Elements
            self.qk8_elements.argtypes = [C.c_int]*3
            self.qk8_elements.restype = C.c_size_t
            self._qk8_workspace = None
        self.library, meta = _load_library(directory, 1, 'PaitonImage21GetAbiVersion',
                                           'PaitonImage21GetTargetArch', 'PaitonImage21Initialize')
        self.norm = self.library.PaitonImage21NormRope
        self.norm.argtypes = [C.c_void_p]*4+[C.c_int,C.c_float,C.c_void_p]
        self.prepare = self.library.PaitonImage21PrepareModulation
        self.prepare.argtypes = [C.c_void_p]*2+[C.c_int,C.c_void_p]
        self.modulate = self.library.PaitonImage21Modulate
        self.modulate.argtypes = [C.c_void_p]*5+[C.c_int]*3+[C.c_void_p]
        self.silu = self.library.PaitonImage21SiluMul
        self.silu.argtypes = [C.c_void_p]*3+[C.c_size_t,C.c_void_p]
        self.pack = self.library.PaitonImage21PackAttention
        self.pack.argtypes = [C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]
        for function in (self.norm,self.prepare,self.modulate,self.silu,self.pack):
            function.restype = C.c_int
        self.counts = dict(blocks=0,norm_rope=0,prepare=0,modulation=0,residual=0,silu=0,attention_pack=0,fallback=0,
                           attention_native=0,attention_mask_fallback=0,norm_modulate=0,residual_norm_modulate=0)
        self.manifest = meta
        # optional exact attention companion
        self.attention_library = None
        self.attention_manifest = None
        self._workspace = None
        self._mask_checks = {}
        if attention_directory is not None:
            self.attention_library, self.attention_manifest = _load_library(
                attention_directory, 2, 'PaitonImage21AttentionGetAbiVersion',
                'PaitonImage21AttentionGetTargetArch', 'PaitonImage21AttentionInitialize')
            self.attention_pack_v = self.attention_library.PaitonImage21AttentionPackV
            self.attention_pack_v.argtypes = [C.c_void_p]*3+[C.c_int]*3+[C.c_long]*4+[C.c_void_p]
            self.attention_pack_v.restype = C.c_int
            self.attention_kernel = self.attention_library.PaitonImage21Attention
            self.attention_kernel.argtypes = [C.c_void_p]*5+[C.c_int]*4+[C.c_long]*8+[C.c_float,C.c_void_p,C.c_int]
            self.attention_kernel.restype = C.c_int
            self.attention_workspace_elements = self.attention_library.PaitonImage21AttentionVtElements
            self.attention_workspace_elements.argtypes = [C.c_int]*3
            self.attention_workspace_elements.restype = C.c_size_t
        # optional fused LayerNorm(+residual)+modulation companion
        self.normfuse_library = None
        self.normfuse_manifest = None
        if normfuse_directory is not None:
            self.normfuse_library, self.normfuse_manifest = _load_library(
                normfuse_directory, 1, 'PaitonImage21NormFuseGetAbiVersion',
                'PaitonImage21NormFuseGetTargetArch', 'PaitonImage21NormModulateInitialize')
            self.norm_modulate_kernel = self.normfuse_library.PaitonImage21NormModulate
            self.norm_modulate_quant_kernel = getattr(self.normfuse_library, 'PaitonImage21NormModulateQuant', None)
            if self.norm_modulate_quant_kernel is not None:
                self.norm_modulate_quant_kernel.argtypes = [C.c_void_p]*7 + [C.c_int]*3 + [C.c_float, C.c_int, C.c_void_p]
                self.norm_modulate_quant_kernel.restype = C.c_int
            self.norm_modulate_kernel.argtypes = [C.c_void_p]*6+[C.c_int]*3+[C.c_float,C.c_void_p]
            self.norm_modulate_kernel.restype = C.c_int

    def call(self, function, *args):
        rc = function(*args, torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f'Image21 native region failed with HIP status {rc}')

    def norm_rope(self, x, norm, frequencies):
        out = torch.empty_like(x)
        self.call(self.norm,x.data_ptr(),norm.weight.data_ptr(),
                  frequencies.data_ptr() if frequencies is not None else 0,
                  out.data_ptr(),x.shape[1],norm.eps)
        self.counts['norm_rope'] += 1
        return out

    def parameters(self, modulation):
        out = torch.empty_like(modulation)
        self.call(self.prepare,modulation.data_ptr(),out.data_ptr(),HIDDEN)
        self.counts['prepare'] += 1
        return out

    def pointwise(self, x, parameters, mask, offset, branch=None):
        out = torch.empty_like(x)
        self.call(self.modulate,x.data_ptr(),0 if branch is None else branch.data_ptr(),
                  parameters.data_ptr(),0 if mask is None else mask.data_ptr(),out.data_ptr(),
                  x.shape[1],HIDDEN,offset)
        self.counts['modulation' if branch is None else 'residual'] += 1
        return out

    def norm_modulate(self, x, parameters, mask, scale_offset, eps):
        """modulated = bf16(LayerNorm(x)) * params[scale_offset] (fused, exact)."""
        out = torch.empty_like(x)
        self.call(self.norm_modulate_kernel,x.data_ptr(),0,parameters.data_ptr(),
                  0 if mask is None else mask.data_ptr(),0,out.data_ptr(),x.shape[1],0,scale_offset,eps)
        self.counts['norm_modulate'] += 1
        return out

    def residual_norm_modulate(self, x, branch, parameters, mask, gate_offset, scale_offset, eps):
        """hidden = bf16(x + bf16(gate*branch)); modulated = bf16(LayerNorm(hidden)) * params[scale_offset]."""
        hidden = torch.empty_like(x)
        out = torch.empty_like(x)
        self.call(self.norm_modulate_kernel,x.data_ptr(),branch.data_ptr(),parameters.data_ptr(),
                  0 if mask is None else mask.data_ptr(),hidden.data_ptr(),out.data_ptr(),
                  x.shape[1],gate_offset,scale_offset,eps)
        self.counts['residual_norm_modulate'] += 1
        return hidden, out

    def fused_quant_available(self):
        fp8 = getattr(self, 'fp8', None)
        return (fp8 is not None and fp8.active and fp8.int8 and fp8.block == 256
                and getattr(self, 'norm_modulate_quant_kernel', None) is not None)

    def norm_modulate_quant(self, x, parameters, mask, scale_offset, eps):
        """LayerNorm + modulation emitted as int8 codes with per-256-block scales (bit-identical to quantizing the bf16
        output); returns (placeholder, codes, scale) — the placeholder keeps the attention processor's tensor API."""
        rows = x.shape[1]
        codes = torch.empty((rows, HIDDEN), dtype=torch.uint8, device=x.device)
        scale = torch.empty((rows, HIDDEN // 256), dtype=torch.float32, device=x.device)
        self.call(self.norm_modulate_quant_kernel, x.data_ptr(), 0, parameters.data_ptr(), 0 if mask is None else mask.data_ptr(),
                  0, codes.data_ptr(), scale.data_ptr(), rows, 0, scale_offset, eps, 256)
        self.counts['norm_modulate_quant'] = self.counts.get('norm_modulate_quant', 0) + 1
        return x, codes, scale

    def residual_norm_modulate_quant(self, x, branch, parameters, mask, gate_offset, scale_offset, eps):
        hidden = torch.empty_like(x)
        rows = x.shape[1]
        codes = torch.empty((rows, HIDDEN), dtype=torch.uint8, device=x.device)
        scale = torch.empty((rows, HIDDEN // 256), dtype=torch.float32, device=x.device)
        self.call(self.norm_modulate_quant_kernel, x.data_ptr(), branch.data_ptr(), parameters.data_ptr(), 0 if mask is None else mask.data_ptr(),
                  hidden.data_ptr(), codes.data_ptr(), scale.data_ptr(), rows, gate_offset, scale_offset, eps, 256)
        self.counts['residual_norm_modulate_quant'] = self.counts.get('residual_norm_modulate_quant', 0) + 1
        return hidden, codes, scale

    def feed_forward(self, mlp, x, quantized=None):
        fp8 = getattr(self, 'fp8', None)
        if fp8 is not None and fp8.active:
            codes, scale = quantized if quantized is not None else self.fp8.quantize(x)
            gate = self.fp8.linear(mlp.gate_layer, codes, scale)
            up = self.fp8.linear(mlp.proj, codes, scale)
            del codes, scale
            product, product_scale = self.fp8.quantize(gate, up, activation=2)
            del gate, up
            self.counts['silu'] += 1
            return self.fp8.linear(mlp.out, product, product_scale).view(1, -1, HIDDEN)
        gate = mlp.gate_layer(x)
        up = mlp.proj(x)
        product = torch.empty_like(gate)
        self.call(self.silu,gate.data_ptr(),up.data_ptr(),product.data_ptr(),gate.numel())
        self.counts['silu'] += 1
        del gate, up
        return mlp.out(product)

    def attention(self, query, key, value, **kwargs):
        inputs=[]
        for x in (query,key,value):
            if (x.dtype!=torch.bfloat16 or not x.is_contiguous() or x.shape[0]!=1
                    or tuple(x.shape[2:])!=(HEADS,HEAD_DIM) or not 0<x.shape[1]<=MAX_TOKENS):
                self.counts['fallback']+=1
                return dispatch_attention_fn(query,key,value,**kwargs)
        for x in (query,key,value):
            packed=torch.empty((1,HEADS,x.shape[1],HEAD_DIM),dtype=x.dtype,device=x.device)
            self.call(self.pack,x.data_ptr(),packed.data_ptr(),x.shape[1])
            inputs.append(packed.permute(0,2,1,3))
            self.counts['attention_pack']+=1
        return dispatch_attention_fn(*inputs,**kwargs)

    def attention_qk8(self, query, key, value, key_prefix=None, value_prefix=None):
        """Approximate attention with per-row int8 Q and K (int32 scores), bf16 softmax/PV as the exact kernel."""
        tokens_q, tokens_k = query.shape[1], key.shape[1]
        prefix = 0 if key_prefix is None else key_prefix.shape[1]
        needed = self.qk8_elements(tokens_k, prefix, HEADS)
        if self._qk8_workspace is None or self._qk8_workspace.numel() < needed or self._qk8_workspace.device != query.device:
            self._qk8_workspace = torch.empty(needed, dtype=torch.bfloat16, device=query.device)
        out = torch.empty_like(query)
        tok, head = HEADS*HEAD_DIM, HEAD_DIM
        self.call(self.qk8_pack, key.data_ptr(), 0 if key_prefix is None else key_prefix.data_ptr(), value.data_ptr(),
                  0 if value_prefix is None else value_prefix.data_ptr(), self._qk8_workspace.data_ptr(), tokens_k, prefix, HEADS, tok, head, tok, head)
        rc = self.qk8_kernel(query.data_ptr(), self._qk8_workspace.data_ptr(), out.data_ptr(), tokens_q, tokens_k, prefix, HEADS,
                             tok, head, tok, head, SM_SCALE, torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f'Image21 int8-QK attention failed with HIP status {rc}')
        self.counts['attention_qk8'] = self.counts.get('attention_qk8', 0) + 1
        return out

    # ---- exact native attention ------------------------------------------------------------------
    @staticmethod
    def _dense_input(x, min_tokens=1):
        return (x is not None and x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous()
                and x.ndim == 4 and x.shape[0] == 1 and tuple(x.shape[2:]) == (HEADS, HEAD_DIM)
                and min_tokens <= x.shape[1] <= MAX_TOKENS)

    def _mask_is_trivial(self, mask):
        """A boolean key-validity mask that is entirely true does not change the reference result
        (verified bitwise against the pinned framework); the check is cached per mask storage."""
        if mask is None:
            return True
        if mask.dtype != torch.bool or mask.ndim != 4 or mask.shape[0] != 1 or mask.shape[1] != 1 or mask.shape[2] != 1:
            return False
        key = (mask.data_ptr(), tuple(mask.shape), mask._version)
        result = self._mask_checks.get(key)
        if result is None:
            result = bool(mask.all().item())
            if len(self._mask_checks) > 64:
                self._mask_checks.clear()
            self._mask_checks[key] = result
        return result

    def attention_native_usable(self, query, key, value, key_prefix, value_prefix, attn_mask):
        if self.attention_library is None:
            return False
        if not (self._dense_input(query) and self._dense_input(key) and self._dense_input(value)):
            return False
        if key.shape[1] != value.shape[1]:
            return False
        if (key_prefix is None) != (value_prefix is None):
            return False
        if key_prefix is not None:
            if not (self._dense_input(key_prefix) and self._dense_input(value_prefix)):
                return False
            if key_prefix.shape[1] != value_prefix.shape[1] or key_prefix.shape[1] + key.shape[1] > MAX_TOKENS:
                return False
        if not self._mask_is_trivial(attn_mask):
            self.counts['attention_mask_fallback'] += 1
            return False
        return True

    def attention_native(self, query, key, value, key_prefix=None, value_prefix=None):
        """Dense BF16 attention over [prefix keys | keys] for token-major [1, S, 32, 128] inputs.

        Reads the cached prefix and the new keys/values in place (no concatenation, no relayout),
        packs the transposed values into a reusable caller-owned workspace and writes the
        token-major output directly."""
        tokens_q, tokens_k = query.shape[1], key.shape[1]
        prefix = 0 if key_prefix is None else key_prefix.shape[1]
        needed = self.attention_workspace_elements(tokens_k, prefix, HEADS)
        if self._workspace is None or self._workspace.numel() < needed or self._workspace.device != query.device:
            self._workspace = torch.empty(needed, dtype=torch.bfloat16, device=query.device)
        out = torch.empty_like(query)
        tok, head = HEADS*HEAD_DIM, HEAD_DIM
        self.call(self.attention_pack_v, value.data_ptr(), 0 if value_prefix is None else value_prefix.data_ptr(),
                  self._workspace.data_ptr(), tokens_k, prefix, HEADS, tok, head, tok, head)
        # the attention entry point takes (…, sm_scale, stream, variant): the stream is not its last argument
        rc = self.attention_kernel(query.data_ptr(), key.data_ptr(), 0 if key_prefix is None else key_prefix.data_ptr(),
                                   self._workspace.data_ptr(), out.data_ptr(), tokens_q, tokens_k, prefix, HEADS,
                                   tok, head, tok, head, tok, head, tok, head, SM_SCALE,
                                   torch.cuda.current_stream().cuda_stream, 0)
        if rc:
            raise RuntimeError(f'Image21 native attention failed with HIP status {rc}')
        self.counts['attention_native'] += 1
        return out


def valid_hidden(x):
    return (not torch.is_grad_enabled() and x.is_cuda and x.dtype == torch.bfloat16
            and x.ndim == 3 and x.shape[0] == 1 and x.shape[2] == HIDDEN
            and 0 < x.shape[1] <= MAX_TOKENS and x.is_contiguous())


class NativeAttentionProcessor(QwenImage21AttnProcessor):
    def __init__(self, native, fallback):
        self.native = native
        self.fallback = fallback
        self._attention_backend = fallback._attention_backend
        self._parallel_config = fallback._parallel_config

    def __call__(self,attn,hidden_states,attention_mask=None,rotary_emb=None,
                 layer_cache=None,kv_cache_mode=None,cache_write_slice=None,
                 segments=None,key_valid=None):
        valid_rope = (rotary_emb is None or (rotary_emb.is_contiguous()
                      and rotary_emb.dtype == torch.complex64
                      and tuple(rotary_emb.shape) == (hidden_states.shape[1],64)))
        if not valid_hidden(hidden_states) or not valid_rope:
            self.native.counts['fallback'] += 1
            return self.fallback(attn,hidden_states,attention_mask,rotary_emb,
                                 layer_cache,kv_cache_mode,cache_write_slice,segments,key_valid)
        fp8 = getattr(self.native, 'fp8', None)
        if fp8 is not None and not fp8.active:
            fp8 = None
        if fp8 is not None:
            pending = getattr(self.native, 'pending_quantized', None)
            self.native.pending_quantized = None
            codes, scale = pending if pending is not None else fp8.quantize(hidden_states)
            query = fp8.linear(attn.to_q, codes, scale).view(1,-1,HEADS,HEAD_DIM)
            key = fp8.linear(attn.to_k, codes, scale).view(1,-1,HEADS,HEAD_DIM)
            value = fp8.linear(attn.to_v, codes, scale).view(1,-1,HEADS,HEAD_DIM)
            del codes, scale
        else:
            query = attn.to_q(hidden_states).unflatten(-1,(HEADS,HEAD_DIM))
            key = attn.to_k(hidden_states).unflatten(-1,(HEADS,HEAD_DIM))
            value = attn.to_v(hidden_states).unflatten(-1,(HEADS,HEAD_DIM))
        query = self.native.norm_rope(query,attn.norm_q,rotary_emb)
        key = self.native.norm_rope(key,attn.norm_k,rotary_emb)
        cached_k = cached_v = None
        if layer_cache is not None:
            if kv_cache_mode == 'extract' and cache_write_slice is not None:
                layer_cache.store(key[:,cache_write_slice].clone(),value[:,cache_write_slice].clone())
            elif kv_cache_mode == 'cached':
                cached_k,cached_v = layer_cache.get()
        if segments is None and self.native.attention_native_usable(query,key,value,cached_k,cached_v,attention_mask):
            if getattr(self.native, 'qk8_library', None) is not None and fp8 is not None:   # fp8 is None unless this forward is low precision
                result = self.native.attention_qk8(query,key,value,cached_k,cached_v)
            else:
                result = self.native.attention_native(query,key,value,cached_k,cached_v)
        else:
            if cached_k is not None:
                key = torch.cat([cached_k,key],dim=1)
                value = torch.cat([cached_v,value],dim=1)
            if segments is None:
                result = self.native.attention(query,key,value,attn_mask=attention_mask,
                                               dropout_p=0.0,backend=self._attention_backend,
                                               parallel_config=self._parallel_config)
            else:
                outputs = []
                for start,end,is_text in segments:
                    mask = None
                    if is_text:
                        length = end-start
                        mask = torch.cat([torch.ones(length,start,dtype=torch.bool,device=query.device),
                                          torch.tril(torch.ones(length,length,dtype=torch.bool,device=query.device))],dim=1)[None,None]
                    if key_valid is not None:
                        valid = key_valid[:,None,None,:end]
                        mask = valid if mask is None else mask & valid
                    outputs.append(self.native.attention(query[:,start:end],key[:,:end],value[:,:end],
                        attn_mask=mask,dropout_p=0.0,backend=None,parallel_config=self._parallel_config))
                prefix = segments[-1][1] if segments else 0
                outputs.append(self.native.attention(query[:,prefix:],key,value,
                    attn_mask=None if key_valid is None else key_valid[:,None,None,:],
                    dropout_p=0.0,backend=None,parallel_config=self._parallel_config))
                result = torch.cat(outputs,dim=1)
        result = result[:,:query.shape[1]].flatten(2,3).type_as(query)
        if fp8 is not None:
            codes, scale = fp8.quantize(result.contiguous())
            return attn.to_out[1](fp8.linear(attn.to_out[0], codes, scale).view(1,-1,HIDDEN))
        return attn.to_out[1](attn.to_out[0](result))


def qualified_upstream():
    # Arithmetic was qualified against this exact framework and normalization source.
    if (str(torch.__version__), torch.version.hip, torch.version.git_version) != (
        "2.15.0.dev20260907+rocm10.0", "7.15.26333",
        "f2ac3133587499ab2dfe33d5f4009bf89accebb0",
    ):
        return False
    expected = (
        (QwenImage21AttnProcessor, "0eb0555e21ca93195e1fe9389113cafbf0e8822f6454c3001cebb3d21521ebd7"),
        (RMSNorm, "5bcc22f3a2aa35d3f1a5efc130755984704c34fa3293dfcb39491d98fe1b611e"),
    )
    try:
        return all(_file_sha256(inspect.getfile(cls)) == digest for cls, digest in expected)
    except (OSError, TypeError):
        return False


def _plain_layer_norm(norm):
    return (isinstance(norm, torch.nn.LayerNorm) and tuple(norm.normalized_shape) == (HIDDEN,)
            and not norm.elementwise_affine and norm.weight is None and norm.bias is None
            and isinstance(norm.eps, float) and 0.0 < norm.eps < 1e-3)


def install(transformer, directory, attention_directory=None, normfuse_directory=None, fp8_directories=None, qk8_directory=None):
    """Validate the complete region before mutating this engine's model objects."""
    if not qualified_upstream():
        return None
    config = transformer.config
    if (config.num_attention_heads,config.attention_head_dim,config.num_layers,config.mlp_ratio) != (HEADS,HEAD_DIM,32,3):
        return None
    for block in transformer.transformer_blocks:
        if type(block.attn.processor) is not QwenImage21AttnProcessor:
            return None
        for norm in (block.attn.norm_q,block.attn.norm_k):
            if (norm.weight is None or tuple(norm.weight.shape) != (HEAD_DIM,)
                    or norm.weight.dtype != torch.bfloat16 or not norm.weight.is_contiguous()
                    or norm.bias is not None):
                return None
    if normfuse_directory is not None and not all(_plain_layer_norm(block.img_norm1) and _plain_layer_norm(block.img_norm2)
                                                  for block in transformer.transformer_blocks):
        normfuse_directory = None   # keep the framework LayerNorm path for an unexpected normalization configuration
    fp8 = None
    if fp8_directories is not None:
        from .fp8_regions import Fp8Regions
        fp8 = Fp8Regions(**fp8_directories) if isinstance(fp8_directories, dict) else Fp8Regions(*fp8_directories)
    native = NativeRegions(directory, attention_directory, normfuse_directory) if qk8_directory is None else NativeRegions(directory, attention_directory, normfuse_directory, qk8_directory)
    native.fp8 = fp8
    fused_norm = native.normfuse_library is not None
    for block in transformer.transformer_blocks:
        block.attn.set_processor(NativeAttentionProcessor(native,block.attn.processor))
        original = block.forward
        def forward(self,hidden_states,modulation,rotary_emb=None,attention_mask=None,
                    target_token_mask=None,layer_cache=None,kv_cache_mode=None,
                    cache_write_slice=None,segments=None,key_valid=None,_fallback=original):
            valid_mod = (modulation.dtype == torch.bfloat16 and tuple(modulation.shape) == (2,16384)
                         and modulation.is_contiguous())
            valid_mask = (target_token_mask is None or (target_token_mask.dtype == torch.bool
                          and target_token_mask.is_contiguous() and target_token_mask.numel() == hidden_states.shape[1]))
            if not valid_hidden(hidden_states) or not valid_mod or not valid_mask:
                native.counts['fallback'] += 1
                return _fallback(hidden_states,modulation,rotary_emb,attention_mask,
                    target_token_mask,layer_cache,kv_cache_mode,cache_write_slice,segments,key_valid)
            parameters = native.parameters(modulation)
            if fp8 is not None:
                fp8.begin_forward(kv_cache_mode)
            fused_quant = fused_norm and native.fused_quant_available()
            second_quantized = None
            if fused_quant:
                first, codes, scale = native.norm_modulate_quant(hidden_states,parameters,target_token_mask,0,self.img_norm1.eps)
                native.pending_quantized = (codes, scale)
            elif fused_norm:
                first = native.norm_modulate(hidden_states,parameters,target_token_mask,0,self.img_norm1.eps)
            else:
                first = native.pointwise(self.img_norm1(hidden_states),parameters,target_token_mask,0)
            branch = self.attn(hidden_states=first,attention_mask=attention_mask,rotary_emb=rotary_emb,
                layer_cache=layer_cache,kv_cache_mode=kv_cache_mode,cache_write_slice=cache_write_slice,
                segments=segments,key_valid=key_valid)
            native.pending_quantized = None
            if fused_quant:
                hidden_states,codes,scale = native.residual_norm_modulate_quant(hidden_states,branch,parameters,target_token_mask,
                                                                                HIDDEN,2*HIDDEN,self.img_norm2.eps)
                second = hidden_states; second_quantized = (codes, scale)
            elif fused_norm:
                hidden_states,second = native.residual_norm_modulate(hidden_states,branch,parameters,target_token_mask,
                                                                     HIDDEN,2*HIDDEN,self.img_norm2.eps)
            else:
                hidden_states = native.pointwise(hidden_states,parameters,target_token_mask,HIDDEN,branch)
                second = native.pointwise(self.img_norm2(hidden_states),parameters,target_token_mask,2*HIDDEN)
            del first, branch
            branch = native.feed_forward(self.img_mlp,second,second_quantized)
            result = native.pointwise(hidden_states,parameters,target_token_mask,3*HIDDEN,branch)
            native.counts['blocks'] += 1
            return result
        block.forward = types.MethodType(forward,block)
    return native
