"""External framework adapter for qualified BF16 regions; no compiler dependency.

Attention masks/cache sequence follows pinned Diffusers QwenImage21AttnProcessor
(Apache-2.0). Attention itself, GEMMs, checkpoint arithmetic and scheduler remain
the existing implementations. Every intermediate BF16 rounding is explicit in
the independently built native library.
"""
import ctypes as C
import hashlib
import json
import inspect
from pathlib import Path
import types

import torch
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.normalization import RMSNorm


def _file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class NativeRegions:
    def __init__(self, directory):
        directory = Path(directory)
        meta = json.loads((directory/'manifest.json').read_text())
        binary = directory/meta['file']
        if (meta['architecture'] != 'gfx1201' or meta['abi_version'] != 1
                or _file_sha256(binary) != meta['sha256']):
            raise RuntimeError('Image21 fusion artifact identity mismatch')
        if not torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx1201'):
            raise RuntimeError('Image21 native fusions require gfx1201')
        self.library = C.CDLL(str(binary.resolve()))
        self.library.PaitonImage21GetAbiVersion.restype = C.c_uint
        if self.library.PaitonImage21GetAbiVersion() != 1:
            raise RuntimeError('Image21 fusion ABI mismatch')
        self.library.PaitonImage21GetTargetArch.restype = C.c_char_p
        if self.library.PaitonImage21GetTargetArch() != b"gfx1201":
            raise RuntimeError("Image21 fusion target mismatch")
        self.library.PaitonImage21Initialize.restype = C.c_int
        if self.library.PaitonImage21Initialize():
            raise RuntimeError('Image21 fusion initialization failed')
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
        self.counts = dict(blocks=0,norm_rope=0,prepare=0,modulation=0,residual=0,silu=0,attention_pack=0,fallback=0)
        self.manifest = meta

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
        self.call(self.prepare,modulation.data_ptr(),out.data_ptr(),4096)
        self.counts['prepare'] += 1
        return out

    def pointwise(self, x, parameters, mask, offset, branch=None):
        out = torch.empty_like(x)
        self.call(self.modulate,x.data_ptr(),0 if branch is None else branch.data_ptr(),
                  parameters.data_ptr(),0 if mask is None else mask.data_ptr(),out.data_ptr(),
                  x.shape[1],4096,offset)
        self.counts['modulation' if branch is None else 'residual'] += 1
        return out

    def feed_forward(self, mlp, x):
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
                    or tuple(x.shape[2:])!=(32,128) or not 0<x.shape[1]<=40000):
                self.counts['fallback']+=1
                return dispatch_attention_fn(query,key,value,**kwargs)
        for x in (query,key,value):
            packed=torch.empty((1,32,x.shape[1],128),dtype=x.dtype,device=x.device)
            self.call(self.pack,x.data_ptr(),packed.data_ptr(),x.shape[1])
            inputs.append(packed.permute(0,2,1,3))
            self.counts['attention_pack']+=1
        return dispatch_attention_fn(*inputs,**kwargs)


def valid_hidden(x):
    return (not torch.is_grad_enabled() and x.is_cuda and x.dtype == torch.bfloat16
            and x.ndim == 3 and x.shape[0] == 1 and x.shape[2] == 4096
            and 0 < x.shape[1] <= 40000 and x.is_contiguous())


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
        query = attn.to_q(hidden_states).unflatten(-1,(32,128))
        key = attn.to_k(hidden_states).unflatten(-1,(32,128))
        value = attn.to_v(hidden_states).unflatten(-1,(32,128))
        query = self.native.norm_rope(query,attn.norm_q,rotary_emb)
        key = self.native.norm_rope(key,attn.norm_k,rotary_emb)
        if layer_cache is not None:
            if kv_cache_mode == 'extract' and cache_write_slice is not None:
                layer_cache.store(key[:,cache_write_slice].clone(),value[:,cache_write_slice].clone())
            elif kv_cache_mode == 'cached':
                cached_k,cached_v = layer_cache.get()
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


def install(transformer, directory):
    """Validate the complete region before mutating this engine's model objects."""
    if not qualified_upstream():
        return None
    config = transformer.config
    if (config.num_attention_heads,config.attention_head_dim,config.num_layers,config.mlp_ratio) != (32,128,32,3):
        return None
    for block in transformer.transformer_blocks:
        if type(block.attn.processor) is not QwenImage21AttnProcessor:
            return None
        for norm in (block.attn.norm_q,block.attn.norm_k):
            if (norm.weight is None or tuple(norm.weight.shape) != (128,)
                    or norm.weight.dtype != torch.bfloat16 or not norm.weight.is_contiguous()
                    or norm.bias is not None):
                return None
    native = NativeRegions(directory)
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
            first = native.pointwise(self.img_norm1(hidden_states),parameters,target_token_mask,0)
            branch = self.attn(hidden_states=first,attention_mask=attention_mask,rotary_emb=rotary_emb,
                layer_cache=layer_cache,kv_cache_mode=kv_cache_mode,cache_write_slice=cache_write_slice,
                segments=segments,key_valid=key_valid)
            hidden_states = native.pointwise(hidden_states,parameters,target_token_mask,4096,branch)
            del first, branch
            second = native.pointwise(self.img_norm2(hidden_states),parameters,target_token_mask,8192)
            branch = native.feed_forward(self.img_mlp,second)
            result = native.pointwise(hidden_states,parameters,target_token_mask,12288,branch)
            native.counts['blocks'] += 1
            return result
        block.forward = types.MethodType(forward,block)
    return native
