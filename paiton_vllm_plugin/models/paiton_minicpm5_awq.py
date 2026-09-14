"""MiniCPM5 asymmetric W4A16 with compiled decode; stock prefill semantics."""
import ctypes
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import torch
from vllm import _custom_ops as ops, envs
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig, AutoAWQLinearMethod
from vllm.model_executor.models.llama import LlamaForCausalLM

SHAPES={(2560,2048),(2048,2048),(12288,2048),(2048,6144)}
QUALIFICATION_COUNTS={} if os.environ.get('PAITON_MINICPM5_QUALIFY')=='1' else None


@lru_cache(maxsize=1)
def load_region():
    path=Path(os.environ['PAITON_MINICPM5_ARTIFACT']).resolve()
    manifest=json.loads(path.with_suffix('.json').read_text())
    expected=dict(abi='PaitonAwqDecodeV1',gpu_arch='gfx1201',max_tokens=2,group_size=128,
                  layout='row_unsigned_int4_lsb_first_g128',activation_dtype='float16',output_dtype='float16')
    if any(manifest.get(k)!=v for k,v in expected.items()) or {tuple(s) for s in manifest.get('shapes',[])}!=SHAPES:
        raise ValueError('Incompatible MiniCPM AWQ artifact contract')
    if torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(':')[0]!='gfx1201':
        raise ValueError('MiniCPM AWQ requires gfx1201')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest.get('sha256'):
        raise ValueError('MiniCPM AWQ artifact checksum mismatch')
    lib=ctypes.CDLL(str(path));functions={}
    for n,k in SHAPES:
        decode=getattr(lib,f'PaitonAwqDecodeN{n}K{k}V1')
        decode.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p];decode.restype=ctypes.c_int
        repack=getattr(lib,f'PaitonAwqRepackN{n}K{k}V1')
        repack.argtypes=[ctypes.c_void_p]*7;repack.restype=ctypes.c_int
        functions[n,k]=(decode,repack)
    return lib,functions


@torch.library.custom_op('paiton::minicpm5_awq',mutates_args=())
def awq_projection(x:torch.Tensor,qweight:torch.Tensor,scales:torch.Tensor,qzeros:torch.Tensor,
                   packed:torch.Tensor,packed_scales:torch.Tensor,packed_zeros:torch.Tensor)->torch.Tensor:
    m,k=x.shape;n=qweight.shape[1]*8
    if ((n,k) not in SHAPES or x.dtype!=torch.float16 or not x.is_cuda
            or any(t.device!=x.device or not t.is_contiguous() for t in (x,qweight,scales,qzeros,packed,packed_scales,packed_zeros))
            or qweight.dtype!=torch.int32 or qzeros.dtype!=torch.int32 or scales.dtype!=torch.float16
            or packed.dtype!=torch.int32 or packed_scales.dtype!=torch.float16 or packed_zeros.dtype!=torch.uint8
            or tuple(qweight.shape)!=(k,n//8) or tuple(scales.shape)!=(k//128,n)
            or tuple(qzeros.shape)!=(k//128,n//8) or tuple(packed.shape)!=(n,k//8)
            or tuple(packed_scales.shape)!=(n,k//128) or packed_zeros.shape!=packed_scales.shape):
        raise ValueError('Unsupported MiniCPM AWQ tensors')
    # A single compiled graph covers prefill and decode. Dispatch stays inside
    # this opaque operation, rather than freezing a traced prefill branch.
    if m not in (1,2):
        if m>=256:
            return torch.matmul(x,ops.awq_dequantize(qweight,scales,qzeros,0,0,0))
        return ops.awq_gemm(x,qweight,scales,qzeros,8)
    output=torch.empty((m,n),device=x.device,dtype=x.dtype)
    status=load_region()[1][n,k][0](x.data_ptr(),packed.data_ptr(),packed_scales.data_ptr(),
        packed_zeros.data_ptr(),output.data_ptr(),m,torch.cuda.current_stream(x.device).cuda_stream)
    if status:raise RuntimeError(f'MiniCPM AWQ decode HIP status {status}')
    if QUALIFICATION_COUNTS is not None:
        key=f'{m}x{n}x{k}';QUALIFICATION_COUNTS[key]=QUALIFICATION_COUNTS.get(key,0)+1
    return output


@awq_projection.register_fake
def awq_projection_fake(x,qweight,scales,qzeros,packed,packed_scales,packed_zeros):
    return x.new_empty((x.shape[0],qweight.shape[1]*8))


class PaitonAWQMethod(AutoAWQLinearMethod):
    def process_weights_after_loading(self,layer):
        super().process_weights_after_loading(layer)
        k,np=layer.qweight.shape;n=np*8
        if (n,k) not in SHAPES:raise ValueError('Unexpected MiniCPM AWQ projection shape')
        packed=torch.empty((n,k//8),device=layer.qweight.device,dtype=torch.int32)
        scales=torch.empty((n,k//128),device=layer.scales.device,dtype=torch.float16)
        zeros=torch.empty((n,k//128),device=layer.qzeros.device,dtype=torch.uint8)
        status=load_region()[1][n,k][1](layer.qweight.data_ptr(),layer.scales.data_ptr(),layer.qzeros.data_ptr(),
            packed.data_ptr(),scales.data_ptr(),zeros.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if status:raise RuntimeError(f'MiniCPM AWQ preparation HIP status {status}')
        layer.register_buffer('paiton_packed',packed,persistent=False)
        layer.register_buffer('paiton_scales',scales,persistent=False)
        layer.register_buffer('paiton_zeros',zeros,persistent=False)

    def apply(self,layer,x,bias=None):
        output=awq_projection(x.reshape(-1,x.shape[-1]),layer.qweight,layer.scales,layer.qzeros,
                              layer.paiton_packed,layer.paiton_scales,layer.paiton_zeros)
        if bias is not None:output=output+bias
        return output.reshape(*x.shape[:-1],layer.qweight.shape[1]*8)


class PaitonMiniCPM5AWQForCausalLM(LlamaForCausalLM):
    def __init__(self,*,vllm_config,prefix=''):
        cfg=vllm_config.model_config.hf_config;quant=vllm_config.quant_config;parallel=vllm_config.parallel_config
        expected=dict(hidden_size=2048,intermediate_size=6144,num_hidden_layers=42,num_attention_heads=16,
                      num_key_value_heads=2,vocab_size=130560,hidden_act='silu',tie_word_embeddings=False)
        if any(getattr(cfg,k,None)!=v for k,v in expected.items()):raise ValueError('Unsupported MiniCPM geometry')
        if (vllm_config.model_config.dtype!=torch.float16 or not isinstance(quant,AutoAWQConfig)
                or quant.weight_bits!=4 or quant.group_size!=128 or not quant.zero_point
                or parallel.tensor_parallel_size!=1 or parallel.pipeline_parallel_size!=1
                or vllm_config.lora_config is not None or vllm_config.speculative_config is not None
                or envs.VLLM_BATCH_INVARIANT):
            raise ValueError('Requires asymmetric AWQ G128, FP16, TP=PP=1, no LoRA/speculation/batch-invariant mode')
        load_region();super().__init__(vllm_config=vllm_config,prefix=prefix)
        for module in self.modules():
            if isinstance(module,LinearBase) and type(module.quant_method) is AutoAWQLinearMethod:
                module.quant_method=PaitonAWQMethod(quant)
