"""Scoped adapter for five DFlash MLPs; arithmetic lives in native HIP."""
import types
import torch
from .native_runtime import POLICY, producer

@torch.library.custom_op('paiton::dflash_silu_quant',mutates_args=())
def native_producer(x:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    return producer(x)

@native_producer.register_fake
def _(x):
    return (torch.empty((x.shape[0],17408),device=x.device,dtype=torch.float8_e4m3fn),
            torch.empty((x.shape[0],136),device=x.device,dtype=torch.float32))

def native_forward(self,x):
    # Eager BF16 act/quant is a distinct contract; preserve it outside compilation.
    supported=(torch.compiler.is_compiling()
        and x.ndim==2 and x.shape[1]==5120 and x.dtype==torch.bfloat16 and x.is_contiguous()
        and getattr(self.down_proj,'_radiance_preshuffled',None)==(5120,17408)
        and self.down_proj.weight.dtype==torch.float8_e4m3fn
        and self.down_proj.weight_scale_inv.dtype==torch.float32
        and self.down_proj.weight_scale_inv.is_contiguous()
        and tuple(self.down_proj.weight_scale_inv.shape)==(40,136)
        and self.down_proj.bias is None)
    if not supported:return self._paiton_original_forward(x)
    gate,_=self.gate_up_proj(x)
    q,scales=native_producer(gate)
    return torch.ops.radiance.preshuffle_gemm(q,self.down_proj.weight,scales,
        self.down_proj.weight_scale_inv,5120,17408)

def configure_mlp(mlp,config):
    if not POLICY.allows('silu_quant'):return
    from vllm.distributed import get_tensor_model_parallel_world_size
    if (config.hidden_size,config.intermediate_size,config.hidden_act)!=(5120,17408,'silu'):
        return
    if get_tensor_model_parallel_world_size()!=1:return
    if hasattr(mlp,'_paiton_original_forward'):raise RuntimeError('Duplicate DFlash MLP adapter')
    mlp._paiton_original_forward=mlp.forward
    mlp.forward=types.MethodType(native_forward,mlp)
