"""Qwen3-Coder using a compiled INT4 expert region on RDNA4.

The stock Qwen3MoE model owns attention, KV cache, RoPE, routing, loading and
prefill. Only expert execution for up to two tokens uses the compiled ABI.
No private compiler source is required at runtime.
"""

import ctypes
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import WNA16MoEBackend
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import CompressedTensorsWNA16MoEMethod
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
from vllm.platforms import current_platform

logger = init_logger(__name__)


@lru_cache(maxsize=4)
def load_region(path):
    artifact = Path(path).resolve()
    manifest = json.loads(artifact.with_suffix(".json").read_text())
    if manifest.get("abi") != "PaitonQwen3CoderMoeV1" or manifest.get("serving_max_tokens") != 2:
        raise ValueError("Unsupported Qwen3-Coder artifact ABI")
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Qwen3-Coder artifact checksum mismatch")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
    if arch != manifest["gpu_arch"] or arch != "gfx1201":
        raise ValueError(f"Qwen3-Coder release requires gfx1201, found {arch}")
    dll = ctypes.CDLL(str(artifact))
    fn = dll.PaitonQwen3CoderMoeV1
    fn.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    logger.info("Loaded Qwen3-Coder compiled region: sha256=%s", manifest["sha256"])
    return dll, fn, manifest


class PaitonCoderW4A16Method(CompressedTensorsWNA16MoEMethod):
    def __init__(self, reference, region):
        super().__init__(reference.weight_quant, reference.input_quant, reference.moe)
        self.is_k_full = reference.is_k_full
        if self.wna16_backend != WNA16MoEBackend.TRITON:
            raise ValueError("Qwen3-Coder compiled experts require the stock Triton WNA16 loading layout")
        if self.num_bits != 4 or self.group_size != 32 or not self.symmetric or self.actorder is not None:
            raise ValueError("Qwen3-Coder compiled experts require symmetric INT4 G32 without activation ordering")
        self.region = region

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        tensors = [(layer.w13_weight, (128, 1536, 1024), torch.uint8),
                   (layer.w2_weight, (128, 2048, 384), torch.uint8),
                   (layer.w13_weight_scale, (128, 1536, 64), torch.bfloat16),
                   (layer.w2_weight_scale, (128, 2048, 24), torch.bfloat16)]
        for tensor, shape, dtype in tensors:
            if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous():
                raise ValueError(f"Unsupported expert tensor: {tensor.shape}/{tensor.dtype}")
        if layer.apply_router_weight_on_input or layer.expert_map is not None:
            raise ValueError("Compiled Qwen3-Coder experts require output routing weights and TP=EP=1")
        self.workspace = torch.empty(2*8*(2048+768)*2, dtype=torch.uint8, device=layer.w13_weight.device)

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        if x.shape[0] > 2:
            return super().apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
        if x.dtype != torch.bfloat16 or x.shape[1] != 2048 or not x.is_contiguous():
            raise ValueError("Unsupported Qwen3-Coder activation layout")
        if shared_experts is not None or topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
            raise ValueError("Unsupported Qwen3-Coder routing contract")
        output = torch.empty_like(x)
        pointers = [t.data_ptr() for t in (x, topk_ids, topk_weights, layer.w13_weight,
                    layer.w13_weight_scale, layer.w2_weight, layer.w2_weight_scale, output, self.workspace)]
        status = self.region[1](*pointers, x.shape[0], torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"Paiton Qwen3-Coder expert execution failed with HIP status {status}")
        return output


class PaitonQwen3CoderForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        if not current_platform.is_rocm() or current_platform.__class__.__name__ != "RocmPlatform":
            raise ValueError("Qwen3-Coder requires stock vLLM RocmPlatform")
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1 or parallel.enable_expert_parallel:
            raise ValueError("Qwen3-Coder compiled experts require one GPU, TP=PP=1, no expert parallelism")
        if vllm_config.speculative_config is not None or vllm_config.lora_config is not None:
            raise ValueError("Speculation and LoRA have not been qualified for this artifact")
        artifact = os.environ.get("PAITON_QWEN3_CODER_ARTIFACT")
        if not artifact:
            raise ValueError("Set PAITON_QWEN3_CODER_ARTIFACT to the compiled .so")
        region = load_region(artifact)
        config = vllm_config.model_config.hf_text_config
        for key, value in {"num_hidden_layers": 48, "hidden_size": 2048,
                           "num_experts": 128, "num_experts_per_tok": 8,
                           "moe_intermediate_size": 768, "norm_topk_prob": True}.items():
            if getattr(config, key, None) != value:
                raise ValueError(f"Unsupported Qwen3-Coder {key}")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._paiton_region = region

    def load_weights(self, weights):
        # vLLM's packed checkpoint loader dispatches on the stock method's exact
        # class name. Preserve it through loading, then replace it before the
        # runtime's process_weights_after_loading pass.
        loaded = super().load_weights(weights)
        count = 0
        for layer in self.model.layers:
            experts = layer.mlp.experts.routed_experts
            reference = experts.quant_method
            if type(reference) is not CompressedTensorsWNA16MoEMethod:
                raise ValueError("Unsupported Qwen3-Coder quantization method")
            experts._replace_quant_method(PaitonCoderW4A16Method(reference, self._paiton_region))
            count += 1
        if count != 48:
            raise ValueError(f"Expected 48 MoE layers, found {count}")
        logger.info("Enabled compiled INT4 experts on all %d Qwen3-Coder layers for 1..2 tokens", count)
        return loaded
