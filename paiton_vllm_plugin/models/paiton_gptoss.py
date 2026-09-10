"""GPT-OSS stock attention and prefill with compiled RDNA4 MXFP4 decode experts."""

import ctypes
from functools import lru_cache
import hashlib, json, os
from pathlib import Path
import torch
from vllm.logger import init_logger
from vllm.model_executor.models.gpt_oss import GptOssForCausalLM
from vllm.model_executor.layers.quantization.mxfp4 import GptOssMxfp4MoEMethod
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from triton_kernels.topk import topk

logger = init_logger(__name__)


@lru_cache(maxsize=2)
def load_region(path):
    p = Path(path).resolve()
    meta = json.loads(p.with_suffix(".json").read_text())
    if meta.get("abi") != "PaitonGptOssMoeV1" or meta.get("max_tokens") != 2:
        raise ValueError("Unsupported GPT-OSS artifact ABI")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx1201" or meta["gpu_arch"] != arch:
        raise ValueError("GPT-OSS artifact requires gfx1201")
    if hashlib.sha256(p.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Artifact checksum mismatch")
    lib = ctypes.CDLL(str(p))
    fn = lib.PaitonGptOssMoeV1
    fn.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return lib, fn, meta


class PaitonGptOssMxfp4Method(GptOssMxfp4MoEMethod):
    def __init__(self, reference, region):
        super().__init__(reference.moe)
        self.__dict__.update(reference.__dict__)
        self.region = region
        if self.mxfp4_backend != Mxfp4MoeBackend.TRITON:
            raise ValueError("Requires stock OAI Triton MXFP4 loading")

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        self.raw = [
            layer.w13_weight.storage.data.transpose(-2, -1),
            self.w13_precision_config.weight_scale.storage.data.transpose(-2, -1),
            layer.w2_weight.storage.data.transpose(-2, -1),
            self.w2_precision_config.weight_scale.storage.data.transpose(-2, -1),
            layer.w13_bias,
            layer.w2_bias,
        ]
        specs = [
            ((32, 6144, 1536), torch.uint8),
            ((32, 6144, 96), torch.uint8),
            ((32, 3072, 1536), torch.uint8),
            ((32, 3072, 96), torch.uint8),
            ((32, 6144), torch.float32),
            ((32, 3072), torch.float32),
        ]
        for t, (shape, dtype) in zip(self.raw, specs):
            if tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous():
                raise ValueError(
                    f"Unsupported MXFP4 layout {t.shape}/{t.dtype}/{t.stride()}"
                )
        if layer.expert_map is not None or layer.apply_router_weight_on_input:
            raise ValueError("Requires unsharded output-weighted experts")
        self.workspace = torch.empty(
            self.region[2]["workspace_bytes"],
            device=self.raw[0].device,
            dtype=torch.uint8,
        )

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        if (
            x.shape[0] not in (1, 2)
            or x.shape[1] != 3072
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
        ):
            return super().apply_monolithic(layer, x, router_logits, input_ids)
        # Preserve the stock top-k result rounding, then convert for the FP32 ABI.
        # Real router logits are BF16, so passing routes.data_ptr() directly is invalid.
        routes, ids, _ = topk(router_logits, 4, apply_softmax=True)
        ids = ids.int()
        routes = routes.float()
        output = torch.empty_like(x)
        ptrs = [
            t.data_ptr() for t in [x, ids, routes, *self.raw, output, self.workspace]
        ]
        status = self.region[1](
            *ptrs, x.shape[0], torch.cuda.current_stream().cuda_stream
        )
        if status:
            raise RuntimeError(f"GPT-OSS compiled region HIP status {status}")
        return output


class PaitonGptOssForCausalLM(GptOssForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        pc = vllm_config.parallel_config
        if (
            pc.tensor_parallel_size != 1
            or pc.pipeline_parallel_size != 1
            or pc.enable_expert_parallel
        ):
            raise ValueError("GPT-OSS release requires one GPU")
        if vllm_config.speculative_config or vllm_config.lora_config:
            raise ValueError("Speculation and LoRA are unqualified")
        c = vllm_config.model_config.hf_config
        for key, val in {
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "num_hidden_layers": 24,
            "num_local_experts": 32,
            "num_experts_per_tok": 4,
            "swiglu_limit": 7.0,
        }.items():
            if getattr(c, key, None) != val:
                raise ValueError("Unsupported GPT-OSS " + key)
        self._paiton_region = load_region(os.environ["PAITON_GPTOSS_ARTIFACT"])
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        count = 0
        for layer in self.model.layers:
            experts = layer.mlp.experts.routed_experts
            ref = experts.quant_method
            if type(ref) is not GptOssMxfp4MoEMethod:
                raise ValueError("Unsupported GPT-OSS quant method")
            experts._replace_quant_method(
                PaitonGptOssMxfp4Method(ref, self._paiton_region)
            )
            count += 1
        if count != 24:
            raise ValueError("Expected 24 expert layers")
        logger.info(
            "Enabled Paiton GPT-OSS MXFP4 experts for 1..2 tokens in %d layers", count
        )
        return loaded
