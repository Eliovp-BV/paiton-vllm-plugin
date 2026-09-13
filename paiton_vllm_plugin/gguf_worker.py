"""Pinned vLLM worker extension preserving native GGUF image activation precision."""
import torch
from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker

logger=init_logger('vllm.paiton.gguf.worker')


class PaitonGGUFWorker(Worker):
    def init_device(self):
        super().init_device()
        config=self.vllm_config.model_config.hf_config
        if getattr(config,'architectures',None)!=['PaitonQwen38GGUFForConditionalGeneration']:
            raise ValueError('Native GGUF worker is scoped to the image GGUF architecture')
        if self.use_v2_model_runner:
            raise ValueError('Native GGUF image profile requires the pinned vLLM V1 runner')
        contract=config.paiton_qwen38_contract
        if contract.get('activation_dtype')!='float32' or not contract.get('multimodal'):
            raise ValueError('Native GGUF image worker requires the FP32 activation contract')
        runner=self.model_runner
        # Upstream uses model dtype for this staging buffer. GGUF BF16 weight
        # metadata must not round the native FP32 image/text embeddings here.
        runner.inputs_embeds=runner._make_buffer(runner.max_num_tokens,
            runner.inputs_embeds_size,dtype=torch.float32,numpy=False)
        logger.info('Preserving native FP32 multimodal embeddings in vLLM input staging')
