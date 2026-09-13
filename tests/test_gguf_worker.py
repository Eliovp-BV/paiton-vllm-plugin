import importlib.util
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
import unittest
import torch


class WorkerPrecisionTest(unittest.TestCase):
    def test_image_embeddings_keep_native_precision(self):
        class BaseWorker:
            def init_device(self):
                self.model_runner=SimpleNamespace(max_num_tokens=2,inputs_embeds_size=3,
                    _make_buffer=lambda *shape,dtype,numpy:torch.empty(shape,dtype=dtype))
        fake=ModuleType('vllm.v1.worker.gpu_worker');fake.Worker=BaseWorker
        logger=ModuleType('vllm.logger');logger.init_logger=lambda name:SimpleNamespace(info=lambda *a:None)
        with patch.dict('sys.modules',{'vllm.v1.worker.gpu_worker':fake,'vllm.logger':logger}):
            spec=importlib.util.spec_from_file_location('gguf_worker_test',Path(__file__).parents[1]/'paiton_vllm_plugin/gguf_worker.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        worker=module.PaitonGGUFWorker();worker.use_v2_model_runner=False
        config=SimpleNamespace(architectures=['PaitonQwen38GGUFForConditionalGeneration'],
            paiton_qwen38_contract={'activation_dtype':'float32','multimodal':True})
        worker.vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=config))
        worker.init_device()
        data=torch.tensor([[1.0001,2.0002,3.0003]]).expand(2,-1)
        worker.model_runner.inputs_embeds.copy_(data)
        self.assertEqual(worker.model_runner.inputs_embeds.dtype,torch.float32)
        self.assertTrue(torch.equal(worker.model_runner.inputs_embeds,data))
        worker.use_v2_model_runner=True
        with self.assertRaisesRegex(ValueError,'V1 runner'):worker.init_device()
