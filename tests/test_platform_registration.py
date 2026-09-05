from enum import Enum
import importlib
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paiton_vllm_plugin import paiton_platform_plugin, register_paiton_models


class PlatformRegistrationTests(unittest.TestCase):
    def test_registers_product_facing_rdna_architectures(self):
        class Registry:
            registered = {}

            @classmethod
            def get_supported_archs(cls):
                return list(cls.registered)

            @classmethod
            def register_model(cls, architecture, model_path):
                cls.registered[architecture] = model_path

        with patch.dict(sys.modules, {"vllm": SimpleNamespace(ModelRegistry=Registry)}):
            register_paiton_models()
        self.assertEqual(
            Registry.registered["PaitonQwen38ForCausalLM"],
            "paiton_vllm_plugin.models.paiton_qwen38:PaitonQwen38ForCausalLM",
        )
        self.assertEqual(
            Registry.registered["PaitonQwen38ForConditionalGeneration"],
            "paiton_vllm_plugin.models.paiton_qwen38_multimodal:PaitonQwen38ForConditionalGeneration",
        )
        self.assertEqual(
            Registry.registered["PaitonOrnith15ForCausalLM"],
            "paiton_vllm_plugin.models.paiton_ornith15:PaitonOrnith15ForCausalLM",
        )

    def test_ornith_resolves_hybrid_cache_dtypes_before_model_init(self):
        class CompilationMode(Enum):
            NONE = 0

        class CUDAGraphMode(Enum):
            NONE = 0

        class RocmPlatform:
            @classmethod
            def check_and_update_config(cls, vllm_config):
                del vllm_config

        fake_vllm = SimpleNamespace()
        fake_logger = SimpleNamespace(
            init_logger=lambda _name: SimpleNamespace(info=lambda *_args: None)
        )
        modules = {
            "vllm": fake_vllm,
            "vllm.logger": fake_logger,
            "vllm.platforms": SimpleNamespace(),
            "vllm.platforms.rocm": SimpleNamespace(RocmPlatform=RocmPlatform),
            "vllm.config": SimpleNamespace(),
            "vllm.config.compilation": SimpleNamespace(
                CompilationMode=CompilationMode,
                CUDAGraphMode=CUDAGraphMode,
            ),
        }
        cache = SimpleNamespace(
            cache_dtype="auto",
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
            mamba_cache_mode="none",
            enable_prefix_caching=False,
            block_size=16,
        )
        compilation = SimpleNamespace(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
            cudagraph_capture_sizes=[],
            custom_ops=[],
        )
        config = SimpleNamespace(
            cache_config=cache,
            compilation_config=compilation,
            parallel_config=SimpleNamespace(worker_cls="auto"),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["PaitonOrnith15ForCausalLM"],
                    quantization_config=None,
                )
            ),
        )
        module_name = "paiton_vllm_plugin.paiton_platform"
        sys.modules.pop(module_name, None)
        try:
            with patch.dict(sys.modules, modules):
                platform = importlib.import_module(module_name).PaitonPlatform
                platform.check_and_update_config(config)
        finally:
            sys.modules.pop(module_name, None)
        self.assertEqual(cache.mamba_ssm_cache_dtype, "float32")
        self.assertEqual(
            config.parallel_config.worker_cls,
            "vllm.v1.worker.gpu_worker.Worker",
        )

    @patch.dict(os.environ, {"PAITON_GPU_ARCH": "gfx1201"}, clear=False)
    def test_enables_platform_for_gfx1201(self):
        self.assertEqual(
            paiton_platform_plugin(),
            "paiton_vllm_plugin.paiton_platform.PaitonPlatform",
        )

    @patch.dict(
        os.environ,
        {"PAITON_GPU_ARCH": "gfx1201", "VLLM_DISABLE_PAITON_PLATFORM": "1"},
        clear=False,
    )
    def test_explicit_disable_wins_on_gfx1201(self):
        self.assertIsNone(paiton_platform_plugin())

    @patch.dict(
        os.environ,
        {
            "PAITON_GPU_ARCH": "gfx1201",
            "VLLM_PAITON_VANILLA_ROCM_PLATFORM": "1",
        },
        clear=False,
    )
    def test_reference_harness_selects_unmodified_rocm_platform(self):
        self.assertEqual(
            paiton_platform_plugin(),
            "vllm.platforms.rocm.RocmPlatform",
        )


if __name__ == "__main__":
    unittest.main()
