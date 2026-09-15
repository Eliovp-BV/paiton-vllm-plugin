"""Streaming checkpoint I/O through vLLM's public model-loader registration.

This external adapter uses safetensors' pread backend without modifying either
library. The compiler/native runtime has no dependency on this loader.
"""
import copy
import time

from safetensors import safe_open
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


@register_model_loader("paiton_native_mxfp4")
class PaitonNativeMxFp4Loader(DefaultModelLoader):
    def __init__(self, load_config):
        # Keep the user's config intact. The inherited file resolver should
        # select safetensors after registration has selected this loader class.
        effective = copy.copy(load_config)
        effective.load_format = "safetensors"
        super().__init__(effective)
        if self.load_config.model_loader_extra_config.get("enable_multithread_load"):
            raise ValueError("The Paiton streaming loader uses one tensor iterator")

    def _get_weights_iterator(self, source):
        if self.local_expert_ids is not None:
            raise ValueError("The Paiton streaming loader currently covers dense models")
        _, files, is_safetensors = self._prepare_weights(
            source.model_or_path, source.subfolder, source.revision,
            False, source.allow_patterns_overrides,
        )
        if not is_safetensors:
            raise ValueError("The Paiton streaming loader requires safetensors")
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        for path in files:
            with safe_open(path, framework="pt", device="cpu", backend="pread") as reader:
                for name in reader.keys():
                    yield source.prefix + name, reader.get_tensor(name)
