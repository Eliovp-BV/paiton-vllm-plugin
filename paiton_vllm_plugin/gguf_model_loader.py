"""Register the GGUF byte-loading path through vLLM's supported loader API."""
from pathlib import Path
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader


@register_model_loader('paiton_gguf')
class PaitonGGUFModelLoader(BaseModelLoader):
    def download_model(self, model_config):
        if not Path(model_config.model).is_dir():
            raise ValueError('Paiton GGUF requires a staged pinned local artifact directory')

    def load_weights(self, model, model_config):
        from .models.paiton_qwen38_gguf import PaitonQwen38GGUFForCausalLM
        if not isinstance(model, PaitonQwen38GGUFForCausalLM):
            from .models.paiton_qwen38_gguf_multimodal import PaitonQwen38GGUFForConditionalGeneration
            if not isinstance(model, PaitonQwen38GGUFForConditionalGeneration):
                raise ValueError('paiton_gguf loader requires the Paiton GGUF architecture')
        model.load_weights(())
