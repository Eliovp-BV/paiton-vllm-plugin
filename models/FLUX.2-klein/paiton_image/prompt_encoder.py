"""Expose the encoder states consumed by the fixed FLUX.2 klein profile."""
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast


class PromptEncoder(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=getattr(model,'_orig_mod',model)
        self.config=self.model.config
        if self.config.model_type!='qwen3' or self.config.num_hidden_layers<28:
            raise ValueError('Unsupported prompt encoder profile')

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    def forward(self,**kwargs):
        if not kwargs.get('output_hidden_states') or kwargs.get('use_cache',True):
            raise ValueError('The prompt encoder requires intermediate states without a KV cache')
        output=self.model(**kwargs)
        # Returning only consumed states lets the standard compiler discard
        # subsequent layers and the language-model output head.
        return BaseModelOutputWithPast(hidden_states=tuple(
            output.hidden_states[i] if i in (9,18,27) else None for i in range(28)))
