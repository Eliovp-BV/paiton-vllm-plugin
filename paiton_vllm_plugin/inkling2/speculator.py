# SPDX-License-Identifier: Apache-2.0
"""DSpark speculator for the Inkling 2 draft on vLLM 0.32 (vLLM main: `vllm/models/inkling2/speculator.py`).

It stores the raw target features (each draft layer mixes them itself) and runs the draft's recurrent Markov head.
vLLM 0.32's DSpark base does the rest: the block layout (anchor plus seven masks), the context keys, the verify step.
Chosen by `drafter.install()` for `--speculative-config '{"method": "dspark", "num_speculative_tokens": 7}'`.
"""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


class Inkling2DSparkSpeculator(DSparkSpeculator):
    _speculator_name = "Inkling2 DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        config = self.draft_model_config.hf_config
        assert not self.sample_from_anchor
        assert self._draft_topk is None and not self.enable_adaptive_verification
        if self.draft_logits is not None:
            # shortcut: the native Markov step returns the argmax only; wire its logits output for sampled drafts
            raise ValueError("the Inkling 2 drafter drafts greedily: leave draft_sample_method at its default")
        # Target features of every row, as the target hands them over (combine_hidden_states is the identity).
        self.hidden_states = torch.zeros(
            self.max_num_tokens, len(config.target_layer_ids) * config.hidden_size, dtype=self.dtype, device=device)
        self.f_rnn = torch.zeros(self.max_num_reqs, config.hidden_size, dtype=self.dtype, device=device)

    def load_draft_model(self, target_model: nn.Module, target_attn_layer_names: set[str]) -> nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        model.attach_target(target_model.get_language_model())
        return model

    def prepare_context_anchor(self, input_batch: InputBatch, num_rejected: torch.Tensor) -> None:
        # The Markov head reads the features of each request's last committed row.
        num_reqs = input_batch.num_reqs
        ends = input_batch.query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs]
        rows = self.hidden_states[(ends - 1).clamp_min(0).long()]
        self.f_rnn[:num_reqs] = self.model.compute_f_rnn(rows)

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        hidden = head_hidden[self.sample_indices[: num_reqs * self.num_speculative_steps]]
        anchors = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        tokens = self.model.markov_tokens(hidden, self.f_rnn[:num_reqs], anchors)
        self.draft_tokens[:num_reqs] = tokens.T
