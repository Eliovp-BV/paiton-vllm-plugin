# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Looped Inkling2 checkpoints (`loop_config`) on ROCm.

Port of `nvidia/loop.py` (vLLM main c4f6d8c7) onto the AMD operators of
`model.py`. What differs from the NVIDIA file: each sublayer's short conv and
the next AttnRes mix run as two kernels (the AMD path has no fused
`conv_mix` and no all-reduce workspace), the controller GEMMs are plain
`torch.bmm`, and the controllers own their weights and are stacked after
loading.

The core layers `[core_start, core_end)` run `num_passes` times: a prelude,
then each pass from the AttnRes state after the prelude, then the coda from
the last pass's state. Each pass of a core layer has its own KV and
conv-state caches. In passes after the first, every core sublayer input
(attention q/k/v and MLP) is blended with a message: a normalized-value
AttnRes read of the previous pass's blocks, merged by a gated low-rank
controller.
"""

import torch
from torch import nn

from vllm.distributed import tensor_model_parallel_all_reduce

from ..configs import Inkling2ModelConfig
from ..quant_config import Inkling2QuantConfig
from .model import (
    TAP_DIR,
    Inkling2DecoderLayer,
    Inkling2ForConditionalGeneration,
    Inkling2Model,
    Inkling2RMSNorm,
)
from . import ops_dump
from .ops.attn_res import attn_res
from .ops.loop import blend_rms_norm, pair_rms_norm

LOOP_STREAMS = ("attn", "k", "v", "mlp")


class Inkling2MessageController(nn.Module):
    """Blends a loop message into a core sublayer input: from the rank-`r`
    projection `h` of the RMS-normalized `[dest, message]`,
    `(1 - tanh(forget(h))) * dest + tanh(gain(h)) * message` in fp32.

    Holds the checkpoint's weights; `Inkling2LoopDecoderLayer.blend` runs
    every stream at once on their stacked copies.
    """

    def __init__(self, rank: int, width: int) -> None:
        super().__init__()
        for name, shape in (
            ("down_weight", (rank, 2 * width)),
            ("gain_weight", (width, rank)),
            ("forget_weight", (width, rank)),
        ):
            param = torch.empty(shape, dtype=torch.bfloat16)
            self.register_parameter(name, nn.Parameter(param, requires_grad=False))


class Inkling2LoopDecoderLayer(Inkling2DecoderLayer):
    """A loop core layer: per-pass caches and a message controller per
    stream (`LOOP_STREAMS`), whose weights are stacked over the streams."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            config, layer_id, quant_config, prefix, num_passes=config.num_loop_passes
        )
        assert config.loop_config is not None
        rank, width = config.loop_config["gate_rank"], config.hidden_size
        self.controllers = nn.ModuleDict(
            {
                stream: Inkling2MessageController(rank, width)
                for stream in LOOP_STREAMS
            }
        )

    def process_weights_after_loading(self) -> None:
        # down (S, r, 2D) and [gain; forget] (S, 2D, r), bf16.
        controllers = list(self.controllers.values())
        down = torch.stack([c.down_weight.data for c in controllers])
        gain_forget = torch.stack(
            [torch.cat((c.gain_weight.data, c.forget_weight.data)) for c in controllers]
        )
        self.register_buffer("controller_down", down, persistent=False)
        self.register_buffer("controller_gain_forget", gain_forget, persistent=False)

    def blend(
        self,
        streams: slice,
        dest: torch.Tensor,
        message: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor:
        """The `streams` controllers on fp32 `(S, T, D)` `dest` (may
        broadcast over streams) and `message`, each output RMS-normalized
        with its row of `norm_weight` `(S, D)`: bf16 `(S, T, D)`."""
        pair = pair_rms_norm(dest, message)
        down = self.controller_down[streams]
        up = self.controller_gain_forget[streams]

        def gemms() -> torch.Tensor:
            hidden = torch.bmm(pair, down.transpose(1, 2))
            # bf16 operands, fp32 accumulation and result.
            return torch.bmm(hidden.float(), up.transpose(1, 2).float())

        gain_forget = ops_dump.record(
            "controller_gemms", {"pair": pair, "down": down, "up": up}, gemms
        )
        return blend_rms_norm(dest, message, gain_forget, norm_weight, norm_eps)


class Inkling2LoopModel(Inkling2Model):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        # Serving options set through --hf-overrides skip the constructor.
        config.check_loop_options()
        if config.loop_share_cache:
            raise ValueError("loop_share_cache is not supported on ROCm")
        super().__init__(config, quant_config, prefix)
        self.loop_core = config.loop_core_layers
        self.hidden_size = config.hidden_size
        # Message pseudo-queries (raw), two per core layer: attn/mlp and k/v.
        shape = (2 * len(self.loop_core), config.hidden_size)
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        # Per core layer, the `LOOP_STREAMS` queries scaled by D**-0.5.
        self.register_buffer(
            "message_queries",
            torch.zeros(
                len(self.loop_core),
                len(LOOP_STREAMS),
                config.hidden_size,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        # Bring-up check: the state after every layer visit of one forward.
        self.tap: list[torch.Tensor] | None = [] if TAP_DIR else None

    def process_weights_after_loading(self) -> None:
        queries = self.middle_cycle_cross_pass_queries.view(-1, 2, self.hidden_size)
        kv_queries = self.middle_cycle_cross_pass_kv_queries.view(
            -1, 2, self.hidden_size
        )
        # (attn, k, v, mlp) = (queries[0], kv[0], kv[1], queries[1]).
        stacked = torch.stack(
            (queries[:, 0], kv_queries[:, 0], kv_queries[:, 1], queries[:, 1]), 1
        )
        self.message_queries.copy_(stacked * self.hidden_size**-0.5)

    def _make_layer(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> Inkling2DecoderLayer:
        if layer_id in config.loop_core_layers:
            return Inkling2LoopDecoderLayer(config, layer_id, quant_config, prefix)
        return super()._make_layer(config, layer_id, quant_config, prefix)

    def _max_loop_blocks(self) -> int:
        """Completed AttnRes blocks a loop forward holds at most: each pass
        restarts from the prelude's, and the coda continues the last pass."""
        num_blocks = max_blocks = 0
        core = self.loop_core
        for i in [*range(core.start), *core, *range(core.stop, len(self.layers))]:
            num_blocks += i % self.block_size == 0
            max_blocks = max(max_blocks, num_blocks)
        return max_blocks

    def _mix_next(
        self,
        stack: torch.Tensor,
        last: torch.Tensor,
        addend: torch.Tensor | None,
        query: torch.Tensor,
        norm: Inkling2RMSNorm | None,
    ) -> torch.Tensor:
        """The AttnRes mixes through `norm`, or in fp32 without one."""
        if norm is None:
            return attn_res(stack, last, addend, query, None, 0.0, self.normalize_values)
        return attn_res(
            stack, last, addend, query, norm.weight, norm.eps, self.normalize_values
        )

    def _run_layer(
        self,
        i: int,
        pass_idx: int,
        stack: torch.Tensor,
        num_blocks: int,
        inputs: torch.Tensor | None,
        bank: torch.Tensor | None,
        next_query: torch.Tensor,
        next_norm: Inkling2RMSNorm | None,
        positions: torch.Tensor,
        tau: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor]:
        """Run decoder layer `i` in loop pass `pass_idx` on its `inputs`
        (the attention inputs; for a visit with messages from `bank`, the
        fp32 mixes they blend into). Returns the number of completed AttnRes
        blocks after it and the mixes of its output with `next_query` through
        `next_norm` (fp32 without it), the next visit's inputs."""
        layer = self.layers[i]
        partial = stack[num_blocks]
        owner = layer.conv_states[pass_idx]
        messages = None
        if bank is not None:
            assert isinstance(layer, Inkling2LoopDecoderLayer) and inputs is not None
            # All four streams' messages: a mix of the bank's two blocks.
            messages = attn_res(
                bank[:1],
                bank[1],
                None,
                self.message_queries[i - self.loop_core.start],
                None,
                0.0,
                True,
            )
            norm = layer.attn_norm
            dests, norm_weight = inputs, norm.weight
            if layer.is_local:
                dests = dests.expand(3, -1, -1)
                norm_weight = norm_weight.expand(3, -1)
            attn_inputs = layer.blend(
                slice(0, 3), dests, messages[:3], norm_weight, norm.eps
            )
        elif inputs is None:
            assert num_blocks == 0
            attn_inputs = [
                layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)
            ]
        else:
            attn_inputs = inputs
        attn_partial = layer.attention(positions, attn_inputs, tau, pass_idx=pass_idx)
        attn_out = layer.attn_sconv(
            tensor_model_parallel_all_reduce(attn_partial), positions, owner=owner
        )

        # The MLP input mixes the blocks with (partial +) attn_out.
        first = i % self.block_size == 0
        mlp_in = self._mix_next(
            stack[: num_blocks + first],
            attn_out if first else partial,
            None if first else attn_out,
            self.attn_res.mlp_queries[i],
            None if messages is not None else layer.mlp_norm,
        )
        if messages is not None:
            assert isinstance(layer, Inkling2LoopDecoderLayer)
            norm = layer.mlp_norm
            mlp_in = layer.blend(
                slice(3, 4), mlp_in, messages[3:], norm.weight, norm.eps
            )

        # The MLP short conv also applies the residual update.
        residual = None if first else partial
        if residual is None:
            num_blocks += 1
        layer.mlp_sconv(
            layer.mlp(mlp_in[0]),
            positions,
            addend=attn_out,
            residual=residual,
            out=stack[num_blocks],
            owner=owner,
        )
        outputs = self._mix_next(
            stack[:num_blocks], stack[num_blocks], None, next_query, next_norm
        )
        return num_blocks, outputs

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Prelude, `num_passes` core passes, then the coda."""
        num_tokens, hidden = inputs_embeds.shape
        core = self.loop_core
        num_passes = self.config.num_loop_passes
        stack = inputs_embeds.new_empty(
            (self._max_loop_blocks() + 1, num_tokens, hidden)
        )
        stack[0].copy_(inputs_embeds)
        tau = self._log_scaling_tau(positions)
        # (layer, pass) visits; the prelude and coda run as pass 0.
        visits = [(i, 0) for i in range(core.start)]
        visits += [(i, p) for p in range(num_passes) for i in core]
        visits += [(i, 0) for i in range(core.stop, len(self.layers))]
        aux_hidden_states = []
        if self.tap is not None:
            self.tap[:] = [inputs_embeds.float().cpu()]
        next_norm: Inkling2RMSNorm | None
        num_blocks = 0
        inputs = bank = prelude_blocks = prelude_partial = prelude_dests = None
        for visit, (i, pass_idx) in enumerate(visits):
            ops_dump.where = f"{i}.{pass_idx}"
            looped = pass_idx > 0
            if i == core.start and not looped:
                prelude_blocks = num_blocks
                prelude_partial = stack[num_blocks].clone()
            elif i == core.start:
                # Every pass restarts from the prelude's state, with the
                # previous pass's blocks (after the prelude's) and partial as
                # the messages' bank.
                assert prelude_blocks is not None and prelude_partial is not None
                bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                num_blocks = prelude_blocks
                stack[num_blocks].copy_(prelude_partial)
                if prelude_dests is None:
                    query = self.attn_res.attn_queries[i - 1]
                    prelude_dests = attn_res(
                        stack[:num_blocks],
                        prelude_partial,
                        None,
                        query,
                        None,
                        0.0,
                        True,
                    )
                inputs = prelude_dests
            if visit in self.aux_hidden_state_layers:
                aux_hidden_states.append(stack[num_blocks].clone())
            if visit + 1 == len(visits):
                next_query, next_norm = self.attn_res.output_query, self.norm
            else:
                next_i, next_pass = visits[visit + 1]
                next_query = self.attn_res.attn_queries[next_i - 1]
                # Messages blend into fp32 mixes; a pass restart discards it.
                next_norm = None if next_pass else self.layers[next_i].attn_norm
            num_blocks, inputs = self._run_layer(
                i,
                pass_idx,
                stack,
                num_blocks,
                inputs,
                bank if looped else None,
                next_query,
                next_norm,
                positions,
                tau,
            )
            if self.tap is not None:
                self.tap.append(stack[num_blocks].float().cpu())
        assert inputs is not None
        ops_dump.where = "post"
        out = inputs[0]
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out


class Inkling2LoopForConditionalGeneration(Inkling2ForConditionalGeneration):
    """The `inkling2_loop` checkpoints' architecture name; the model is
    `Inkling2ForConditionalGeneration` with an `Inkling2LoopModel`."""


if ops_dump.DIR is not None:  # per-operator fixtures, bring-up only
    ops_dump.install()
