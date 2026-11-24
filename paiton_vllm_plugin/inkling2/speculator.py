# SPDX-License-Identifier: Apache-2.0
"""DSpark speculator for the Inkling 2 draft on vLLM 0.32 (vLLM main: `vllm/models/inkling2/speculator.py`).

It stores the raw target features (each draft layer mixes them itself) and runs the draft's recurrent Markov head.
vLLM 0.32's DSpark base does the rest: the block layout (anchor plus seven masks), the context keys, the verify step.
Chosen by `drafter.install()` for `--speculative-config '{"method": "dspark", "num_speculative_tokens": 7}'`.
"""

import json
import os
import time

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


# PAITON_INK2_DRAFT_CUT=<p> (needs --no-async-scheduling and a DSpark library of ABI 2): verify only the drafts
# whose chance that the target accepts them all (the product of the draft probabilities up to there) is at least p,
# and at least one. The scheduler must know the count before it sizes the verify step, so the probabilities are
# read back from the card after every draft step; the product is taken on the host.
CUT = float(os.environ.get("PAITON_INK2_DRAFT_CUT", "0"))
_kept: list[int] | None = None  # drafts to verify per request of the last draft step, in batch order


def _kept_of(probabilities: list[float], cut: float) -> int:
    product, kept = 1.0, 0
    for p in probabilities:
        product *= p
        if product < cut:
            break
        kept += 1
    return max(kept, 1)


assert [_kept_of(row, 0.3) for row in ([0.9, 0.8, 0.1, 0.9], [0.05, 0.9], [0.9] * 7, [0.6, 0.6, 0.6])] == [2, 1, 7, 2]


def _cut_draft_lists() -> None:
    from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

    original = DraftTokensHandler.get_draft_tokens

    def get_draft_tokens(handler):
        out = original(handler)
        if out is not None and _kept is not None and len(_kept) == len(out.draft_token_ids):
            for ids, kept in zip(out.draft_token_ids, _kept):
                del ids[kept:]
        return out

    DraftTokensHandler.get_draft_tokens = get_draft_tokens


def _timed(label: str, fn, rows_of, max_rows: int):
    """PAITON_INK2_DRAFT_TIME: mean ms of `fn` over calls of at most `max_rows` rows (the decode steps), printed
    every 100 calls. Synchronizes around each call, so not for serving."""
    stat = [0, 0.0]

    def wrapper(*args, **kwargs):
        if rows_of(*args, **kwargs) > max_rows:
            return fn(*args, **kwargs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        stat[0] += 1
        stat[1] += time.perf_counter() - start
        if stat[0] % 100 == 0:
            print(f"INK2 DRAFT TIME {label}: {10 * stat[1]:.2f} ms (mean of 100, calls of at most {max_rows} rows)",
                  flush=True)
            stat[1] = 0.0
        return out

    return wrapper


def _period(label: str, fn):
    """Mean ms between two calls of `fn` (pauses above half a second are left out), printed every 100 calls."""
    stat = [0, 0.0, None]

    def wrapper(*args, **kwargs):
        now = time.perf_counter()
        if stat[2] is not None and now - stat[2] < 0.5:
            stat[0] += 1
            stat[1] += now - stat[2]
            if stat[0] % 100 == 0:
                print(f"INK2 DRAFT TIME {label}: {10 * stat[1]:.2f} ms (mean of 100)", flush=True)
                stat[1] = 0.0
        stat[2] = now
        return fn(*args, **kwargs)

    return wrapper


def _time_runner(max_rows: int, parts: bool) -> None:
    """PAITON_INK2_DRAFT_TIME=2: the parts of vLLM's step around the two forwards, each synchronized, and the
    time from step to step. =3: only the time from step to step, nothing synchronized (the served period)."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as Runner

    for name in ("update_requests", "gather_batch_req_state", "prepare_inputs", "prepare_attn", "sample",
                 "postprocess_sampled", "execute_model", "sample_tokens") if parts else ():
        setattr(Runner, name, _timed(f"runner {name}", getattr(Runner, name), lambda *a, **k: 0, max_rows))
    Runner.execute_model = _period("step period (execute_model to execute_model)", Runner.execute_model)


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
        # PAITON_INK2_DRAFT_CONF=<file.json>: per verified block, the draft probabilities of its 7 tokens and how
        # many the target accepted (an experiment: it reads back from the card every step).
        self._conf_path = os.environ.get("PAITON_INK2_DRAFT_CONF")
        self._conf: dict[int, list[float]] = {}  # request slot -> probabilities of its last drafts
        self._conf_log: list = []
        self._slots: torch.Tensor | None = None
        if CUT:
            if vllm_config.scheduler_config.async_scheduling:
                raise ValueError("PAITON_INK2_DRAFT_CUT needs --no-async-scheduling")
            _cut_draft_lists()

    def load_draft_model(self, target_model: nn.Module, target_attn_layer_names: set[str]) -> nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        model.attach_target(target_model.get_language_model())
        if os.environ.get("PAITON_INK2_DRAFT_TIME") == "3":
            _time_runner(0, False)
        elif os.environ.get("PAITON_INK2_DRAFT_TIME"):
            rows = self.max_num_reqs * self.num_query_per_req
            target_model.forward = _timed("verify forward", target_model.forward,
                                          lambda *a, **k: (k["positions"] if "positions" in k else a[1]).shape[0], rows)
            self.propose = _timed("draft step", self.propose,
                                  lambda *a, **k: (k["input_batch"] if "input_batch" in k else a[0]).num_tokens, rows)
        if os.environ.get("PAITON_INK2_DRAFT_TIME") == "2":
            # Parts of a draft step (each synchronizes; the Markov head's figure contains the base logits).
            for label, name, rows_of in (
                ("draft step: context keys", "precompute_and_store_context_kv", lambda *a, **k: a[0].shape[0]),
                ("draft step: three layers", "forward", lambda *a, **k: 0),
                ("draft step: base logits", "compute_logits", lambda *a, **k: 0),
                ("draft step: Markov head with base logits", "markov_tokens", lambda *a, **k: 0),
            ):
                setattr(model, name, _timed(label, getattr(model, name), rows_of, rows))
            self.prepare_context_anchor = _timed("draft step: Markov features", self.prepare_context_anchor,
                                                 lambda *a, **k: 0, rows)
            target_model.compute_logits = _timed("verify logits", target_model.compute_logits,
                                                 lambda *a, **k: a[0].shape[0], rows)
            _time_runner(rows, True)
        return model

    def prepare_context_anchor(self, input_batch: InputBatch, num_rejected: torch.Tensor) -> None:
        # The Markov head reads the features of each request's last committed row.
        num_reqs = input_batch.num_reqs
        self._slots = input_batch.idx_mapping[:num_reqs]
        ends = input_batch.query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs]
        rows = self.hidden_states[(ends - 1).clamp_min(0).long()]
        self.f_rnn[:num_reqs] = self.model.compute_f_rnn(rows)

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        hidden = head_hidden[self.sample_indices[: num_reqs * self.num_speculative_steps]]
        anchors = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        if not (self._conf_path or CUT):
            self.draft_tokens[:num_reqs] = self.model.markov_tokens(hidden, self.f_rnn[:num_reqs], anchors).T
            return
        tokens, probabilities = self.model.markov_tokens(hidden, self.f_rnn[:num_reqs], anchors, True)
        self.draft_tokens[:num_reqs] = tokens.T
        rows = probabilities.T.tolist()  # per request, on the host
        if CUT:
            global _kept
            _kept = [_kept_of(row, CUT) for row in rows]
        if self._conf_path:
            self._conf.update(zip(self._slots.tolist(), rows))

    def observe_verification(self, idx_mapping: torch.Tensor, num_sampled: torch.Tensor,
                             num_rejected: torch.Tensor) -> None:
        super().observe_verification(idx_mapping, num_sampled, num_rejected)
        if self._conf_path:
            n = idx_mapping.shape[0]
            none = [0.0] * self.num_speculative_steps
            self._conf_log.append([num_sampled[:n].tolist(), num_rejected[:n].tolist(),
                                   [self._conf.get(slot, none) for slot in idx_mapping.tolist()]])
            if len(self._conf_log) % 50 == 0:
                json.dump(self._conf_log, open(self._conf_path, "w"))
