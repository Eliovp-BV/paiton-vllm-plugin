# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 reasoning and tool parser for vLLM 0.32.

vLLM 0.32's `InklingParser` already has the block grammar Inkling 2 writes.
It lacks the Inkling 2 part of vLLM main's `vllm/parser/inkling.py`
(c4f6d8c7): an Inkling 2 prompt ends between messages and the model opens
its own turn, so 0.32's rule takes the text block of the last user message
for a finished reasoning phase. `is_reasoning_end` below is main's
`_inkling2_is_reasoning_end`, unchanged. Main's `adjust_request` (JSON
response formats) is not here: see `UPSTREAM.md`.
"""

from collections.abc import Sequence

from vllm.parser.engine.adapters import make_adapters
from vllm.parser.inkling import InklingParser
from vllm.renderers.inkling_encoding import (
    CONTENT_MODEL_END_SAMPLING,
    CONTENT_TEXT,
    CONTENT_THINKING,
    MESSAGE_MODEL,
    MESSAGE_SYSTEM,
    MESSAGE_TOOL,
    MESSAGE_USER,
)


class Inkling2Parser(InklingParser):
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Whether the latest block is a text block of a model message.

        A prompt that ends between messages has not ended reasoning: the
        model may open a thinking block.
        """
        vocab = self.vocab
        text_id = vocab[CONTENT_TEXT]
        block_ids = {
            text_id,
            vocab[CONTENT_THINKING],
            vocab[CONTENT_MODEL_END_SAMPLING],
        }
        role_ids = {
            vocab[token]
            for token in (MESSAGE_MODEL, MESSAGE_USER, MESSAGE_SYSTEM, MESSAGE_TOOL)
        }
        latest_block = None
        for token_id in reversed(input_ids):
            if token_id in role_ids:
                return latest_block == text_id and token_id == vocab[MESSAGE_MODEL]
            if latest_block is None and token_id in block_ids:
                latest_block = token_id
        return latest_block == text_id


(
    Inkling2ParserReasoningAdapter,
    Inkling2ParserToolAdapter,
) = make_adapters(Inkling2Parser)


class Inkling2EngineToolParser(Inkling2ParserToolAdapter):  # type: ignore[valid-type, misc]
    # No Inkling structural-tag grammar is wired up yet; fall back to auto
    # parsing for named/required tool choice.
    structural_tag_model = None
    supports_required_and_named = False
