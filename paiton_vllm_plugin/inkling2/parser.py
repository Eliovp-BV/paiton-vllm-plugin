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

Tool arguments stream incrementally (`_compute_arg_delta` below). 0.32's
engine rescans the whole argument text on every delta (the `args` member,
its closing brace, the safe prefix), so a 48 KB tool call cost about 190 s
of parser CPU. Here each new piece of text is scanned once; the output is the
same, delta by delta.
"""

import re
from collections.abc import Sequence

from vllm.parser.engine.adapters import make_adapters
from vllm.parser.inkling import InklingParser, _inkling_arg_converter
from vllm.renderers.inkling_encoding import (
    CONTENT_MODEL_END_SAMPLING,
    CONTENT_TEXT,
    CONTENT_THINKING,
    MESSAGE_MODEL,
    MESSAGE_SYSTEM,
    MESSAGE_TOOL,
    MESSAGE_USER,
)


_NON_WS = re.compile(r"[^ \t\r\n]")
_IN_STRING = re.compile(r'["\\]')
_STRUCT = re.compile(r'["{}\[\]:]')  # what extract_json_member_object and _safe_arg_prefix react to
_BRACES = re.compile(r'["{}]')  # what scan_json_object_end reacts to


class _Scan:
    """Left-to-right state of one of 0.32's JSON scans, fed one new piece at a time.

    kind "wrapper": `extract_json_member_object(raw, "args")` up to the colon of the `args` member.
    kind "object": `scan_json_object_end(raw, args_start)`.
    kind "prefix": the scan of `ParserEngine._safe_arg_prefix` over the `args` text.
    Every branch mirrors those functions (string and escape handling, depth, keys), so the state after
    feeding a text equals what they find in one pass over it.
    """

    __slots__ = ("kind", "pos", "depth", "in_str", "esc", "s_start", "s_parts", "last_str", "colon", "key",
                 "val_start", "val_char", "val_close", "end")

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.pos = 0  # characters fed so far
        self.depth = 0
        self.in_str = self.esc = False
        self.s_start = -1
        self.s_parts: list[str] | None = None  # text of the open string when it is a depth-1 string
        self.last_str: str | None = None  # wrapper: last_string; prefix: pending_key
        self.colon = -1  # wrapper: colon of the `args` member; prefix: last_colon
        self.key: str | None = None  # prefix: last_key
        self.val_start = -1  # prefix: first non-whitespace index after last_colon
        self.val_char = ""
        self.val_close = -1  # prefix: closing quote of the string that starts at val_start
        self.end = -1  # object: index after the closing brace

    def feed(self, text: str) -> int:
        """Scan text; return how many characters were used (fewer when the scan ends inside it)."""
        base, n, p, kind = self.pos, len(text), 0, self.kind
        while p < n:
            if self.in_str:
                if self.esc:
                    self.esc = False
                    if self.s_parts is not None:
                        self.s_parts.append(text[p])
                    p += 1
                    continue
                m = _IN_STRING.search(text, p)
                if m is None:
                    if self.s_parts is not None:
                        self.s_parts.append(text[p:])
                    break
                q = m.start()
                if text[q] == "\\":
                    if self.s_parts is not None:
                        self.s_parts.append(text[p:q + 1])
                    self.esc = True
                else:
                    if self.s_parts is not None:
                        self.s_parts.append(text[p:q])
                        self.last_str = "".join(self.s_parts)
                        self.s_parts = None
                    if kind == "prefix" and self.s_start == self.val_start:
                        self.val_close = base + q
                    self.in_str = False
                p = q + 1
                continue
            if kind == "prefix" and self.colon >= 0 and self.val_start < 0:
                m = _NON_WS.search(text, p)
                if m is None:
                    break
                self.val_start, self.val_char = base + m.start(), text[m.start()]
            m = (_BRACES if kind == "object" else _STRUCT).search(text, p)
            if m is None:
                break
            q = m.start()
            c, p = text[q], q + 1
            if c == '"':
                self.in_str = True
                self.s_start = base + q
                self.s_parts = [] if kind != "object" and self.depth == 1 else None
            elif c in "{[":
                self.depth += 1
            elif c in "}]":
                self.depth -= 1
                if kind == "object" and self.depth == 0:
                    self.end = base + p
                    self.pos = base + p
                    return p
            elif self.depth == 1:  # ":"
                if kind == "wrapper":
                    if self.last_str == "args":
                        self.colon = base + q
                        self.pos = base + p
                        return p
                else:
                    self.colon, self.key, self.last_str = base + q, self.last_str, None
                    self.val_start, self.val_char, self.val_close = -1, "", -1
        self.pos = base + n
        return n

    def safe_len(self, length: int, string_keys: set[str] | None) -> int:
        """Length of `_safe_arg_prefix(args_text, string_keys)` for the fed text of this length."""
        if self.colon < 0:
            return 0
        end = self.val_start if self.val_start >= 0 else length
        if end >= length or self.val_char != '"':
            return end
        if string_keys is not None and self.key not in string_keys:
            return end
        return self.val_close if self.val_close >= 0 else length


class _ArgStream:
    """Streaming state of one tool call: the three scans plus the arguments text not streamed yet."""

    __slots__ = ("slot", "n_parts", "raw_len", "wrapper", "args_start", "obj", "prefix", "span_len", "sent",
                 "pending", "stop")

    def __init__(self, slot) -> None:
        self.slot = slot
        self.n_parts = 0  # pieces of slot._args_parts fed so far
        self.raw_len = 0
        self.wrapper = _Scan("wrapper")
        self.args_start = -1  # index of the `args` object's "{" in the raw text
        self.obj = _Scan("object")
        self.prefix = _Scan("prefix")
        self.span_len = 0  # length of the args text so far (the converter's output)
        self.sent = 0  # length of the streamed prefix (slot.streamed_json, kept lazily)
        self.pending: list[str] = []  # args text from `sent` to `span_len`
        self.stop = ""  # "error": args is not an object; "closed": the args object is complete

    def feed(self, piece: str) -> None:
        self.raw_len += len(piece)
        t = piece
        while t and not self.stop:
            if self.args_start < 0:
                if self.wrapper.colon < 0:
                    t = t[self.wrapper.feed(t):]
                    continue
                m = _NON_WS.search(t)
                if m is None:
                    return
                if t[m.start()] != "{":
                    self.stop = "error"
                    return
                t = t[m.start():]
                self.args_start = self.raw_len - len(t)
            used = self.obj.feed(t)
            self.prefix.feed(t[:used])
            self.pending.append(t[:used])
            self.span_len += used
            if self.obj.end >= 0:
                self.stop = "closed"
            t = t[used:]

    def materialize(self) -> None:
        """Write the streamed prefix to slot.streamed_json, where 0.32's own methods read it."""
        if self.sent:
            self.slot.streamed_json = self.slot.args[self.args_start:self.args_start + self.sent]
        self.stop = "delegated"


class Inkling2Parser(InklingParser):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._arg_streams: dict[int, _ArgStream] = {}

    def _compute_arg_delta(self, idx: int, raw_delta: str) -> str | None:
        """0.32's `_compute_arg_delta` for Inkling's converter, in time proportional to the new text.

        While the `args` object is open, 0.32 computes the converter's span (verbatim, growing), leaves it
        unchanged in `_fix_arg_types` (`json.loads` fails on an unterminated object) and streams up to
        `_safe_arg_prefix`. The scans keep that state between deltas. Once the object is closed (type
        coercion may rewrite it) or `args` is not an object, 0.32's method takes over with the same
        streamed prefix.
        """
        slot = self._tool_slots[idx]
        parts = getattr(slot, "_args_parts", None)
        if self._arg_converter is not _inkling_arg_converter or not self._stream_arg_deltas or parts is None:
            return super()._compute_arg_delta(idx, raw_delta)
        st = self._arg_streams.get(idx)
        if st is None or st.slot is not slot:
            st = self._arg_streams[idx] = _ArgStream(slot)
        if st.stop != "delegated":
            for piece in parts[st.n_parts:]:
                st.feed(piece)
            st.n_parts = len(parts)
        if st.stop:
            if st.stop != "delegated":
                st.materialize()
            return super()._compute_arg_delta(idx, raw_delta)
        structural = self._arg_structural_chars
        if structural is not None and structural.isdisjoint(raw_delta):
            return None
        k = st.prefix.safe_len(st.span_len, slot.string_keys)
        if k <= st.sent:  # empty, equal to what was streamed, or not an extension of it
            return None
        text = "".join(st.pending)
        diff, rest = text[:k - st.sent], text[k - st.sent:]
        st.pending = [rest] if rest else []
        st.sent = k
        return diff

    def _flush_arg_converter(self, idx: int) -> str | None:
        st = self._arg_streams.get(idx)
        if st is not None and st.slot is self._tool_slots[idx] and st.stop != "delegated":
            st.materialize()
        return super()._flush_arg_converter(idx)

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
