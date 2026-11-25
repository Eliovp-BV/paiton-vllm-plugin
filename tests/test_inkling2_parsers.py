"""CPU checks of the Inkling 2 reply parsers on vLLM 0.32 (needs vllm 0.32; no GPU, no model files)."""
import json
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm.parser.inkling")
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.inkling import InklingParser
from vllm.parser.parser_manager import ParserManager

import paiton_vllm_plugin.inkling2 as inkling2
from paiton_vllm_plugin.inkling2.parser import Inkling2Parser

inkling2.register()

# Ids of the checkpoint's tokenizer.json.
VOCAB = {
    "<|message_user|>": 200000, "<|message_model|>": 200001, "<|message_system|>": 200002, "<|message_tool|>": 200003,
    "<|content_text|>": 200004, "<|content_image|>": 200005, "<|content_model_end_sampling|>": 200006,
    "<|content_thinking|>": 200008, "<|end_message|>": 200010, "<|content_tool_error|>": 200022,
    "<|content_xml|>": 200024, "<|content_invoke_tool_json|>": 200049, "<|content_invoke_tool_text|>": 200057,
}
# Prompt and greedy reply of the reference fixture `tool_low`, as the checkpoint's chat template and the model write them.
PROMPT = ("<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|><|message_user|><|content_text|>"
          "What is the temperature in Antwerp right now, in celsius?<|end_message|>")
THINKING = ('The user asks for current temperature in Antwerp in celsius. Need use get_current_temperature. '
            'Location likely "Antwerp". Unit celsius.')
ARGS = {"location": "Antwerp", "unit": "celsius"}
TOOL_REPLY = (f"<|message_model|><|content_thinking|>{THINKING}<|end_message|><|message_model|>get_current_temperature"
              '<|content_invoke_tool_json|>{"name":"get_current_temperature","args":{"location":"Antwerp",'
              '"unit":"celsius"}}<|end_message|>')
TEXT_REPLY = "<|message_model|><|content_thinking|>Brussels.<|end_message|><|message_model|><|content_text|>Brussels<|end_message|>"


def tokenize(text):
    """Special tokens are one token, every other character is its own token (as vLLM's parser tests do)."""
    tokens, i = [], 0
    while i < len(text):
        marker = next((m for m in VOCAB if text.startswith(m, i)), None)
        tokens.append((VOCAB[marker], marker) if marker else (ord(text[i]), text[i]))
        i += len(marker or " ")
    return tokens


def ids(text):
    return [token_id for token_id, _ in tokenize(text)]


@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.get_vocab.return_value = dict(VOCAB)
    tok.decode.side_effect = lambda token_ids: "".join({v: k for k, v in VOCAB.items()}.get(i, chr(i)) for i in token_ids)
    tok.all_special_tokens, tok.all_special_ids = list(VOCAB), list(VOCAB.values())
    return tok


@pytest.fixture
def request_():
    req = MagicMock(spec=ChatCompletionRequest)
    req.tools, req.tool_choice, req.include_reasoning = [], "auto", True
    return req


def served_parser(tokenizer):
    """The parser `--reasoning-parser inkling2 --enable-auto-tool-choice --tool-call-parser inkling2` serves with."""
    return ParserManager.get_parser(tool_parser_name="inkling2", reasoning_parser_name="inkling2",
                                    enable_auto_tools=True)(tokenizer, [])


def stream(parser, request, text, chunk_size):
    tokens = tokenize(text)
    content, reasoning, names, args = "", "", {}, {}
    for start in range(0, len(tokens), chunk_size):
        batch = tokens[start:start + chunk_size]
        delta = parser.parse_delta("".join(t for _, t in batch), [i for i, _ in batch], request,
                                   prompt_token_ids=ids(PROMPT) if start == 0 else None,
                                   finished=start + chunk_size >= len(tokens))
        if delta is None:
            continue
        content += delta.content or ""
        reasoning += delta.reasoning or ""
        for call in delta.tool_calls or []:
            if call.function and call.function.name:
                names[call.index] = call.function.name
            if call.function and call.function.arguments:
                args[call.index] = args.get(call.index, "") + call.function.arguments
    return content, reasoning, list(names.values()), list(args.values())


def test_reasoning_end_rule(tokenizer):
    # Why the subclass exists: 0.32's rule reads the user's text block as the end of reasoning.
    assert InklingParser(tokenizer).is_reasoning_end(ids(PROMPT)) is True
    parser = Inkling2Parser(tokenizer)
    assert parser.is_reasoning_end(ids(PROMPT)) is False
    assert parser.is_reasoning_end(ids(PROMPT + "<|message_model|><|content_thinking|>Hm")) is False
    assert parser.is_reasoning_end(ids(PROMPT + "<|message_model|><|content_text|>Br")) is True


def test_tool_reply(tokenizer, request_):
    reasoning, content, calls = served_parser(tokenizer).parse(TOOL_REPLY, request_, enable_auto_tools=True)
    assert reasoning == THINKING
    assert not content
    assert [call.name for call in calls] == ["get_current_temperature"]
    assert json.loads(calls[0].arguments) == ARGS


def test_text_reply(tokenizer, request_):
    reasoning, content, calls = served_parser(tokenizer).parse(TEXT_REPLY, request_, enable_auto_tools=True)
    assert (reasoning, content) == ("Brussels.", "Brussels")
    assert not calls


@pytest.mark.parametrize("chunk_size", [1, 3, 64])
def test_tool_reply_streaming(tokenizer, request_, chunk_size):
    content, reasoning, names, args = stream(served_parser(tokenizer), request_, TOOL_REPLY, chunk_size)
    assert reasoning == THINKING
    assert content == ""
    assert names == ["get_current_temperature"]
    assert json.loads(args[0]) == ARGS


@pytest.mark.parametrize("chunk_size", [1, 3, 64])
def test_text_reply_streaming(tokenizer, request_, chunk_size):
    content, reasoning, names, _ = stream(served_parser(tokenizer), request_, TEXT_REPLY, chunk_size)
    assert (reasoning, content, names) == ("Brussels.", "Brussels", [])


# Tool payloads for the incremental argument streaming: escapes (split mid-escape at 1 character per delta), unicode,
# nested objects and arrays, whitespace, an `args` value that is not an object.
TOOL_BODIES = [
    '{"name":"write_file","args":{"path":"/a\\"b","content":"' + 'x\\\\y\\"z\\u00e9 \U0001F600 ' * 40 + '"}}',
    '{"name":"configure","args":{"o":{"a":[1,{"b":"}]\\"{"}],"c":"\\\\"},"n":7,"content":"done"}}',
    '{ "name" : "f" , "args" : { "k" : "v \\" w" , "n" : 7 } }',
    '{"name":"f","args":"not an object"}',
]


def tool_text(body):
    return f"<|message_model|><|content_thinking|>t<|end_message|><|message_model|><|content_invoke_tool_json|>{body}<|end_message|>"


def deltas(parser, request, text, chunk_size):
    """Every delta the served parser returns, tool call ids blanked (random per parser)."""
    tokens, out = tokenize(text), []
    for start in range(0, len(tokens), chunk_size):
        batch = tokens[start:start + chunk_size]
        delta = parser.parse_delta("".join(t for _, t in batch), [i for i, _ in batch], request,
                                   prompt_token_ids=ids(PROMPT) if start == 0 else None,
                                   finished=start + chunk_size >= len(tokens))
        dump = delta and delta.model_dump(exclude_none=True)
        for call in (dump or {}).get("tool_calls") or []:
            call.pop("id", None)
        out.append(dump)
    return out


@pytest.mark.parametrize("chunk_size", [1, 2, 5])
@pytest.mark.parametrize("body", TOOL_BODIES)
def test_tool_args_stream_as_vllm_does(tokenizer, request_, body, chunk_size, monkeypatch):
    new = deltas(served_parser(tokenizer), request_, tool_text(body), chunk_size)
    monkeypatch.setattr(Inkling2Parser, "_compute_arg_delta", ParserEngine._compute_arg_delta)
    monkeypatch.setattr(Inkling2Parser, "_flush_arg_converter", ParserEngine._flush_arg_converter)
    assert new == deltas(served_parser(tokenizer), request_, tool_text(body), chunk_size)


def test_tool_args_are_not_rescanned(tokenizer, request_, monkeypatch):
    # 0.32 rescans the whole argument text on every delta; the open object is only scanned once here.
    calls = []
    real = ParserEngine._safe_arg_prefix
    monkeypatch.setattr(ParserEngine, "_safe_arg_prefix", staticmethod(lambda *a: calls.append(1) or real(*a)))
    body = '{"name":"write_file","args":{"path":"/a","content":"' + "y" * 3000 + '"}}'
    content, reasoning, names, args = stream(served_parser(tokenizer), request_, tool_text(body), 1)
    assert json.loads(args[0]) == {"path": "/a", "content": "y" * 3000}
    assert len(calls) <= 2  # only after the object closes, where 0.32's own method takes over
