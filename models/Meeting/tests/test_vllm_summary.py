from types import SimpleNamespace
import sys
import pytest
from paiton_meeting.vllm_summary import VLLMSummarizer


def helper(monkeypatch, finish_reason='stop', tokens=20):
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=lambda **kw: kw))
    model = VLLMSummarizer.__new__(VLLMSummarizer)
    model.timings = []
    model.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: 'prompt',
                                      encode=lambda _: [1]*tokens)
    model.model = SimpleNamespace(generate=lambda *a, **kw: [SimpleNamespace(outputs=[
        SimpleNamespace(text='reasoning</think>{"overview":"draft"}',
                        finish_reason=finish_reason, token_ids=[1,2,3])])])
    return model


def test_vllm_rejects_truncated_generation_even_with_complete_json(monkeypatch):
    model = helper(monkeypatch, finish_reason='length')
    with pytest.raises(ValueError, match='did not finish'):
        model.generate('system', 'transcript')
    assert model.timings == []


def test_vllm_context_limit_is_checked_before_inference(monkeypatch):
    model = helper(monkeypatch, tokens=13000)
    model.model.generate = lambda *a, **kw: pytest.fail('oversized prompt reached GPU')
    with pytest.raises(ValueError, match='context budget'):
        model.generate('system', 'transcript')
