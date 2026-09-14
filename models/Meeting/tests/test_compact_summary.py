import pytest
from paiton_meeting.compact_summary import final_content


def test_rejects_truncated_generation_even_if_json_exists():
    with pytest.raises(ValueError, match='did not finish'):
        final_content('reasoning</think>{"actions": []}', False)


def test_rejects_unfinished_reasoning_or_empty_final():
    for text in ('reasoning only', 'reasoning</think>'):
        with pytest.raises(ValueError):
            final_content(text, True)


def test_only_final_content_reaches_summary_parser():
    assert final_content('reasoning</think>\n{"actions": []}', True) == '{"actions": []}'
