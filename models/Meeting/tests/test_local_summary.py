import pytest
from paiton_meeting.local_summary import LocalSummarizer, NoRedirect

@pytest.mark.parametrize('endpoint', [
    'https://api.example.com/v1/chat/completions',
    'http://192.168.1.2/v1/chat/completions',
    'http://127.0.0.1@evil.example/v1/chat/completions',
    'http://localhost/v1/chat/completions?forward=cloud',
    'http://localhost/other',
])
def test_nonlocal_endpoints_rejected_before_model_load(endpoint):
    with pytest.raises(ValueError):
        LocalSummarizer(endpoint, 'local', '/does-not-exist')


def test_redirect_cannot_forward_meeting_content():
    with pytest.raises(ValueError):
        NoRedirect().redirect_request(None, None, 307, '', {}, 'https://example.com')
