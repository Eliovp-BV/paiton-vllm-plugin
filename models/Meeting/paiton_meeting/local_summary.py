"""Bounded requests to an explicitly local summarizer; no raw-content logging."""
import ipaddress
import json
import time
import urllib.request
from urllib.parse import urlsplit

from .summary import SUMMARY_SCHEMA, VERIFY_SCHEMA, summarize, verify_summary


class LocalSummarizer:
    def __init__(self, endpoint, model, tokenizer_directory, timeout=180, reasoning_effort='low', enable_thinking=None):
        parsed = urlsplit(endpoint)
        try:
            loopback = ipaddress.ip_address(parsed.hostname or '').is_loopback
        except ValueError:
            loopback = parsed.hostname == 'localhost'
        if not loopback or parsed.scheme != 'http' or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Summarizer must be an HTTP loopback endpoint in the local runtime.')
        if parsed.path.rstrip('/') != '/v1/chat/completions':
            raise ValueError('Unexpected summarizer endpoint path.')
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.template_kwargs = {} if enable_thinking is None else {'enable_thinking': enable_thinking}
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory, local_files_only=True)
        # Ignore proxy environment variables: meeting content must stay local.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.timings = []

    def generate(self, system, data, schema=SUMMARY_SCHEMA):
        messages = [dict(role='system', content=system), dict(role='user', content=data)]
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **self.template_kwargs)
        # 8K context, reserve generation headroom. Fail before dropping evidence.
        generation_budget = min(3000, 8192 - len(tokens))
        if generation_budget < 1536:
            raise ValueError('Summary request exceeds the configured context budget.')
        body = dict(model=self.model, messages=messages, temperature=0, max_tokens=generation_budget,
                    response_format={'type': 'json_schema', 'json_schema': {
                        'name': 'meeting_summary', 'strict': True, 'schema': schema}})
        if self.reasoning_effort is not None: body['reasoning_effort'] = self.reasoning_effort
        if self.template_kwargs: body['chat_template_kwargs'] = self.template_kwargs
        request = urllib.request.Request(self.endpoint, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        start = time.perf_counter()
        with self.opener.open(request, timeout=self.timeout) as response:
            raw = response.read(2*1024*1024 + 1)
        if len(raw) > 2*1024*1024:
            raise ValueError('Summary response exceeds the size limit.')
        result = json.loads(raw)
        choice = result['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Summary generation did not finish; no partial summary was accepted.')
        content = choice['message'].get('content')
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Summarizer returned no structured content.')
        self.timings.append(dict(seconds=time.perf_counter()-start, usage=result.get('usage', {})))
        return content

    def summarize(self, segments):
        start = time.perf_counter()
        self.timings = []
        draft = summarize(segments, self.generate, lambda s: len(self.tokenizer.encode(s)))
        final, audit = verify_summary(draft, segments,
            lambda system, data: self.generate(system, data, VERIFY_SCHEMA))
        return dict(summary=final, summary_audit=audit,
                    summary_seconds=time.perf_counter()-start, summary_calls=self.timings,
                    summary_status='generated-draft')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Summarizer redirects are forbidden.')
