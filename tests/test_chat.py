import argparse
import json
import unittest

from paiton_vllm_plugin import chat


class ChatTests(unittest.TestCase):
    def test_request_defaults_to_streaming_and_disables_thinking(self):
        args = argparse.Namespace(
            model="qwen38", max_tokens=32, temperature=0.7, thinking=False
        )
        messages = [{"role": "user", "content": "Hello"}]
        payload = chat._request_payload(messages, args)
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_stream_text_ignores_metadata_and_stops_at_done(self):
        events = [
            b": keep-alive\n",
            b"data: "
            + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}).encode()
            + b"\n",
            b"data: "
            + json.dumps({"choices": [{"delta": {"content": " world"}}]}).encode()
            + b"\n",
            b"data: [DONE]\n",
            b"data: this is never parsed\n",
        ]
        self.assertEqual(list(chat._stream_text(events)), ["Hello", " world"])


if __name__ == "__main__":
    unittest.main()
