# Image requests

Start the [NEO model package](README.md#launch), then send a local image through
vLLM's ordinary chat-completions API. This example uses Python's standard library:

```bash
python3 - picture.png <<'PY'
import base64
import json
import sys
import urllib.request
from pathlib import Path

image = Path(sys.argv[1])
mime = 'image/png' if image.suffix.lower() == '.png' else 'image/jpeg'
data_url = f'data:{mime};base64,' + base64.b64encode(image.read_bytes()).decode()
body = {
    'model': 'qwen38-neo',
    'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': data_url}},
        {'type': 'text', 'text': 'Describe this image.'},
    ]}],
    'temperature': 0,
    'max_tokens': 256,
    'chat_template_kwargs': {'enable_thinking': False},
}
request = urllib.request.Request(
    'http://127.0.0.1:8000/v1/chat/completions',
    data=json.dumps(body).encode(),
    headers={'Content-Type': 'application/json'},
)
with urllib.request.urlopen(request, timeout=180) as response:
    result = json.load(response)
print(result['choices'][0]['message']['content'])
PY
```

The qualified profile accepts one still image per request, including the image
items in conversation history. Additional requests queue behind one active
sequence. Images are resized to at most 1,048,576 pixels / 4,096 vision patches,
which merge into at most 1,024 language embeddings. The image, text prompt, and
output share the 8,192-token context limit. Video and MTP are disabled.

PNG and JPEG use the existing vLLM/Transformers image processor. Vision encoding,
projection, language prefill, and language decode execute native Paiton artifacts.
Use `stream: true` for ordinary OpenAI-compatible server-sent event streaming.
