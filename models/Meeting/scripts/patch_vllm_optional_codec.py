"""Treat an unavailable optional TorchCodec video library as unavailable.

The pinned inherited image has a CUDA TorchCodec wheel. The meeting pipeline
uses CPU PyAV audio decoding; text-only vLLM must not load that video backend.
No CUDA library or replacement video implementation is installed.
"""
import hashlib
from pathlib import Path

path = Path('/opt/python/lib/python3.14/site-packages/vllm/multimodal/video.py')
expected = '3a2987da066ed47e4db8377c06f99b49a4e0f30620c752514612ea50381fd7ad'
if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
    raise RuntimeError('Pinned vLLM optional-codec source changed; review the patch before rebuilding.')
text = path.read_text()
old = 'from torchcodec.decoders import VideoDecoder\nexcept (ImportError, RuntimeError):'
assert text.count(old) == 1
path.write_text(text.replace(old, 'from torchcodec.decoders import VideoDecoder\nexcept (ImportError, RuntimeError, OSError):'))
