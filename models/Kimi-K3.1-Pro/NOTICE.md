# License and provenance

- Kimi-K3.1-Pro weights, tokenizer and chat encoding: Moonshot AI, under the Kimi K3.1-Pro License (`MODEL_LICENSE`),
  revision `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e`. The weights are not in the image; you download them and mount
  them at `/model`. The server loads the checkpoint's own `tokenization_kimi.py` and `encoding_k3.py` from that folder.
  Note the license's condition for Model-as-a-Service businesses above 20 million US dollars of revenue.
- The code in this folder (`kimi31_community/`, `tests/`, `serve-docker.sh`, `Dockerfile`) is under this repository's
  Apache-2.0 license. It is plain PyTorch and contains no Paiton engine code, kernels or weights.
- The runtime (ROCm 10.1.0, PyTorch 2.13.0, Python 3.12) comes from AMD's
  `rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.13.0`; their license files remain in the image.
  Transformers 5.20.2 (Apache-2.0), tiktoken 0.14.0 (MIT) and safetensors 0.8.0 (Apache-2.0) are installed from PyPI.
