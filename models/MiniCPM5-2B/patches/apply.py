"""Apply the reviewed lazy-import fix only to the exact qualified vLLM input."""
import hashlib
from pathlib import Path
import subprocess

root=Path('/opt/python/lib/python3.14/site-packages')
source=root/'vllm/v1/attention/backends/fa_utils.py'
expected='f8e34b566d101d30e9def08110058bd04e87f465282cbbe7f5f0b1d8ae985a2c'
if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
    raise SystemExit('Unexpected vLLM FlashAttention source; review patch compatibility first')
subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(Path(__file__).with_name('lazy-rocm-flash-attn.patch'))],cwd=root,check=True)
