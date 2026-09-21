"""Exercise real libc networking and subprocess inheritance without a GPU."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(
    sys.platform == "linux" and shutil.which("cc"),
    "Linux C toolchain needed for guard fixture",
)
class OfflineTests(unittest.TestCase):
    def test_native_and_python_egress_blocked_with_loopback_and_children_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = Path(tmp) / "guard.so"
            source = (
                Path(__file__).resolve().parents[1] / "tools/native/offline_guard.c"
            )
            subprocess.run(
                [
                    "cc",
                    "-shared",
                    "-fPIC",
                    "-O2",
                    str(source),
                    "-o",
                    str(guard),
                    "-ldl",
                ],
                check=True,
            )
            env = dict(os.environ, LD_PRELOAD=str(guard), PAITON_OFFLINE_ACTIVE="1")
            code = r"""
import ctypes as C,errno,socket,subprocess,sys
from paiton_vllm_plugin.execution.offline import install,validate_native_guard
validate_native_guard()
lib=C.CDLL(None,use_errno=True)
class IPv4(C.Structure):
    _fields_=[('family',C.c_ushort),('port',C.c_ushort),('address',C.c_ubyte*4),('padding',C.c_ubyte*8)]
address=IPv4(socket.AF_INET,socket.htons(80),(C.c_ubyte*4)(203,0,113,1))
with socket.socket() as s:
    assert lib.connect(s.fileno(),C.byref(address),C.sizeof(address))==-1
    assert C.get_errno()==errno.ENETUNREACH
with socket.socket(type=socket.SOCK_DGRAM) as s:
    assert lib.sendto(s.fileno(),b'x',1,0,C.byref(address),C.sizeof(address))==-1
    assert C.get_errno()==errno.ENETUNREACH
result=C.c_void_p()
assert lib.getaddrinfo(b'example.invalid',b'80',None,C.byref(result))!=0
install()
try:socket.getaddrinfo('example.invalid',80)
except PermissionError:pass
else:raise AssertionError('Python DNS was allowed')
try:socket.create_connection(('203.0.113.1',80),timeout=.1)
except PermissionError:pass
else:raise AssertionError('Python egress was allowed')
with socket.socket() as server:
    server.bind(('127.0.0.1',0));server.listen()
    with socket.create_connection(server.getsockname()) as client:
        peer,_=server.accept()
        with peer:
            peer.sendall(b'ok');assert client.recv(2)==b'ok'
subprocess.run([sys.executable,'-c','from paiton_vllm_plugin.execution.offline import validate_native_guard; validate_native_guard()'],check=True)
"""
            result = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                text=True,
                capture_output=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
