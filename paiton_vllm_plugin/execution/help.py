"""Render the installed vLLM parser without runtime device discovery.

The pinned 0.26 parser instantiates DeviceConfig while constructing help and
otherwise fails on a device-less machine. A metadata-only ROCm device type is
enough for that factory. This process can only render help, never serve.
"""

import importlib.metadata
import runpy
import sys


def main():
    arguments = sys.argv[1:]
    if not any(x in ("--help", "-h") or x.startswith("--help=") for x in arguments):
        raise RuntimeError("This helper only forwards vLLM help")
    version = importlib.metadata.version("vllm")
    if version not in (
        "0.29.0",
        "0.26.1.dev1+g396cd1a43.rocm714",
        "0.28.0.dev0+eliovp.quark48606.g39bd959b5.rocm714",
    ):
        raise RuntimeError("GPU-free help has not been reviewed for vLLM " + version)
    import vllm.platforms
    from vllm.platforms.interface import Platform

    class HelpDevice(Platform):
        device_type = "cuda"  # ROCm's torch device spelling; no device is opened.

    vllm.platforms._current_platform = HelpDevice()
    import torch

    def no_gpu(*args, **kwargs):
        raise RuntimeError("The installed help parser attempted to initialize a GPU")

    torch.cuda._lazy_init = no_gpu
    sys.argv = ["vllm", *arguments]
    runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__")


if __name__ == "__main__":
    main()
