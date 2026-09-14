"""Pinned Qwen3-Coder coding server for Radeon AI PRO R9700."""

import argparse
import json
import os
import sys


MODEL = "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
REVISION = "4bd30395b72ea6045edd04806c4fea448d4467b3"


def command(snapshot, stock=False):
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(snapshot), "--served-model-name", "qwen3-coder",
            "--host", "0.0.0.0", "--port", "8010", "--dtype", "bfloat16",
            "--max-model-len", "4096", "--max-num-seqs", "2",
            "--max-num-batched-tokens", "512", "--kv-cache-memory-bytes", "2G",
            "--no-enable-prefix-caching", "--attention-backend", "ROCM_ATTN",
            "--moe-backend", "triton", "--generation-config", "vllm",
            "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml",
            "--safetensors-load-strategy", "lazy", "--seed", "1201", "-O2",
            "--compilation-config", json.dumps({"cudagraph_capture_sizes": [1, 2],
                                                 "max_cudagraph_capture_size": 2})]
    if not stock:
        argv.extend(["--hf-overrides", json.dumps({"architectures": ["PaitonQwen3CoderForCausalLM"]})])
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    os.environ.update(VLLM_PLUGINS="" if args.stock else "register_paiton_models",
                      VLLM_ROCM_USE_AITER="0", OMP_NUM_THREADS="1")
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(MODEL, revision=REVISION, max_workers=2,
                                 local_files_only=args.offline)
    if args.download_only:
        print(snapshot)
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    argv = command(snapshot, args.stock)
    # These values contain model paths and public generation settings only.
    print(json.dumps({"model": MODEL, "revision": REVISION, "command": argv}), flush=True)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
