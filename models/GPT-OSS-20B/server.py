"""Pinned GPT-OSS chat server for Radeon AI PRO R9700."""

import argparse
import json
import os
import sys
from pathlib import Path


MODEL = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"


def command(snapshot, stock=False, qualification=False):
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(snapshot),
        "--served-model-name",
        "gpt-oss-20b",
        "--host",
        "127.0.0.1" if qualification else "0.0.0.0",
        "--port",
        "8020",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "2",
        "--max-num-batched-tokens",
        "512",
        "--kv-cache-memory-bytes",
        "2G",
        "--no-enable-prefix-caching",
        "--attention-backend",
        "TRITON_ATTN",
        "--moe-backend",
        "triton",
        "--generation-config",
        "vllm",
        "--reasoning-parser",
        "openai_gptoss",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "openai",
        "--safetensors-load-strategy",
        "lazy",
        "--seed",
        "1201",
        "-O2",
        "--compilation-config",
        json.dumps(
            {"cudagraph_capture_sizes": [1, 2], "max_cudagraph_capture_size": 2}
        ),
    ]
    if not stock:
        argv.extend(
            [
                "--hf-overrides",
                json.dumps({"architectures": ["PaitonGptOssForCausalLM"]}),
            ]
        )
    if qualification:
        argv += ["--worker-extension-cls", "qualification_worker.QualificationWorker",
                 "--chat-template", str(Path(__file__).parent / "benchmark" / "chat-template-2026-09-09.jinja")]
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--qualification", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.update(
        VLLM_PLUGINS="" if args.stock else "register_paiton_models",
        VLLM_ROCM_USE_AITER="0",
        OMP_NUM_THREADS="1",
    )
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        MODEL,
        revision=REVISION,
        max_workers=2,
        allow_patterns=["config.json", "generation_config.json", "tokenizer.json",
                        "tokenizer_config.json", "chat_template.jinja",
                        "model.safetensors.index.json", "model-*.safetensors",
                        "LICENSE", "USAGE_POLICY"],
        ignore_patterns=["original/*", "metal/*"],
        local_files_only=args.offline,
    )
    if args.download_only:
        print(snapshot)
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    if args.qualification:
        os.environ["VLLM_SERVER_DEV_MODE"] = "1"
        # GPT-OSS uses Harmony rendering, which bypasses the Jinja template date.
        os.environ["VLLM_SYSTEM_START_DATE"] = "2026-09-09"
        os.environ["PYTHONPATH"] = str(Path(__file__).parent / "benchmark") + os.pathsep + os.environ.get("PYTHONPATH", "")
    argv = command(snapshot, args.stock, args.qualification)
    # These values contain model paths and public generation settings only.
    print(
        json.dumps({"model": MODEL, "revision": REVISION, "command": argv}), flush=True
    )
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
