"""Stage the hash-locked Qwen3.8 MXFP4 image context without downloads or builds.

Run from a checkout containing the matching public adapter sources. Pass the
Hugging Face companion's ``overlay`` directory to --overlay. Only the explicit
runtime allowlist is copied; compiler sources and build tools are never inputs.
The output directory must not already exist.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


ADAPTER_MODULES = (
    "__init__.py",
    "artifact_manifest.py",
    "attention_native.py",
    "dflash2_native_context.py",
    "dflash2_native_head.py",
    "gdn_native_conv_prefill.py",
    "gdn_native_prefill.py",
    "gdn_native_replay.py",
    "gdn_upstream_control.py",
    "mxfp4_native.py",
    "mxfp4_native_loader.py",
    "mxfp4_native_model.py",
    "mxfp4_native_worker.py",
    "norm_native_fp8.py",
    "silu_native_fp8.py",
    "target_head_native.py",
    "target_head_policy.py",
)

NATIVE_LIBRARIES = (
    "attention.so",
    "bias_cast.so",
    "context_expand.so",
    "conv.so",
    "draft_head.so",
    "gdn_prefill.so",
    "layout.so",
    "norm_fp8.so",
    "prefill.so",
    "projection.so",
    "projection_pipeline.so",
    "projection_rows.so",
    "quantization.so",
    "replay.so",
    "silu_fp8.so",
    "target_logits.so",
)

NATIVE_MANIFESTS = (
    "attention.json",
    "dflash_context.json",
    "draft_head.json",
    "gdn_conv_prefill.json",
    "gdn_prefill.json",
    "gdn_runtime.json",
    "norm_fp8.json",
    "runtime.json",
    "silu_fp8.json",
    "target_logits.json",
)

RELEASE_FILES = (
    "THIRD_PARTY_NOTICES.md",
    "checkpoint.lock.json",
    "container_entrypoint.py",
    "engine-profile.json",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(model_dir, overlay):
    repository = model_dir.parents[1]
    sources = {
        "Dockerfile": model_dir / "Dockerfile",
        "plugin/pyproject.toml": model_dir / "runtime-pyproject.toml",
    }
    for name in ADAPTER_MODULES:
        sources[f"plugin/paiton_vllm_plugin/{name}"] = (
            repository / "paiton_vllm_plugin" / name
        )
    for name in NATIVE_LIBRARIES + NATIVE_MANIFESTS:
        sources[f"native/{name}"] = overlay / name
    for name in RELEASE_FILES:
        sources[f"release/{name}"] = model_dir / name
    return sources


def stage_context(model_dir, overlay, output):
    """Validate every input, then copy only the locked 49-file payload."""
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output must be a new directory: {output}")
    if not overlay.is_dir():
        raise ValueError(f"Overlay directory does not exist: {overlay}")
    sources = source_files(model_dir, overlay)
    lock = json.loads((model_dir / "build-context.lock.json").read_text())
    hashes = lock.get("files")
    if lock.get("schema") != 1 or not isinstance(hashes, dict):
        raise ValueError("Unsupported build-context lock schema")
    if set(hashes) != set(sources):
        raise ValueError("Build-context lock must match the explicit runtime allowlist")
    for name, expected in hashes.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid SHA256 in build-context lock: {name}")
    for name, source in sources.items():
        if not source.is_file():
            raise ValueError(f"Required runtime input is missing: {source}")
        if sha256_file(source) != hashes[name]:
            raise ValueError(f"Runtime input hash mismatch: {source}")

    # HF cache snapshots may use symlinks to immutable blobs. Copy the verified
    # bytes, not symlinks, and recheck the result in case an input changed.
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, source in sources.items():
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha256_file(destination) != hashes[name]:
                raise ValueError(f"Runtime input changed while staging: {source}")
    except BaseException:
        shutil.rmtree(output)
        raise
    return len(sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True,
                        help="Downloaded HF companion overlay directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="New Docker build-context directory")
    args = parser.parse_args()
    model_dir = Path(__file__).resolve().parent
    try:
        count = stage_context(model_dir, args.overlay, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot prepare image context: {error}\n")
    print(f"Verified and staged {count} runtime files in {args.output.resolve()}")
    print("Build with docker build using this directory as the build context.")


if __name__ == "__main__":
    main()
