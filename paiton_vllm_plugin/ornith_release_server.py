"""One-command server entrypoint for Paiton Ornith 1.5 on RDNA4."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from paiton_vllm_plugin.ornith_reshard import reshard_checkpoint, sha256_file
from paiton_vllm_plugin.qwen38_release_server import (
    ReleaseModelError,
    _validate_runtime_environment,
)


BASE_MODEL = "Capicua25x/Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4"
BASE_REVISION = "9e488f46c0f7969f84c9923ee0256311cd50316e"
CHECKPOINT_NAME = "model.safetensors"
CHECKPOINT_SIZE = 22_931_751_024
CHECKPOINT_SHA256 = "a33626f89489eaea92ab2f03f330c2df3c52da2b271fc3987ee253cea270f7b7"
DRAFT_NAME = "dflash-draft/model.safetensors"
DRAFT_SIZE = 771_819_674
DRAFT_SHA256 = "1fb90ef50a32bfb8dd2abfe601dd3608d6d5b59dc342820a98830f76f8cd72b7"

ARTIFACT_NAME = (
    "Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4_"
    "gfx1201_tp1_mt8192_ctx8192.so"
)
MANIFEST_NAME = ARTIFACT_NAME.removesuffix(".so") + ".manifest.json"
OVERLAY_FILES = {
    ARTIFACT_NAME: (
        4_096_480,
        "6351af55b031691c170310ac93efb754868da3c7509f37d8b30eb5b9ef17c30d",
    ),
    MANIFEST_NAME: (
        342_284,
        "78e1e2abfa540fcbd8da3ca9449a8c25cecb75a6ec6c3a5d8d6b052d129aeebb",
    ),
    "config.json": (
        17_739,
        "65c9cbd329d26d5a4c856a3d82a27ba28ce2129f3b8e981e1b0a9be3e8984c92",
    ),
}
TOP_LEVEL_METADATA = (
    "chat_template.jinja",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
DRAFT_METADATA = ("README.md", "config.json", "model.safetensors")
SOURCE_METADATA_FILES = {
    "chat_template.jinja": (
        7_536,
        "182e77dd83bd8e9ca818b240b82e28f243762cd5dda32e6eef327df7b1cd107e",
    ),
    "configuration.json": (
        58,
        "c1b09db419119513247e9b8b912c4b9897106c9b20c6cada7e107d993c5435eb",
    ),
    "generation_config.json": (
        214,
        "b8eb74d15e0a56623d00ccd14950a4bb87fabbf84b5cc030dcc904b899fb1eb5",
    ),
    "merges.txt": (
        3_353_259,
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    ),
    "preprocessor_config.json": (
        390,
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    ),
    "processor_config.json": (
        1_191,
        "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    ),
    "tokenizer.json": (
        19_989_325,
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    ),
    "tokenizer_config.json": (
        1_124,
        "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b",
    ),
    "video_preprocessor_config.json": (
        385,
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    ),
    "vocab.json": (
        6_722_759,
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    ),
    "dflash-draft/README.md": (
        6_749,
        "550185d622c9134c07f0f679e5a7919dc1194567c1810d9c3ea4f75fbdf02522",
    ),
    "dflash-draft/config.json": (
        1_222,
        "20784b063040f7871efec4bf63342e5cb941907925943540176dfbb8aaf186dc",
    ),
    DRAFT_NAME: (DRAFT_SIZE, DRAFT_SHA256),
}
RESHARDED_FILES = {
    "model.safetensors.index.json": (
        6_863_662,
        "539ba77bd898e1e24b0a0defd0b22acc78b2ed05b3272f82a4be60d72c2f4792",
    ),
    "model-00001-of-00006.safetensors": (
        3_999_524_216,
        "8a63bb123d1967c564aec1cf4f4815ff61a031cfb531b884fae70f8bfb7705c5",
    ),
    "model-00002-of-00006.safetensors": (
        4_001_295_064,
        "766b5a69a69ca9d967766d5cd8803228120ba3c91a001ed88efbeadd54eb28ed",
    ),
    "model-00003-of-00006.safetensors": (
        4_001_541_216,
        "008f114b295a5ba6efcf7d0735aaa7533cd50fc02511bba235d01e78c08e6d78",
    ),
    "model-00004-of-00006.safetensors": (
        4_001_479_208,
        "f880a5ea24c4159a1a4e5f6a31db8051758a33080c8b5737a30909f5cc160259",
    ),
    "model-00005-of-00006.safetensors": (
        4_001_480_744,
        "3c90e22db4584da41bd56fa4634c661881015676c3cf585cf4afcf24cb5df966",
    ),
    "model-00006-of-00006.safetensors": (
        2_926_307_856,
        "cb95d1e6340b4c5cd33b69b24c05baefa2358366124aa0817baf5bce6b1992a9",
    ),
}


def _snapshot_download(**kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _verify(path: Path, size: int, digest: str, description: str) -> Path:
    if not path.is_file() or (path.is_symlink() and not path.resolve().is_file()):
        raise ReleaseModelError(f"missing {description}: {path}")
    if path.stat().st_size != size:
        raise ReleaseModelError(f"wrong size for {description}: {path}")
    if sha256_file(path) != digest:
        raise ReleaseModelError(f"wrong SHA-256 for {description}: {path}")
    return path


def _resolve_base_model() -> Path:
    configured = os.environ.get("PAITON_BASE_MODEL")
    if configured:
        root = Path(configured).resolve()
    else:
        root = Path(
            _snapshot_download(
                repo_id=BASE_MODEL,
                revision=BASE_REVISION,
                allow_patterns=[
                    "*.json",
                    "*.jinja",
                    "*.txt",
                    CHECKPOINT_NAME,
                    "dflash-draft/*",
                ],
            )
        ).resolve()
    _verify(root / CHECKPOINT_NAME, CHECKPOINT_SIZE, CHECKPOINT_SHA256, "checkpoint")
    _verify(root / DRAFT_NAME, DRAFT_SIZE, DRAFT_SHA256, "DFlash draft")
    return root


def _overlay_dir() -> Path:
    root = Path(os.environ.get("PAITON_ORNITH_OVERLAY", "/opt/paiton/ornith-overlay"))
    for name, (size, digest) in OVERLAY_FILES.items():
        _verify(root / name, size, digest, f"release overlay {name}")
    return root


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source.resolve(), destination)
    except OSError:
        shutil.copy2(source.resolve(), destination)


def _validate_existing_stage(stage: Path) -> bool:
    marker = stage / "paiton-ornith-release.json"
    if not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if (
        metadata.get("base_model") != BASE_MODEL
        or metadata.get("revision") != BASE_REVISION
    ):
        return False
    if metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        return False
    for name, (size, digest) in {
        **OVERLAY_FILES,
        **SOURCE_METADATA_FILES,
        **RESHARDED_FILES,
    }.items():
        path = stage / name
        if (
            not path.is_file()
            or path.stat().st_size != size
            or sha256_file(path) != digest
        ):
            return False
    report_path = stage / "reshard-report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if (
        report.get("source_sha256") != CHECKPOINT_SHA256
        or report.get("source_file_bytes") != CHECKPOINT_SIZE
        or report.get("logical_payload_sha256")
        != "5e2eb4f54472e6490325ca382e732d2b9af3758504cfab8d5682db9be9fbf94a"
        or report.get("logical_payload_bytes") != 22_922_817_248
        or report.get("tensor_count") != 63_441
        or report.get("max_shard_bytes") != 4_000_000_000
    ):
        return False
    report_files = {
        record.get("filename"): (record.get("file_bytes"), record.get("sha256"))
        for record in report.get("shards", [])
        if isinstance(record, dict)
    }
    return report_files == {
        name: fingerprint
        for name, fingerprint in RESHARDED_FILES.items()
        if name.endswith(".safetensors")
    }


def _stage_model(base: Path, overlay: Path) -> Path:
    cache_root = Path(os.environ.get("PAITON_ORNITH_CACHE", "/models/cache/ornith"))
    final = cache_root / BASE_REVISION
    if final.exists():
        if _validate_existing_stage(final):
            return final
        raise ReleaseModelError(
            f"existing Ornith runtime is incomplete or modified: {final}"
        )
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".staging-", dir=cache_root))
    try:
        report = reshard_checkpoint(base / CHECKPOINT_NAME, temporary)
        if report["source_sha256"] != CHECKPOINT_SHA256:
            raise ReleaseModelError("checkpoint changed while it was being resharded")
        for name in TOP_LEVEL_METADATA:
            _copy_or_link(base / name, temporary / name)
        for name in DRAFT_METADATA:
            _copy_or_link(
                base / "dflash-draft" / name,
                temporary / "dflash-draft" / name,
            )
        for name in OVERLAY_FILES:
            _copy_or_link(overlay / name, temporary / name)
        marker = {
            "base_model": BASE_MODEL,
            "revision": BASE_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "artifact_sha256": OVERLAY_FILES[ARTIFACT_NAME][1],
            "dflash_max_speculative_tokens": 16,
        }
        (temporary / "paiton-ornith-release.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def _select_model() -> Path:
    explicit = os.environ.get("PAITON_MODEL")
    if explicit:
        return Path(explicit).resolve()
    return _stage_model(_resolve_base_model(), _overlay_dir())


def build_server_command(extra_args: list[str] | None = None) -> list[str]:
    model = _select_model()
    os.environ.setdefault("PAITON_CHAT_MODEL", "ornith")
    command = [
        "vllm",
        "serve",
        str(model),
        "--served-model-name",
        os.environ.get("PAITON_SERVED_MODEL_NAME", "ornith"),
        "--trust-remote-code",
        "--language-model-only",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "8192",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "1",
        "--kv-cache-dtype",
        "auto",
        "--kv-cache-memory-bytes",
        "3G",
        "--load-format",
        "safetensors",
        "--no-enable-prefix-caching",
        "--reasoning-parser",
        "qwen3",
        "--generation-config",
        "vllm",
        "--enforce-eager",
        "--speculative-config",
        json.dumps(
            {
                "method": "dflash",
                "model": str(model / "dflash-draft"),
                "num_speculative_tokens": 16,
                "attention_backend": "ROCM_ATTN",
            },
            separators=(",", ":"),
        ),
        "--host",
        "0.0.0.0",
        "--port",
        os.environ.get("PAITON_PORT", "8000"),
    ]
    command.extend(extra_args or [])
    return command


def main() -> None:
    _validate_runtime_environment()
    if sys.argv[1:] == ["--check-runtime"]:
        print("Qualified Paiton Ornith 1.5 runtime detected.")
        return
    command = build_server_command(sys.argv[1:])
    print("Starting the pinned Paiton Ornith 1.5 RDNA4 server:", flush=True)
    print("  " + shlex.join(command), flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
