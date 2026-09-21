"""Read metadata in the launcher; initialize HIP only in an isolated probe."""

from __future__ import annotations
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from dataclasses import replace
from .contracts import Environment, Model, PreparationError, digest


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fingerprint(path: Path) -> tuple[int, ...]:
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def inspect_environment(
    *, probe_gpu: bool = False, probe_environment: dict | None = None
) -> Environment:
    versions = []
    for name in (
        "paiton-vllm-plugin",
        "vllm",
        "torch",
        "triton",
        "transformers",
        "safetensors",
        "huggingface-hub",
        "rocm-sdk-core",
        "amd-aiter",
        "compressed-tensors",
    ):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = None
        versions.append((name, version))
    result = Environment(
        sys.executable,
        platform.python_version(),
        platform.system(),
        platform.machine(),
        tuple(versions),
        libc_version=platform.libc_ver()[1] or None,
    )
    if not probe_gpu:
        return result
    code = """import json,torch,ctypes
n=torch.cuda.device_count()
p=torch.cuda.get_device_properties(0) if n else None
hip=ctypes.c_int()
rc=ctypes.CDLL('libamdhip64.so').hipRuntimeGetVersion(ctypes.byref(hip))
if rc:raise RuntimeError('HIP runtime version query failed')
print(json.dumps(dict(gpu_arch=getattr(p,'gcnArchName','').split(':')[0] or None,gpu_name=getattr(p,'name',None),gpu_count=n,hip_version=torch.version.hip,torch_cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,hip_runtime_version=hip.value)))"""
    try:
        run = subprocess.run(
            [sys.executable, "-c", code],
            env=probe_environment,
            capture_output=True,
            text=True,
            timeout=90,
            check=True,
        )
        values = json.loads(run.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise PreparationError(
            "gpu-probe",
            "The isolated ROCm probe failed. Check the selected environment, device permissions and visibility variables.",
        ) from exc
    return replace(result, **values)


def local_model(
    reference: str,
    model_contract: dict,
    *,
    offline: bool,
    metadata_only: bool = False,
    download_dir: str | None = None,
) -> Path:
    path = Path(reference).expanduser()
    if path.is_dir():
        return path.resolve()
    if reference != model_contract["repository"]:
        raise PreparationError(
            "model-reference",
            "Use an existing model directory or a repository ID listed by paiton models.",
        )
    try:
        from huggingface_hub import snapshot_download

        resolved = snapshot_download(
            repo_id=reference,
            revision=model_contract["revision"],
            allow_patterns=list(model_contract["files"]),
            local_files_only=offline or metadata_only,
            cache_dir=str(Path(download_dir).expanduser().resolve())
            if download_dir
            else None,
        )
    except ImportError as exc:
        raise PreparationError(
            "model-cache",
            "huggingface_hub is unavailable; supply an existing local model directory.",
        ) from exc
    except Exception as exc:
        raise PreparationError(
            "model-cache",
            "The pinned checkpoint is absent from the configured Hugging Face cache. Supply a local directory or prepare online explicitly.",
        ) from exc
    return Path(resolved).resolve()


def inspect_model(path: Path, contract: dict, *, verify_weights: bool = False) -> Model:
    # Extra loader-discovered files could change execution despite a matching
    # config hash. Maintainer provenance receipts are never read by the engine.
    unexpected = [
        p.name
        for p in path.iterdir()
        if p.is_file()
        and p.suffix
        in (".json", ".safetensors", ".bin", ".pt", ".gguf", ".model", ".txt", ".jinja")
        and p.name not in contract["files"]
        and not p.name.endswith("-receipt.json")
    ]
    if unexpected:
        raise PreparationError(
            "model-inventory",
            "Unqualified model/loader files are present: "
            + ", ".join(sorted(unexpected)),
        )
    if contract.get("identification") == "gguf-files":
        # The selected GGUF and projector carry their own metadata. The trusted
        # execution bundle supplies the released vLLM configuration/tokenizer.
        config_digest = digest(contract["files"])
    else:
        try:
            config = json.loads((path / "config.json").read_text())
        except (OSError, ValueError) as exc:
            raise PreparationError(
                "model-config",
                "A readable config.json is required in the supplied model directory.",
            ) from exc
        config_digest = contract["files"]["config.json"]["sha256"]
        if sha256(path / "config.json") != config_digest:
            raise PreparationError(
                "model-identity",
                "This config is not the narrowly qualified checkpoint. Matching architecture names alone is insufficient.",
            )
        if config.get("auto_map"):
            raise PreparationError(
                "remote-code",
                "Remote model code is not supported by this execution package.",
            )
    records = []
    for name, record in contract["files"].items():
        candidate = path / name
        if not candidate.is_file():
            raise PreparationError(
                "model-missing", f"Missing original checkpoint file: {name}"
            )
        before = fingerprint(candidate)
        if before[2] != record["size_bytes"]:
            raise PreparationError(
                "model-identity", f"Checkpoint size mismatch: {name}"
            )
        is_weight = name.endswith((".safetensors", ".gguf"))
        if verify_weights or not is_weight:
            if sha256(candidate) != record["sha256"]:
                raise PreparationError(
                    "model-identity",
                    f"Checkpoint SHA256 mismatch: {name}; the source file was not modified.",
                )
            if fingerprint(candidate) != before:
                raise PreparationError(
                    "model-race", f"Checkpoint changed during verification: {name}"
                )
        records.append((name, record["sha256"], *before))
    return Model(
        str(path),
        contract["repository"],
        contract["revision"],
        contract["storage_format"],
        config_digest,
        verify_weights,
        tuple(records),
    )


def validate_source(model: Model) -> None:
    if not model.verified:
        raise PreparationError(
            "model-unverified", "Launch requires verified checkpoint identity."
        )
    for name, expected, *record in model.files:
        if fingerprint(Path(model.path) / name) != tuple(record):
            raise PreparationError(
                "model-changed",
                f"Source file changed after preparation: {name}. Resolve a new plan.",
            )
