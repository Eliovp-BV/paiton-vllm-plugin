"""Zero-configuration server entrypoint for the public RDNA4 Qwen3.8 image."""

from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import shutil
import stat
import sys
import tempfile
import urllib.parse
import urllib.request
from runpy import run_path
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


BASE_MODEL = "amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16"
BASE_REVISION = "649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2"
BUNDLED_OVERLAY_DIR = "/opt/paiton/qwen38-overlay"
RELEASE_TAG = "paiton-qwen38-qronos-w4a16-gfx1201-v1.3.0"
RELEASE_METADATA_NAME = "paiton-release.json"
RELEASE_ASSET_BASE_URL = (
    "https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/"
    f"{RELEASE_TAG}"
)
RELEASE_METADATA_URL = f"{RELEASE_ASSET_BASE_URL}/{RELEASE_METADATA_NAME}"

RELEASE_ID = "paiton-qwen38-qronos-w4a16-gfx1201-v1.3.0"
RELEASE_METADATA_SIZE = 2_431
RELEASE_METADATA_SHA256 = (
    "6dddd10ebb5bd58ca292b50fa1190ecfa3895642bd553940c02de6b976e438ac"
)
CHECKPOINT_NAME = "model.safetensors"
CHECKPOINT_SIZE = 19_893_384_832
CHECKPOINT_SHA256 = "32190ba51af3e048f927b446f251a171e475cc91456a831e374709e74a8f0454"

ARTIFACT_NAME = (
    "Qwen3.8-27B-Quark-Qronos-INT4-W4A16_"
    "gfx1201_tp1_mt8192_ctx8192.so"
)
ARTIFACT_MANIFEST_NAME = ARTIFACT_NAME.removesuffix(".so") + ".manifest.json"
LM_HEAD_ARTIFACT_NAME = "benchmark_w4a16_n248320_k5120_gfx1201.so"
LM_HEAD_ARTIFACT_MANIFEST_NAME = LM_HEAD_ARTIFACT_NAME.removesuffix(
    ".so"
) + ".manifest.json"

EXPECTED_OVERLAY_FILES = {
    "artifact": {
        "file": ARTIFACT_NAME,
        "size_bytes": 5_834_720,
        "sha256": "be534b72dc47094e1d067594e8642174909f7ab223676daa29d0f9a95c5b55b3",
    },
    "artifact_manifest": {
        "file": ARTIFACT_MANIFEST_NAME,
        "size_bytes": 963_895,
        "sha256": "18bd30d71285356472f7482ed7ec45dbba6f733ad902f0059ad3e1364edb2f23",
    },
    "config": {
        "file": "config.json",
        "size_bytes": 16_298,
        "sha256": "4bb0ded506a18ccadc84b4c97426df66042309042a23d63109191fdff8434c2e",
    },
    "lm_head_artifact": {
        "file": LM_HEAD_ARTIFACT_NAME,
        "size_bytes": 385_736,
        "sha256": "6c5d36808b4785bd0cfb6f7d7f76f54b8e3498b5d9e92eb79a5fba5ecf866ae9",
    },
    "lm_head_artifact_manifest": {
        "file": LM_HEAD_ARTIFACT_MANIFEST_NAME,
        "size_bytes": 2_501,
        "sha256": "f35ae96417046b7a3c5fc524ae5cbf1f95af93bcc52519dc84690238673d97fa",
    },
}

BASE_RUNTIME_FILES = (
    "chat_template.jinja",
    "crc32.txt",
    "generation_config.json",
    "merges.txt",
    CHECKPOINT_NAME,
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)

BASE_RUNTIME_SHA256 = {
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "crc32.txt": "0193833c505e6e4b57530d50ce524f5c8d88120b2dd76257f1ccba14bf26a92d",
    "generation_config.json": "07f857aba5260b2ea2513f80de8062d086661f00eada0a3794964e665ba680f5",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "preprocessor_config.json": "957eb01d1ea45341a92d543daec95857a7cbeff5803834bc0603b27ba7b41b3f",
    "processor_config.json": "14932921ca485d458a04dafd8069fbb0a4505622a48208d19ed247115801385b",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
}

QUALIFIED_VLLM_VERSION = "0.28.0.dev0+eliovp.quark48606.g39bd959b5.rocm714"
QUALIFIED_VLLM_COMMIT = "g39bd959b5"
QUALIFIED_PACKAGES = {
    "huggingface_hub": "1.29.0",
    "numpy": "2.3.5",
    "safetensors": "0.8.0",
}
QUALIFIED_PYTHON = (3, 12)
QUALIFIED_TORCH = "2.12.0+rocm7.14.0"
QUALIFIED_HIP = "7.14.60850"
QUALIFIED_GPU_ARCH = "gfx1201"


class ReleaseModelError(RuntimeError):
    """Raised when a cached-base model cannot be assembled safely."""


@dataclass(frozen=True)
class FileFingerprint:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    links: int


@dataclass(frozen=True)
class ValidatedBaseModel:
    directory: Path
    fingerprints: Mapping[str, FileFingerprint]


def _installed_vllm_commit() -> str | None:
    """Read generated VCS metadata without importing the GPU-dependent package."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        return None
    roots = list(spec.submodule_search_locations)
    if len(roots) != 1:
        return None
    version_file = Path(roots[0]) / "_version.py"
    if not version_file.is_file():
        return None
    value = run_path(str(version_file)).get("__commit_id__")
    return value if isinstance(value, str) else None


def _snapshot_download(**kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _validate_runtime_environment() -> None:
    """Fail before any download when the host is outside the released stack."""
    from importlib.metadata import PackageNotFoundError, version

    errors: list[str] = []
    if shutil.which("vllm") is None:
        errors.append("the vllm command is missing")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        errors.append(
            f"host is {platform.system()}/{platform.machine()}, expected Linux/x86_64"
        )
    if sys.version_info[:2] != QUALIFIED_PYTHON:
        errors.append(
            f"Python is {sys.version_info.major}.{sys.version_info.minor}, "
            f"expected {QUALIFIED_PYTHON[0]}.{QUALIFIED_PYTHON[1]}"
        )
    try:
        actual_vllm = version("vllm")
    except PackageNotFoundError:
        errors.append("vllm is not installed")
    else:
        if actual_vllm != QUALIFIED_VLLM_VERSION:
            try:
                actual_commit = _installed_vllm_commit()
            except (OSError, RuntimeError, SyntaxError, ValueError) as error:
                errors.append(f"vllm VCS metadata could not be read: {error}")
            else:
                if actual_commit != QUALIFIED_VLLM_COMMIT:
                    errors.append(
                        f"vllm is {actual_vllm} at commit {actual_commit!r}, "
                        f"expected {QUALIFIED_VLLM_VERSION} or exact commit "
                        f"{QUALIFIED_VLLM_COMMIT}"
                    )
    for package, expected in QUALIFIED_PACKAGES.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            errors.append(f"{package} is not installed")
        else:
            if actual != expected:
                errors.append(f"{package} is {actual}, expected {expected}")

    try:
        import torch
    except ImportError as error:
        errors.append(f"torch could not be imported: {error}")
    else:
        if torch.__version__ != QUALIFIED_TORCH:
            errors.append(
                f"torch is {torch.__version__}, expected {QUALIFIED_TORCH}"
            )
        if torch.version.hip != QUALIFIED_HIP:
            errors.append(f"HIP is {torch.version.hip}, expected {QUALIFIED_HIP}")
        if not torch.cuda.is_available():
            errors.append("ROCm did not expose a GPU")
        else:
            try:
                gpu_arch = torch.cuda.get_device_properties(0).gcnArchName.split(
                    ":", 1
                )[0]
            except (AttributeError, RuntimeError) as error:
                errors.append(f"could not identify the ROCm GPU: {error}")
            else:
                if gpu_arch != QUALIFIED_GPU_ARCH:
                    errors.append(
                        f"GPU architecture is {gpu_arch}, expected {QUALIFIED_GPU_ARCH}"
                    )

    if errors:
        raise ReleaseModelError(
            "incompatible local runtime; use the qualified container or fix: "
            + "; ".join(errors)
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path) -> FileFingerprint:
    value = path.stat()
    return FileFingerprint(
        device=value.st_dev,
        inode=value.st_ino,
        size=value.st_size,
        mtime_ns=value.st_mtime_ns,
        ctime_ns=value.st_ctime_ns,
        links=value.st_nlink,
    )


def _regular_file(path: Path, description: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        file_stat = resolved.stat()
    except OSError as error:
        raise ReleaseModelError(f"missing {description}: {path}: {error}") from error
    if not stat.S_ISREG(file_stat.st_mode):
        raise ReleaseModelError(f"{description} is not a regular file: {path}")
    return resolved


def _safe_filename(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseModelError(f"invalid {description}: {value!r}")
    relative = Path(value)
    if relative.is_absolute() or relative.name != value or ".." in relative.parts:
        raise ReleaseModelError(f"unsafe {description}: {value!r}")
    return value


def _validate_release_metadata(path: Path) -> dict[str, Any]:
    metadata_path = _regular_file(path, RELEASE_METADATA_NAME)
    try:
        release = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseModelError(
            f"could not read release metadata {metadata_path}: {error}"
        ) from error
    if not isinstance(release, dict):
        raise ReleaseModelError("release metadata must be a JSON object")
    if release.get("schema_version") != 2:
        raise ReleaseModelError(
            f"unsupported release schema: {release.get('schema_version')!r}"
        )
    if release.get("release_id") != RELEASE_ID:
        raise ReleaseModelError(
            f"release ID mismatch: expected {RELEASE_ID!r}, "
            f"got {release.get('release_id')!r}"
        )
    if release.get("status") != "released":
        raise ReleaseModelError(
            f"release status must be 'released', got {release.get('status')!r}"
        )

    base = release.get("base_model")
    if not isinstance(base, Mapping):
        raise ReleaseModelError("release metadata is missing base_model")
    if base.get("repo_id") != BASE_MODEL or base.get("revision") != BASE_REVISION:
        raise ReleaseModelError("release metadata does not pin the qualified AMD base")
    if base.get("checkpoint") != {
        "file": CHECKPOINT_NAME,
        "size_bytes": CHECKPOINT_SIZE,
        "sha256": CHECKPOINT_SHA256,
    }:
        raise ReleaseModelError("release checkpoint declaration mismatch")

    files = release.get("files")
    if not isinstance(files, Mapping) or set(files) != set(EXPECTED_OVERLAY_FILES):
        raise ReleaseModelError("release metadata must declare exactly five overlay files")
    seen: set[str] = set()
    for key, expected in EXPECTED_OVERLAY_FILES.items():
        declaration = files.get(key)
        if not isinstance(declaration, Mapping):
            raise ReleaseModelError(f"release metadata is missing files.{key}")
        name = _safe_filename(declaration.get("file"), f"files.{key}.file")
        if name in seen:
            raise ReleaseModelError(f"duplicate release filename: {name}")
        seen.add(name)
        if dict(declaration) != expected:
            raise ReleaseModelError(f"release declaration mismatch for files.{key}")

    return release


def _verify_declared_file(path: Path, declaration: Mapping[str, Any]) -> Path:
    expected_name = _safe_filename(declaration.get("file"), "release filename")
    resolved = _regular_file(path, expected_name)
    if path.name != expected_name:
        raise ReleaseModelError(
            f"downloaded filename mismatch: expected {expected_name!r}, got {path.name!r}"
        )
    expected_size = declaration.get("size_bytes")
    if type(expected_size) is not int or resolved.stat().st_size != expected_size:
        raise ReleaseModelError(
            f"size mismatch for {expected_name}: expected {expected_size}, "
            f"got {resolved.stat().st_size}"
        )
    expected_sha256 = declaration.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ReleaseModelError(f"invalid SHA256 declaration for {expected_name}")
    actual_sha256 = _sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise ReleaseModelError(
            f"SHA256 mismatch for {expected_name}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    return resolved


def _strict_overlay_file(path: Path, description: str) -> Path:
    try:
        value = path.lstat()
    except OSError as error:
        raise ReleaseModelError(f"missing {description}: {path}: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(value.st_mode):
        raise ReleaseModelError(
            f"{description} must be a regular file, not a link: {path}"
        )
    return path


def _load_overlay_files(overlay_dir: Path) -> dict[str, Path]:
    try:
        entries = list(overlay_dir.iterdir())
    except OSError as error:
        raise ReleaseModelError(
            f"could not read Paiton overlay directory {overlay_dir}: {error}"
        ) from error
    expected_names = {RELEASE_METADATA_NAME} | {
        declaration["file"] for declaration in EXPECTED_OVERLAY_FILES.values()
    }
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        raise ReleaseModelError(
            "Paiton overlay must contain exactly the released files: expected "
            f"{sorted(expected_names)}, got {sorted(actual_names)}"
        )
    for entry in entries:
        _strict_overlay_file(entry, f"Paiton overlay file {entry.name}")
    metadata_path = overlay_dir / RELEASE_METADATA_NAME
    _verify_declared_file(
        metadata_path,
        {
            "file": RELEASE_METADATA_NAME,
            "size_bytes": RELEASE_METADATA_SIZE,
            "sha256": RELEASE_METADATA_SHA256,
        },
    )
    release = _validate_release_metadata(metadata_path)
    resolved = {
        RELEASE_METADATA_NAME: _regular_file(metadata_path, RELEASE_METADATA_NAME)
    }
    for key in EXPECTED_OVERLAY_FILES:
        declaration = release["files"][key]
        name = declaration["file"]
        resolved[name] = _verify_declared_file(overlay_dir / name, declaration)
    return resolved


def _overlay_cache_dir() -> Path:
    explicit = os.environ.get("PAITON_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser() / RELEASE_ID
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "paiton" / RELEASE_ID


def _download_exact_file(
    *, url: str, destination: Path, declaration: Mapping[str, Any]
) -> Path:
    try:
        return _verify_declared_file(destination, declaration)
    except ReleaseModelError:
        pass

    expected_size = declaration.get("size_bytes")
    if type(expected_size) is not int or expected_size < 0:
        raise ReleaseModelError(f"invalid download size for {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    temporary = temporary_dir / destination.name
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "paiton-vllm-plugin/0.1"}
        )
        with temporary.open("xb") as output, urllib.request.urlopen(
            request, timeout=120
        ) as response:
            headers = getattr(response, "headers", None)
            content_length = (
                headers.get("Content-Length") if headers is not None else None
            )
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError as error:
                    raise ReleaseModelError(
                        f"invalid Content-Length for {destination.name}: "
                        f"{content_length!r}"
                    ) from error
                if declared_length != expected_size:
                    raise ReleaseModelError(
                        f"Content-Length mismatch for {destination.name}: expected "
                        f"{expected_size}, got {declared_length}"
                    )
            downloaded = 0
            while block := response.read(1024 * 1024):
                downloaded += len(block)
                if downloaded > expected_size:
                    raise ReleaseModelError(
                        f"download exceeded released size for {destination.name}"
                    )
                output.write(block)
            if downloaded != expected_size:
                raise ReleaseModelError(
                    f"download size mismatch for {destination.name}: expected "
                    f"{expected_size}, got {downloaded}"
                )
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o644)
        _verify_declared_file(temporary, declaration)
        os.replace(temporary, destination)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return _verify_declared_file(destination, declaration)


def _download_overlay_files() -> dict[str, Path]:
    overlay_dir = _overlay_cache_dir()
    metadata_declaration = {
        "file": RELEASE_METADATA_NAME,
        "size_bytes": RELEASE_METADATA_SIZE,
        "sha256": RELEASE_METADATA_SHA256,
    }
    _download_exact_file(
        url=RELEASE_METADATA_URL,
        destination=overlay_dir / RELEASE_METADATA_NAME,
        declaration=metadata_declaration,
    )
    release = _validate_release_metadata(overlay_dir / RELEASE_METADATA_NAME)
    for key in EXPECTED_OVERLAY_FILES:
        declaration = release["files"][key]
        name = declaration["file"]
        encoded_name = urllib.parse.quote(name, safe="")
        _download_exact_file(
            url=f"{RELEASE_ASSET_BASE_URL}/{encoded_name}",
            destination=overlay_dir / name,
            declaration=declaration,
        )
    return _load_overlay_files(overlay_dir)


def _resolve_overlay_files() -> dict[str, Path]:
    explicit = os.environ.get("PAITON_OVERLAY_DIR")
    if explicit is not None:
        return _load_overlay_files(Path(explicit))
    bundled_env = os.environ.get("PAITON_BUNDLED_OVERLAY")
    if bundled_env is not None:
        return _load_overlay_files(Path(bundled_env))
    bundled = Path(BUNDLED_OVERLAY_DIR)
    if bundled.is_dir():
        return _load_overlay_files(bundled)
    return _download_overlay_files()


def _validate_base_model_dir(
    path: Path, *, explicit: bool
) -> ValidatedBaseModel | None:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        if explicit:
            raise ReleaseModelError(
                f"base model does not exist: {path}: {error}"
            ) from error
        return None
    if not resolved.is_dir():
        if explicit:
            raise ReleaseModelError(f"base model is not a directory: {path}")
        return None
    fingerprints: dict[str, FileFingerprint] = {}
    verify_checkpoint = os.environ.get("PAITON_VERIFY_BASE_SHA256", "1") != "0"
    for name in BASE_RUNTIME_FILES:
        candidate = resolved / name
        try:
            source = _regular_file(candidate, f"base model file {name}")
        except ReleaseModelError:
            if explicit:
                raise
            return None
        before = _fingerprint(source)
        if name == CHECKPOINT_NAME:
            if before.size != CHECKPOINT_SIZE:
                if explicit:
                    raise ReleaseModelError(
                        f"checkpoint size mismatch: expected {CHECKPOINT_SIZE}, "
                        f"got {before.size}"
                    )
                return None
            if verify_checkpoint:
                actual_sha256 = _sha256_file(source)
                if actual_sha256 != CHECKPOINT_SHA256:
                    if explicit:
                        raise ReleaseModelError(
                            "base checkpoint SHA256 mismatch: expected "
                            f"{CHECKPOINT_SHA256}, got {actual_sha256}"
                        )
                    return None
        else:
            actual_sha256 = _sha256_file(source)
            expected_sha256 = BASE_RUNTIME_SHA256[name]
            if actual_sha256 != expected_sha256:
                if explicit:
                    raise ReleaseModelError(
                        f"base model SHA256 mismatch for {name}: expected "
                        f"{expected_sha256}, got {actual_sha256}"
                    )
                return None
        after = _fingerprint(source)
        if before != after:
            if explicit:
                raise ReleaseModelError(
                    f"base model file changed while it was verified: {name}"
                )
            return None
        fingerprints[name] = after
    return ValidatedBaseModel(resolved, fingerprints)


def _is_local_cache_miss(error: Exception) -> bool:
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        return False
    return isinstance(error, LocalEntryNotFoundError)


def _resolve_base_model() -> ValidatedBaseModel | None:
    explicit_base = os.environ.get("PAITON_BASE_MODEL")
    if explicit_base is not None:
        candidate = Path(explicit_base)
        if candidate.exists():
            return _validate_base_model_dir(candidate, explicit=True)
        downloaded = Path(
            _snapshot_download(
                repo_id=explicit_base,
                repo_type="model",
                revision=BASE_REVISION,
                allow_patterns=list(BASE_RUNTIME_FILES),
            )
        )
        return _validate_base_model_dir(downloaded, explicit=True)

    if os.environ.get("PAITON_REUSE_HF_CACHE", "1") != "0":
        cache_directories: list[str | None] = []
        base_cache = os.environ.get("PAITON_BASE_HF_CACHE")
        if base_cache:
            cache_directories.append(base_cache)
        cache_directories.append(None)
        for cache_directory in cache_directories:
            cache_args: dict[str, Any] = {
                "repo_id": BASE_MODEL,
                "repo_type": "model",
                "revision": BASE_REVISION,
                "allow_patterns": list(BASE_RUNTIME_FILES),
                "local_files_only": True,
            }
            if cache_directory is not None:
                cache_args["cache_dir"] = cache_directory
            try:
                cached = Path(_snapshot_download(**cache_args))
            except Exception as error:
                if not _is_local_cache_miss(error):
                    raise
            else:
                validated = _validate_base_model_dir(cached, explicit=False)
                if validated is not None:
                    return validated

    downloaded = Path(
        _snapshot_download(
            repo_id=BASE_MODEL,
            repo_type="model",
            revision=BASE_REVISION,
            allow_patterns=list(BASE_RUNTIME_FILES),
        )
    )
    return _validate_base_model_dir(downloaded, explicit=True)


def _link_file(source: Path, destination: Path) -> str:
    resolved = _regular_file(source, f"link source for {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise ReleaseModelError(f"refusing to overwrite model file: {destination}")
    try:
        os.link(resolved, destination)
        return "hardlink"
    except OSError as error:
        if error.errno not in {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.EMLINK,
            errno.ENOTSUP,
        }:
            raise ReleaseModelError(
                f"could not link model file {destination.name}: {error}"
            ) from error
    os.symlink(resolved, destination)
    return "symlink"


def _build_reused_model_dir(
    base: ValidatedBaseModel, overlay_files: Mapping[str, Path]
) -> Path:
    runtime_root = Path(tempfile.mkdtemp(prefix="paiton-qwen38-reused-"))
    staging = runtime_root / ".staging"
    final = runtime_root / "model"
    try:
        staging.mkdir(mode=0o755)
        assembled_fingerprints: dict[str, FileFingerprint] = {}
        for name in BASE_RUNTIME_FILES:
            source = _regular_file(
                base.directory / name, f"validated base model file {name}"
            )
            expected = base.fingerprints[name]
            if _fingerprint(source) != expected:
                raise ReleaseModelError(
                    f"base model file changed before assembly: {name}"
                )
            method = _link_file(source, staging / name)
            after = _fingerprint(source)
            if method == "hardlink":
                if (
                    after.device != expected.device
                    or after.inode != expected.inode
                    or after.size != expected.size
                    or after.mtime_ns != expected.mtime_ns
                    or after.links != expected.links + 1
                ):
                    raise ReleaseModelError(
                        f"base model file changed while it was linked: {name}"
                    )
            elif after != expected:
                raise ReleaseModelError(
                    f"base model file changed while it was linked: {name}"
                )
            assembled = _regular_file(staging / name, f"assembled file {name}")
            if _fingerprint(assembled) != after:
                raise ReleaseModelError(
                    f"assembled base model link does not match its source: {name}"
                )
            assembled_fingerprints[name] = after

        expected_overlay_names = {RELEASE_METADATA_NAME} | {
            declaration["file"] for declaration in EXPECTED_OVERLAY_FILES.values()
        }
        if set(overlay_files) != expected_overlay_names:
            raise ReleaseModelError(
                "overlay file set changed before assembly: expected "
                f"{sorted(expected_overlay_names)}, got {sorted(overlay_files)}"
            )
        for name, source in sorted(overlay_files.items()):
            safe_name = _safe_filename(name, "overlay filename")
            _link_file(source, staging / safe_name)

        for name, expected in assembled_fingerprints.items():
            source = _regular_file(
                base.directory / name, f"validated base model file {name}"
            )
            assembled = _regular_file(staging / name, f"assembled file {name}")
            if (
                _fingerprint(source) != expected
                or _fingerprint(assembled) != expected
            ):
                raise ReleaseModelError(
                    f"base model file changed after assembly: {name}"
                )
            if name != CHECKPOINT_NAME:
                actual_sha256 = _sha256_file(assembled)
                if actual_sha256 != BASE_RUNTIME_SHA256[name]:
                    raise ReleaseModelError(
                        f"assembled base model SHA256 mismatch for {name}"
                    )
        checkpoint = _regular_file(
            staging / CHECKPOINT_NAME, "assembled base model checkpoint"
        )
        if checkpoint.stat().st_size != CHECKPOINT_SIZE:
            raise ReleaseModelError("linked checkpoint changed during assembly")

        release = _validate_release_metadata(staging / RELEASE_METADATA_NAME)
        for declaration in release["files"].values():
            _verify_declared_file(staging / declaration["file"], declaration)
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(runtime_root, ignore_errors=True)
        raise
    return final


def _resolve_reused_model() -> Path:
    # Validate or fetch the small release overlay before touching the 19.9 GB
    # checkpoint. A missing release must fail without wasting a model download.
    overlay_files = _resolve_overlay_files()
    base = _resolve_base_model()
    if base is None:
        raise ReleaseModelError("could not resolve the pinned AMD base model")
    return _build_reused_model_dir(base, overlay_files)


def _select_model() -> tuple[str, str | None]:
    if "PAITON_MODEL" in os.environ:
        return os.environ["PAITON_MODEL"], os.environ.get("PAITON_MODEL_REVISION")
    return str(_resolve_reused_model()), None


def build_server_command(extra_args: list[str] | None = None) -> list[str]:
    model, revision = _select_model()
    if Path(model).is_dir():
        lm_head_path = Path(model) / LM_HEAD_ARTIFACT_NAME
        if not lm_head_path.is_file():
            raise ReleaseModelError(f"missing released W4 LM-head artifact: {lm_head_path}")
        os.environ["PAITON_QWEN38_W4_LM_HEAD_SO"] = str(lm_head_path)
    os.environ["PAITON_QWEN38_W4_LM_HEAD"] = "1"
    os.environ["PAITON_QWEN38_W4_LM_HEAD_SHA256"] = EXPECTED_OVERLAY_FILES[
        "lm_head_artifact"
    ]["sha256"]
    os.environ["PAITON_QWEN38_SERIALIZED_EXTERNAL_GRAPH_CAPTURE"] = "1"
    os.environ["PAITON_DECODE_ATTENTION_AOT"] = "1"
    os.environ["PAITON_PREFILL_ATTENTION_AOT"] = "1"
    command = ["vllm", "serve", model]
    if revision and not Path(model).exists():
        command.extend(["--revision", revision])
    command.extend(
        [
            "--served-model-name",
            os.environ.get("PAITON_SERVED_MODEL_NAME", "qwen38"),
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
            "2G",
            "--load-format",
            "safetensors",
            "--no-enforce-eager",
            "-O2",
            "--compilation-config",
            '{"cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1,"cudagraph_num_of_warmups":1}',
            "--no-enable-prefix-caching",
            "--reasoning-parser",
            "qwen3",
            "--host",
            "0.0.0.0",
            "--port",
            os.environ.get("PAITON_PORT", "8000"),
        ]
    )
    command.extend(extra_args or [])
    return command


def main() -> None:
    _validate_runtime_environment()
    if sys.argv[1:] == ["--check-runtime"]:
        print("Qualified Paiton Qwen3.8 runtime detected.")
        return
    command = build_server_command(sys.argv[1:])
    print("Starting the pinned Paiton RDNA4 server:", flush=True)
    print("  " + shlex.join(command), flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
