"""Immutable values at the native launch boundary. No framework imports."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json


class PreparationError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Environment:
    python: str
    python_version: str
    system: str
    machine: str
    versions: tuple[tuple[str, str | None], ...]
    gpu_arch: str | None = None
    gpu_name: str | None = None
    gpu_count: int | None = None
    hip_version: str | None = None
    libc_version: str | None = None
    torch_cxx11_abi: bool | None = None
    hip_runtime_version: int | None = None


@dataclass(frozen=True)
class Model:
    path: str
    repository: str
    revision: str
    storage_format: str
    config_sha256: str
    verified: bool = False
    # (relative path, expected digest, device, inode, size, mtime_ns, ctime_ns)
    files: tuple[tuple, ...] = ()


@dataclass(frozen=True)
class Resolution:
    package_id: str
    package_digest: str
    catalogue_digest: str
    profile: str
    adapter: str
    model: Model
    environment: Environment
    settings_json: str
    defaults: tuple[str, ...]
    reasons: tuple[str, ...]
    config_files: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LaunchPlan:
    schema: int
    resolution: Resolution
    bundle: str
    environment: tuple[tuple[str, str], ...]
    offline: bool
    serving_model: str | None = None
    serving_files: tuple[tuple, ...] = ()

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def identity(self) -> str:
        return digest(self.as_dict())

    @classmethod
    def from_dict(cls, value: dict) -> "LaunchPlan":
        if value.get("schema") != 1:
            raise PreparationError(
                "plan-schema", "Unsupported launch-plan schema; resolve again."
            )
        r = dict(value["resolution"])
        m = dict(r["model"])
        m["files"] = tuple(tuple(x) for x in m["files"])
        r["model"] = Model(**m)
        e = dict(r["environment"])
        e["versions"] = tuple(tuple(x) for x in e["versions"])
        r["environment"] = Environment(**e)
        for key in ("defaults", "reasons"):
            r[key] = tuple(r[key])
        r["config_files"] = tuple(tuple(x) for x in r["config_files"])
        return cls(
            value["schema"],
            Resolution(**r),
            value["bundle"],
            tuple(tuple(x) for x in value["environment"]),
            value["offline"],
            value.get("serving_model"),
            tuple(tuple(x) for x in value.get("serving_files", ())),
        )
