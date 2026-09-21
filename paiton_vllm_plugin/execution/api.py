"""Shared Python/CLI preparation API. No shell or runtime package installation."""

from __future__ import annotations
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
from . import cache
from .catalogue import catalogue, package_by_id, package_digest
from .contracts import LaunchPlan, PreparationError, Resolution, digest
from .inspection import inspect_environment, inspect_model, sha256, validate_source
from ..activation import activate


def prepare(
    resolution: Resolution,
    *,
    cache_dir: str | Path | None = None,
    bundle: str | Path | None = None,
    offline: bool = False,
) -> LaunchPlan:
    package = package_by_id(resolution.package_id)
    if package_digest(package) != resolution.package_digest:
        raise PreparationError(
            "catalogue-changed", "Resolve again with the installed catalogue."
        )
    from .environment import validate as validate_environment

    validate_environment(package, resolution.profile, offline=offline)
    if offline:
        from .offline import install

        install()
    native = cache.prepare_package(
        cache.cache_root(cache_dir),
        package,
        source=Path(bundle) if bundle else None,
        offline=offline,
    )
    probe_env = dict(os.environ)
    if offline:
        from .offline import environment as offline_environment

        probe_env = offline_environment(native, probe_env)
    actual = inspect_environment(probe_gpu=True, probe_environment=probe_env)
    from .resolver import check_environment

    check_environment(actual, package["runtime"], require_gpu=True)
    resolution = replace(resolution, environment=actual)
    candidate = inspect_model(Path(resolution.model.path), package["model"])
    receipt = {"verification_schema": 1, "model": asdict(candidate)}
    receipt_path = (
        cache.cache_root(cache_dir) / "verified-models" / (digest(receipt) + ".json")
    )
    if receipt_path.is_file() and digest(
        json.loads(receipt_path.read_text())
    ) == digest(receipt):
        # A private cache receipt binds expected digests to inode/device/ctime/
        # mtime/size. Changed files or trusted metadata require a new full hash.
        model = replace(candidate, verified=True)
    else:
        model = inspect_model(
            Path(resolution.model.path), package["model"], verify_weights=True
        )
        if model.files != candidate.files:
            raise PreparationError(
                "model-race", "Checkpoint changed during preparation."
            )
        cache.write_json_immutable(receipt_path, receipt)
    chosen = package["profiles"][resolution.profile]
    from .serving import prepare as prepare_serving

    serving_model, serving_files = prepare_serving(
        cache.cache_root(cache_dir), package, model, native
    )
    environment = {
        key: value.replace("{bundle}", str(native)).replace("{model}", model.path)
        for key, value in chosen["environment"].items()
    }
    runtime_cache = (
        cache.cache_root(cache_dir).resolve()
        / "runtime"
        / digest(
            [
                resolution.package_digest,
                resolution.settings_json,
                resolution.model.revision,
            ]
        )
    )
    environment.update(
        VLLM_CACHE_ROOT=str(runtime_cache / "vllm"),
        TRITON_CACHE_DIR=str(runtime_cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(runtime_cache / "inductor"),
    )
    environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    environment["PAITON_PLUGIN_MODE"] = "native"
    environment["VLLM_NO_USAGE_STATS"] = "1"
    if offline:
        guarded = offline_environment(native, dict(os.environ))
        environment.update(
            {
                k: guarded[k]
                for k in (
                    "PAITON_OFFLINE_ACTIVE",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "HF_DATASETS_OFFLINE",
                    "HF_HUB_DISABLE_TELEMETRY",
                    "DO_NOT_TRACK",
                    "LD_PRELOAD",
                    "VLLM_HOST_IP",
                )
            }
        )
    for key in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "VLLM_PLUGINS",
    ):
        if key in os.environ:
            environment[key] = os.environ[key]
    return LaunchPlan(
        1,
        replace(resolution, model=model),
        str(native),
        tuple(sorted(environment.items())),
        offline,
        serving_model,
        serving_files,
    )


def lock_record(plan: LaunchPlan) -> dict:
    r = plan.resolution
    return {
        "lock_schema": 1,
        "catalogue_digest": r.catalogue_digest,
        "package_id": r.package_id,
        "package_digest": r.package_digest,
        "adapter": r.adapter,
        "adapter_version": dict(r.environment.versions).get("paiton-vllm-plugin"),
        "profile": r.profile,
        "model": {
            "repository": r.model.repository,
            "revision": r.model.revision,
            "files": {x[0]: x[1] for x in r.model.files},
        },
        "effective_settings": json.loads(r.settings_json),
        "runtime": dict(r.environment.versions),
        "python": r.environment.python_version,
        "native_abi": {
            "hip_runtime_version": r.environment.hip_runtime_version,
            "torch_cxx11_abi": r.environment.torch_cxx11_abi,
        },
    }


def check_lock(plan: LaunchPlan, path: Path) -> None:
    expected = json.loads(path.read_text())
    if expected != lock_record(plan):
        raise PreparationError(
            "lock-conflict",
            "Lockfile conflicts with the selected package, checkpoint, runtime or effective settings. Use an explicit new resolution and lockfile.",
        )


def validate_plan(plan: LaunchPlan) -> None:
    r = plan.resolution
    if digest(catalogue()) != r.catalogue_digest:
        raise PreparationError(
            "catalogue-changed", "Installed catalogue differs from the launch plan."
        )
    package = package_by_id(r.package_id)
    if package_digest(package) != r.package_digest or package["adapter"] != r.adapter:
        raise PreparationError(
            "package-changed", "Execution package/adapter differs from the launch plan."
        )
    from .resolver import check_environment

    check_environment(inspect_environment(), package["runtime"], require_gpu=False)
    cache.verify_package(Path(plan.bundle), package)
    validate_source(r.model)
    from .serving import validate as validate_serving

    validate_serving(plan, package)
    for path, expected in r.config_files:
        if sha256(Path(path)) != expected:
            raise PreparationError(
                "config-changed", "vLLM configuration changed after resolution."
            )


def launch(
    plan: LaunchPlan, arguments: list[str], *, cache_dir: str | Path | None = None
) -> None:
    validate_plan(plan)
    # Recheck argv against the immutable effective settings, not just a digest of
    # a config filename. Transport options/secrets stay only in this exec vector.
    from .resolver import resolve

    actual = resolve(
        plan.resolution.model.path,
        arguments,
        plan.resolution.environment,
        profile=plan.resolution.profile,
        offline=True,
        metadata_only=True,
        require_gpu=True,
    )
    if (
        actual.settings_json != plan.resolution.settings_json
        or actual.config_files != plan.resolution.config_files
    ):
        raise PreparationError(
            "arguments-changed", "Arguments differ from the validated launch plan."
        )
    env = dict(os.environ)
    from .adapters import PLATFORM

    activate("native", env, requires_platform=plan.resolution.adapter in PLATFORM)
    from .environment import validate as validate_environment

    validate_environment(
        package_by_id(plan.resolution.package_id),
        plan.resolution.profile,
        offline=plan.offline,
    )
    for key, value in plan.environment:
        if (
            key in env
            and env[key] != value
            and key
            not in (
                "VLLM_CACHE_ROOT",
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
                "LD_PRELOAD",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY",
                "DO_NOT_TRACK",
                "VLLM_NO_USAGE_STATS",
            )
        ):
            raise PreparationError(
                "environment-conflict",
                f"{key} conflicts with the selected explicit profile; unset it or resolve a compatible profile.",
            )
        env[key] = value
    plan_path = cache.cache_root(cache_dir) / "plans" / (plan.identity + ".json")
    cache.write_json_immutable(plan_path, plan.as_dict())
    env["PAITON_ACTIVE_PLAN"] = str(plan_path.resolve())
    env["PAITON_ACTIVE_PLAN_SHA256"] = plan.identity
    # The Python executable is fixed to this environment. No PATH-selected vllm,
    # shell, container, install step or private compiler is involved.
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        plan.serving_model or plan.resolution.model.path,
        *plan.resolution.defaults,
        *arguments,
    ]
    os.execve(sys.executable, command, env)
