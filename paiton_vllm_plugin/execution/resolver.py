"""Deterministic selection with an explicit explanation, shared by CLI and Python."""

from pathlib import Path
from .arguments import BOOLS, ENGINE, SAFE_VALUES, inspect_arguments, option_tokens
from .catalogue import catalogue, package_digest
from .contracts import Environment, PreparationError, Resolution, canonical, digest
from .inspection import inspect_model, local_model


def check_environment(
    environment: Environment, required: dict, *, require_gpu: bool
) -> None:
    errors = []
    for name in ("system", "machine"):
        if getattr(environment, name) != required[name]:
            errors.append(
                f"{name}: need {required[name]}, found {getattr(environment, name)}"
            )
    if ".".join(environment.python_version.split(".")[:2]) != required["python"]:
        errors.append("Python " + required["python"] + " required")
    versions = dict(environment.versions)
    for name, value in required["versions"].items():
        if versions.get(name) != value:
            errors.append(
                f"{name}: need {value}, found {versions.get(name) or 'not installed'}"
            )
    if required.get("libc_min"):
        try:
            libc = tuple(int(x) for x in (environment.libc_version or "").split("."))
        except ValueError:
            libc = ()
        if libc < tuple(map(int, required["libc_min"].split("."))):
            errors.append("glibc " + required["libc_min"] + " or newer required")
    if require_gpu:
        if environment.gpu_arch != required["gpu_arch"]:
            errors.append("selected GPU must be " + required["gpu_arch"])
        if environment.gpu_count != required["gpu_count"]:
            errors.append("select exactly one visible GPU")
        for name in ("hip_version", "torch_cxx11_abi", "hip_runtime_version"):
            if name in required and getattr(environment, name) != required[name]:
                errors.append(name + " differs from the qualified native ABI")
    if errors:
        raise PreparationError(
            "environment",
            "; ".join(errors)
            + ". Activate a compatible existing environment; Paiton will not install or replace it.",
        )


def resolve(
    reference: str,
    arguments: list[str],
    environment: Environment,
    *,
    profile: str | None,
    offline: bool = False,
    metadata_only: bool = False,
    require_gpu: bool = False,
) -> Resolution:
    values, config_files = inspect_arguments(arguments)
    cat = catalogue()
    candidates = []
    rejected = []
    for package in cat["packages"]:
        if profile not in package["profiles"]:
            rejected.append(
                package["id"] + ": an explicit qualified profile is required"
            )
            continue
        try:
            check_environment(environment, package["runtime"], require_gpu=require_gpu)
            revision = package["model"]["revision"]
            if any(
                values.get(key, revision) != revision
                for key in ("revision", "tokenizer-revision")
            ):
                raise PreparationError(
                    "revision",
                    "Requested revision differs from the qualified checkpoint.",
                )
            path = local_model(
                reference,
                package["model"],
                offline=offline,
                metadata_only=metadata_only,
                download_dir=values.get("download-dir"),
            )
            model = inspect_model(path, package["model"])
            candidates.append((package, model))
        except PreparationError as exc:
            rejected.append(package["id"] + ": " + str(exc))
    if len(candidates) != 1:
        detail = (
            "; ".join(rejected)
            if not candidates
            else "several non-equivalent packages match"
        )
        profiles = ", ".join(k for p in cat["packages"] for k in p["profiles"])
        raise PreparationError(
            "resolution", detail + ". Available explicit profiles: " + profiles
        )
    package, model = candidates[0]
    chosen = package["profiles"][profile]
    if "tokenizer" in values and Path(values["tokenizer"]).resolve() != Path(
        model.path
    ):
        raise PreparationError(
            "tokenizer", "This package requires the original checkpoint tokenizer."
        )
    if values.get("speculative-config") is not None:
        raise PreparationError(
            "speculation",
            "This implemented profile is non-speculative; no draft will be substituted.",
        )
    if "quantization" in values or (
        "optimization-level" in values
        and "optimization-level" not in chosen["requirements"]
    ):
        raise PreparationError(
            "execution-setting",
            "Quantization/optimization overrides are not qualified for this profile.",
        )
    if values.get("load-format", "auto") not in (
        "auto",
        "safetensors",
        chosen["requirements"].get("load-format"),
    ):
        raise PreparationError("load-format", "Use the original safetensors loader.")
    if (
        values.get("enforce-eager") and "enforce-eager" not in chosen["requirements"]
    ) or ("block-size" in values and "block-size" not in chosen["requirements"]):
        raise PreparationError(
            "execution-setting",
            "Eager execution and block-size overrides have not been qualified for this profile.",
        )
    if "profiler-config" in values:
        config = values["profiler-config"]
        if (
            not isinstance(config, dict)
            or config.get("profiler") != "torch"
            or set(config)
            - {
                "profiler",
                "torch_profiler_dir",
                "torch_profiler_with_stack",
                "torch_profiler_record_shapes",
            }
        ):
            raise PreparationError(
                "profiler",
                "Only the existing torch profiler with local trace output is qualified.",
            )
    settings = dict(chosen["settings"])
    defaults = []
    for key, value in settings.items():
        if key not in values:
            defaults.extend(option_tokens(key, value))
    settings.update(
        {k: v for k, v in values.items() if k in ENGINE or k in BOOLS or k in settings}
    )
    for key, expected in chosen["requirements"].items():
        actual = settings.get(key)
        if actual != expected and str(actual) != str(expected):
            raise PreparationError(
                "profile-conflict",
                f"--{key}={actual!r} conflicts with profile requirement {expected!r}.",
            )
        settings[key] = expected
    for key, (lo, hi) in chosen["bounds"].items():
        try:
            value = (
                float(settings[key]) if isinstance(lo, float) else int(settings[key])
            )
        except (TypeError, ValueError) as exc:
            raise PreparationError(
                "profile-bound", f"--{key} requires a numeric value in [{lo}, {hi}]."
            ) from exc
        if not lo <= value <= hi:
            raise PreparationError("profile-bound", f"--{key} must be in [{lo}, {hi}].")
        settings[key] = value
    # Only effective engine settings enter a plan/lock, never API keys/HF tokens.
    for key in SAFE_VALUES:
        settings.pop(key, None)
    return Resolution(
        package["id"],
        package_digest(package),
        digest(cat),
        profile,
        package["adapter"],
        model,
        environment,
        canonical(settings),
        tuple(defaults),
        tuple(
            [
                chosen["description"],
                "Exact checkpoint file identity is required.",
                *rejected,
            ]
        ),
        config_files,
    )
