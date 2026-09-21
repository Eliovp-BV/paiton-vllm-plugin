"""Explicit native preparation and launch of the user's installed vLLM."""

from __future__ import annotations
import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="paiton", description=__doc__, allow_abbrev=False)
    p.add_argument(
        "--profile",
        help="Explicit qualified serving profile; inspect with paiton models",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        help="Reuse this existing checkpoint with a named serving preset",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect metadata only; no downloads, GPU initialization or engine",
    )
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Verify and prepare the same launch plan without starting vLLM",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Use local inputs and disable outbound runtime networking",
    )
    p.add_argument(
        "--bundle",
        type=Path,
        help="Verified local execution package, for unpublished/offline payloads",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        help="Paiton artifacts/plans only; model weights use their existing location",
    )
    p.add_argument(
        "--write-lock", type=Path, help="Write a new immutable reproducibility lock"
    )
    p.add_argument(
        "--lock",
        type=Path,
        help="Require exact checkpoint/package/effective-setting agreement",
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "doctor", help="Structured environment diagnostics without GPU initialization"
    )
    sub.add_parser(
        "models", help="Structured installed catalogue and local preparation status"
    )
    passthrough = sub.add_parser("vllm", add_help=False, allow_abbrev=False)
    passthrough.add_argument("arguments", nargs=argparse.REMAINDER)
    serve = sub.add_parser(
        "serve",
        add_help=False,
        allow_abbrev=False,
        help="Serve a named preset, checkpoint directory or repository with native Paiton",
    )
    serve.add_argument("arguments", nargs=argparse.REMAINDER)
    serve.add_argument("-h", "--help", dest="serve_help", nargs="?", const="")
    return p


def emit(stage: str, **values):
    print(
        json.dumps({"paiton_stage": stage, **values}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    from .execution.contracts import PreparationError

    try:
        from .execution.inspection import inspect_environment

        if args.command == "doctor":
            environment = inspect_environment()
            print(
                json.dumps(
                    {
                        "environment": asdict(environment),
                        "gpu_probe": "not requested",
                        "activation": "explicit only",
                        "runtime_modified": False,
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "models":
            from .execution.discovery import discover

            print(json.dumps(discover(args.cache_dir), indent=2))
            return 0
        forwarded = (["serve"] if args.command == "serve" else []) + args.arguments
        if args.command == "serve" and args.serve_help is not None:
            forwarded.append(
                "--help" + ("=" + args.serve_help if args.serve_help else "")
            )
        if not forwarded or forwarded[0] != "serve":
            p.error(
                "Use paiton [OPTIONS] serve MODEL [VLLM_ARGS...] (vllm serve also works)."
            )
        if any(t in ("--help", "-h") or t.startswith("--help=") for t in forwarded[1:]):
            if importlib.util.find_spec("vllm") is None:
                raise PreparationError(
                    "missing-vllm",
                    "vLLM is absent. Activate your existing vLLM environment to view its serve help.",
                )
            env = dict(os.environ)
            # Help is never a request to activate an inherited Paiton profile.
            for k in list(env):
                if k.startswith(("PAITON_", "RADIANCE_", "R4D_", "_PAITON_")):
                    env.pop(k)
            for k in ("VLLM_USE_PAITON_PLATFORM", "VLLM_PAITON_VANILLA_ROCM_PLATFORM"):
                env.pop(k, None)
            env["PAITON_PLUGIN_MODE"] = "off"
            os.execve(
                sys.executable,
                [sys.executable, "-m", "paiton_vllm_plugin.execution.help", *forwarded],
                env,
            )
        if len(forwarded) < 2 or forwarded[1].startswith("-"):
            p.error("A model directory or qualified repository ID must follow serve.")
        if args.dry_run and (args.prepare_only or args.write_lock):
            p.error("--dry-run cannot prepare or write a lockfile.")
        if args.offline:
            from .execution.offline import install

            install()
        from .execution.resolver import resolve
        from .execution.presets import select

        environment = inspect_environment()
        reference, profile = select(
            forwarded[1], args.profile, args.model_dir, cache_dir=args.cache_dir
        )
        resolution = resolve(
            reference,
            forwarded[2:],
            environment,
            profile=profile,
            offline=args.offline,
            metadata_only=args.dry_run,
            require_gpu=False,
        )
        emit(
            "Resolved",
            package=resolution.package_id,
            profile=resolution.profile,
            model=resolution.model.path,
            explanation=list(resolution.reasons),
            python=environment.python,
            vllm=dict(environment.versions).get("vllm"),
            effective_settings=json.loads(resolution.settings_json),
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "resolution": asdict(resolution),
                        "pending": [
                            "verify exact weight hashes",
                            "prepare/verify native bundle",
                            "probe selected GPU",
                            "load and execute native kernels",
                        ],
                    },
                    indent=2,
                )
            )
            return 0
        from .execution.api import prepare, launch, lock_record, check_lock
        from .execution.cache import write_json_immutable

        plan = prepare(
            resolution,
            cache_dir=args.cache_dir,
            bundle=args.bundle,
            offline=args.offline,
        )
        emit(
            "Prepared",
            bundle=plan.bundle,
            plan=plan.identity,
            weights_reused=True,
            gpu=plan.resolution.environment.gpu_name,
            gpu_arch=plan.resolution.environment.gpu_arch,
        )
        if args.lock:
            check_lock(plan, args.lock)
        if args.write_lock:
            write_json_immutable(args.write_lock, lock_record(plan))
        if args.model_dir is not None:
            from .execution.presets import remember

            remember(forwarded[1], plan, cache_dir=args.cache_dir)
        emit(
            "Validated",
            package_digest=resolution.package_digest,
            checkpoint_verified=plan.resolution.model.verified,
        )
        if args.prepare_only:
            print(json.dumps(plan.as_dict(), indent=2))
            return 0
        launch(plan, forwarded[2:], cache_dir=args.cache_dir)
        return 0
    except (PreparationError, ValueError, OSError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "paiton_stage": "Unresolved" if args.dry_run else "Failed",
                    "code": getattr(error, "code", "compatibility"),
                    "message": str(error),
                    "stock_fallback": False,
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
