"""Bounded inspection of the pinned vLLM 0.29/0.26 CLIs, never shell parsing.

vLLM remains the final parser. Unknown options are rejected until reviewed, rather
than treating them as safe overrides. Original CLI tokens are retained for exec.
"""

from __future__ import annotations
import json
from pathlib import Path
from .contracts import PreparationError
from .inspection import sha256

ENGINE = {
    "dtype",
    "tensor-parallel-size",
    "pipeline-parallel-size",
    "max-model-len",
    "max-num-seqs",
    "max-num-batched-tokens",
    "kv-cache-dtype",
    "kv-cache-memory-bytes",
    "gpu-memory-utilization",
    "attention-backend",
    "mamba-cache-mode",
    "mamba-cache-dtype",
    "mamba-ssm-cache-dtype",
    "safetensors-load-strategy",
    "load-format",
    "quantization",
    "speculative-config",
    "compilation-config",
    "optimization-level",
    "tokenizer",
    "revision",
    "tokenizer-revision",
    "seed",
    "reasoning-parser",
    "tool-call-parser",
    "generation-config",
    "override-generation-config",
    "block-size",
    "profiler-config",
    "default-chat-template-kwargs",
    "moe-backend",
    "limit-mm-per-prompt",
    "mm-processor-kwargs",
}
SAFE_VALUES = {
    "shutdown-timeout",
    "port",
    "host",
    "served-model-name",
    "uvicorn-log-level",
    "api-key",
    "ssl-keyfile",
    "ssl-certfile",
    "ssl-ca-certs",
    "ssl-cert-reqs",
    "root-path",
    "allowed-origins",
    "allowed-methods",
    "allowed-headers",
    "response-role",
    "max-log-len",
    "chat-template-content-format",
    "hf-token",
    "download-dir",
    "config",
}
BOOLS = {
    "enable-prefix-caching",
    "enable-chunked-prefill",
    "language-model-only",
    "async-scheduling",
    "enforce-eager",
    "enable-auto-tool-choice",
    "disable-log-requests",
    "disable-log-stats",
    "enable-request-id-headers",
    "disable-fastapi-docs",
    "allow-credentials",
    "enable-prompt-tokens-details",
}
ALIASES = {
    "-tp": "tensor-parallel-size",
    "-pp": "pipeline-parallel-size",
    "-q": "quantization",
    "-O": "optimization-level",
}
FORBIDDEN = {
    "trust-remote-code",
    "hf-overrides",
    "model",
    "model-impl",
    "worker-cls",
    "worker-extension-cls",
    "model-loader-extra-config",
    "code-revision",
    "lora-modules",
    "enable-lora",
    "prompt-adapters",
    "chat-template",
}


def inspect_tokens(tokens: list[str]) -> dict:
    values = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        i += 1
        if token.startswith("-O") and token != "-O":
            token = "--optimization-level=" + token[2:].lstrip("=")
        key, equal, value = token.partition("=")
        key = ALIASES.get(
            key, key[2:].replace("_", "-") if key.startswith("--") else key
        )
        if key in FORBIDDEN:
            raise PreparationError(
                "unsupported-setting",
                f"--{key} changes the qualified model/loader contract and is not supported.",
            )
        negative = key.startswith("no-") and key[3:] in BOOLS
        if key in BOOLS or negative:
            if equal:
                raise PreparationError(
                    "argument",
                    f"Use the explicit --{key} boolean switch without =VALUE.",
                )
            values[key[3:] if negative else key] = not negative
            continue
        if key not in ENGINE | SAFE_VALUES:
            raise PreparationError(
                "unsupported-argument",
                f"{token!r} has not been qualified for this vLLM build. No launch occurred.",
            )
        if not equal:
            if i == len(tokens) or tokens[i].startswith("--"):
                raise PreparationError("argument", f"Missing value for --{key}")
            value = tokens[i]
            i += 1
        if key == "served-model-name":
            # vLLM accepts one or more names; these never affect artifacts.
            while i < len(tokens) and not tokens[i].startswith("-"):
                i += 1
        if key in (
            "compilation-config",
            "speculative-config",
            "override-generation-config",
            "profiler-config",
            "default-chat-template-kwargs",
            "limit-mm-per-prompt",
            "mm-processor-kwargs",
        ):
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise PreparationError(
                    "argument", f"--{key} requires valid JSON."
                ) from exc
        values[key] = value
    return values


def inspect_arguments(tokens: list[str]) -> tuple[dict, tuple[tuple[str, str], ...]]:
    cli = inspect_tokens(tokens)
    configs = [i for i, t in enumerate(tokens) if t == "--config"]
    if any(t.startswith("--config=") or t == "--config_file" for t in tokens):
        raise PreparationError(
            "config", "The pinned parser requires --config FILE.yaml as two tokens."
        )
    if len(configs) > 1:
        raise PreparationError("config", "Use exactly one vLLM configuration file.")
    if not configs:
        return cli, ()
    path = Path(cli["config"]).expanduser().resolve(strict=True)
    if path.suffix not in (".yml", ".yaml"):
        raise PreparationError("config", "vLLM configuration must be YAML.")
    try:
        import yaml
    except ImportError as exc:
        raise PreparationError(
            "config", "The existing vLLM environment needs PyYAML to read --config."
        ) from exc
    try:

        class UniqueLoader(yaml.SafeLoader):
            pass

        def mapping(loader, node):
            pairs = loader.construct_pairs(node, deep=True)
            out = {}
            for k, v in pairs:
                if not isinstance(k, str) or k in out:
                    raise ValueError("duplicate or non-string key")
                out[k] = v
            return out

        UniqueLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping
        )
        value = yaml.load(path.read_text(), Loader=UniqueLoader)
    except (yaml.YAMLError, ValueError, OSError) as exc:
        raise PreparationError(
            "config",
            "Cannot safely interpret this YAML configuration with the qualified parser.",
        ) from exc
    if not isinstance(value, dict):
        raise PreparationError("config", "Expected a YAML mapping.")
    flattened = []
    normalized_keys = set()
    for key, v in value.items():
        normalized = key.replace("_", "-")
        if normalized in normalized_keys:
            raise PreparationError(
                "config", "Duplicate normalized configuration key: " + normalized
            )
        normalized_keys.add(normalized)
        if normalized == "config":
            raise PreparationError(
                "config", "Nested configuration files are not supported."
            )
        if isinstance(v, bool):
            if normalized not in BOOLS:
                raise PreparationError("config", f"Unqualified boolean key: {key}")
            flattened.append("--" + ("" if v else "no-") + normalized)
        elif isinstance(v, dict):
            flattened.extend(["--" + normalized, json.dumps(v)])
        elif isinstance(v, list):
            if normalized != "served-model-name" or not v:
                raise PreparationError("config", f"Unqualified list key: {key}")
            flattened.extend(["--" + normalized, *map(str, v)])
        elif v is None:
            raise PreparationError("config", f"Null values are not qualified: {key}")
        else:
            flattened.extend(["--" + normalized, str(v)])
    resolved = inspect_tokens(flattened)
    resolved.update(cli)
    return resolved, ((str(path), sha256(path)),)


def option_tokens(key: str, value) -> list[str]:
    if key == "optimization-level":
        return ["-O" + str(value)]
    if isinstance(value, bool):
        return ["--" + ("" if value else "no-") + key]
    return [
        "--" + key,
        json.dumps(value, separators=(",", ":"))
        if isinstance(value, (dict, list))
        else str(value),
    ]
