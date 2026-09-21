#!/usr/bin/env python3
"""Maintainer-only manifest generation from an explicit reviewed file allowlist.

No compiler, dependency installation or download is invoked. The spec's files
mapping pins the approved input bytes. A completed manifest must subsequently
be pinned in the installed catalogue; this tool is not a trust authority.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paiton_vllm_plugin.artifact_manifest import validate_execution_package, verify_file
from paiton_vllm_plugin.execution.contracts import digest
from paiton_vllm_plugin.execution.inspection import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    root = args.root.resolve(strict=True)
    for relative, expected in spec["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or str(path) != relative:
            parser.error("Unsafe allowlist path")
        verify_file(root / path, expected)
    result = dict(spec)
    result["files"] = {
        name: {
            **record,
            "size_bytes": (root / name).stat().st_size,
            "sha256": sha256(root / name),
        }
        for name, record in sorted(spec["files"].items())
    }
    output = root / "execution.json"
    if output.exists() and json.loads(output.read_text()) != result:
        parser.error("Refusing to replace an existing different execution manifest")
    if not args.verify and not output.exists():
        output.write_text(json.dumps(result, indent=2) + "\n")
    if not output.is_file():
        parser.error("Manifest is absent; build it before verification")
    validate_execution_package(
        root,
        result,
        expected_arch=result["target"]["arch"],
        expected_tp_size=result["tp_size"],
    )
    print(
        json.dumps(
            {
                "validated": str(root),
                "package_digest": digest(result),
                "files": len(result["files"]),
            }
        )
    )


if __name__ == "__main__":
    main()
