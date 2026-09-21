"""Shared API example for a direct Python caller or a future Studio worker.

Run in the existing compatible environment. prepare() does not start an engine;
launch() replaces this process, so a GUI should use a dedicated serving worker.
"""

import argparse
import json
from pathlib import Path
from paiton_vllm_plugin.execution import (
    inspect_environment,
    resolve,
    prepare,
    launch,
    lock_record,
)
from paiton_vllm_plugin.execution.cache import write_json_immutable


def prepare_model(model, profile, bundle, cache_dir, arguments, *, offline=False):
    resolution = resolve(
        model, arguments, inspect_environment(), profile=profile, offline=offline
    )
    return prepare(resolution, bundle=bundle, cache_dir=cache_dir, offline=offline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--bundle")
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--write-lock", type=Path)
    options, arguments = parser.parse_known_args()
    plan = prepare_model(
        options.model,
        options.profile,
        options.bundle,
        options.cache_dir,
        arguments,
        offline=options.offline,
    )
    if options.write_lock:
        write_json_immutable(options.write_lock, lock_record(plan))
    if options.serve:
        launch(plan, arguments, cache_dir=options.cache_dir)
    else:
        print(json.dumps(plan.as_dict(), indent=2))


if __name__ == "__main__":
    main()
