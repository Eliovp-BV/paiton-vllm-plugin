# Rebuilding the image from locked inputs

The published image is one layer on top of the previous Qwen3.8 release image (pinned by digest in `build-context.lock.json`):
the plugin package at the locked plugin commit, three compiled runtime bundles, two compile-cache seeds, and six removed compatibility hooks.

1. Check out `paiton_vllm_plugin` at the commit named in `build-context.lock.json` (`plugin_package.source`).
2. Obtain the release artefacts directory (the compiled bundles under `runtime/` and the compile-cache seeds under `cache-seed/`,
   distributed with the release; their sources are not inputs).
3. `python3 prepare_image_context.py --plugin-src <checkout> --artifacts <artefacts> --out <new dir>` verifies every file against the lock
   and lays out `Dockerfile` + `staging/`.
4. `docker build -t <your tag> <new dir>`; the result must pass the same release scan and the start tests in `release-audit.json`.
