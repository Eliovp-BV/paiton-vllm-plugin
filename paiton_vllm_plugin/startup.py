"""Inert at normal interpreter startup; fail closed for explicit prepared runs."""

import os

if os.environ.get("PAITON_ACTIVE_PLAN"):
    import site

    _original = site.execsitecustomize

    def _after_site_paths():
        # A venv's .pth files run before system-site-packages is added. Wait for
        # site to finish paths, but run before any sitecustomize/runtime imports.
        site.execsitecustomize = _original
        try:
            from .execution.bootstrap import initialize

            initialize()
        except BaseException as error:
            import json

            os.write(
                2,
                (
                    json.dumps(
                        {
                            "paiton_stage": "Failed",
                            "code": getattr(error, "code", "startup"),
                            "error": str(error),
                            "phase": "process-start",
                            "stock_fallback": False,
                        }
                    )
                    + "\n"
                ).encode(),
            )
            # site.py normally swallows startup errors. Explicit launches must
            # stop instead of accidentally continuing with stock inference.
            os._exit(78)
        _original()

    site.execsitecustomize = _after_site_paths
