"""Reference-only bounded loading; no model or kernel arithmetic changes."""
import os
if os.environ.get("REFERENCE_SAFETENSORS_PREAD") == "1":
    import safetensors
    _original_safe_open = safetensors.safe_open
    def _pread_safe_open(*args, **kwargs):
        kwargs.setdefault("backend", "pread")
        return _original_safe_open(*args, **kwargs)
    safetensors.safe_open = _pread_safe_open
