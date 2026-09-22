"""Paiton image library integration; native weight handling, external diffusion runtime."""

__version__ = "0.1.0"

def load_engine(*args, **kwargs):
    from .runtime import ImageEngine
    return ImageEngine(*args, **kwargs)
