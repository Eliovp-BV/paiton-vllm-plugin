"""Typed, shared native preparation API; importing it initializes no GPU."""

from .contracts import Environment, Model, Resolution, LaunchPlan, PreparationError
from .inspection import inspect_environment, inspect_model
from .resolver import resolve
from .api import prepare, launch, lock_record, check_lock, validate_plan

__all__ = [
    "Environment",
    "Model",
    "Resolution",
    "LaunchPlan",
    "PreparationError",
    "inspect_environment",
    "inspect_model",
    "resolve",
    "prepare",
    "launch",
    "lock_record",
    "check_lock",
    "validate_plan",
]
