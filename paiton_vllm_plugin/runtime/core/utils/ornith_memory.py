"""Manifest-driven VRAM preflight for Ornith 1.5 MXFP4 artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from .ornith_mxfp4_loader import ornith_specs_from_manifest


_DTYPE_BYTES = {
    "bfloat16": 2,
    "float32": 4,
    "uint8": 1,
}
_MINIMUM_HEADROOM_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class OrnithMemoryEstimate:
    compiled_unbound_constants_bytes: int
    runtime_shell_parameters_bytes: int
    hybrid_cache_bytes: int
    activation_blob_bytes: int
    workspace_bytes: int
    allocator_headroom_bytes: int
    required_bytes: int
    available_bytes: int | None
    fits: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numel(values: object, *, label: str) -> int:
    if not isinstance(values, list) or any(
        not isinstance(dim, list)
        or len(dim) != 1
        or not isinstance(dim[0], int)
        or dim[0] <= 0
        for dim in values
    ):
        raise ValueError(f"{label} must have a positive static shape")
    return math.prod(dim[0] for dim in values)


def estimate_ornith_memory(
    manifest: Mapping[str, object],
    *,
    available_bytes: int | None = None,
    headroom_fraction: float = 0.10,
    hybrid_cache_reservation_bytes: int | None = None,
) -> OrnithMemoryEstimate:
    ornith_specs_from_manifest(manifest)
    if not 0.0 <= headroom_fraction <= 1.0:
        raise ValueError("headroom_fraction must be in [0,1]")
    contract = manifest["paiton_ornith15_contract"]
    interface = manifest["interface"]["tensors"]
    compiled = 0
    for record in interface:
        if "param" not in record.get("roles", ()) or record.get("binding") != "unbound":
            continue
        try:
            item_bytes = _DTYPE_BYTES[record["dtype"]]
        except KeyError as error:
            raise ValueError(f"unsupported Ornith dtype {record.get('dtype')!r}") from error
        compiled += _numel(
            record.get("shape_values"), label=f"constant {record.get('name')!r}"
        ) * item_bytes

    shell = 0
    for record in contract["runtime_shell_parameters"]:
        shape = record.get("shape")
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or dim <= 0 for dim in shape
        ):
            raise ValueError("Ornith runtime shell parameter has invalid shape")
        shell += math.prod(shape) * _DTYPE_BYTES[record["dtype"]]

    planning = manifest.get("memory_planning")
    required_fields = (
        "activation_blob_bytes",
        "compiler_owned_constant_bytes",
        "shared_workspace_bytes",
        "unique_workspace_bytes",
    )
    if not isinstance(planning, Mapping) or any(
        not isinstance(planning.get(field), int) or planning[field] < 0
        for field in required_fields
    ):
        raise ValueError("Ornith manifest lacks exact memory-planning metadata")
    activation = planning["activation_blob_bytes"]
    workspace = planning["shared_workspace_bytes"] + planning["unique_workspace_bytes"]
    compiled += planning["compiler_owned_constant_bytes"]

    kv_cache = (
        int(contract["max_context_length"])
        * int(contract["num_full_attention_layers"])
        * 2
        * int(contract["num_key_value_heads"])
        * int(contract["head_dim"])
        * 2
    )
    conv_numel = math.prod(contract["gdn_conv_state_shape"])
    recurrent_numel = math.prod(contract["gdn_recurrent_state_shape"])
    gdn_cache = int(contract["num_gdn_layers"]) * (
        conv_numel * 2 + recurrent_numel * 4
    )
    hybrid = max(kv_cache + gdn_cache, hybrid_cache_reservation_bytes or 0)
    subtotal = compiled + shell + hybrid + activation + workspace
    headroom = max(_MINIMUM_HEADROOM_BYTES, math.ceil(subtotal * headroom_fraction))
    required = subtotal + headroom
    fits = None if available_bytes is None else required <= available_bytes
    return OrnithMemoryEstimate(
        compiled_unbound_constants_bytes=compiled,
        runtime_shell_parameters_bytes=shell,
        hybrid_cache_bytes=hybrid,
        activation_blob_bytes=activation,
        workspace_bytes=workspace,
        allocator_headroom_bytes=headroom,
        required_bytes=required,
        available_bytes=available_bytes,
        fits=fits,
    )


def preflight_ornith_memory(
    manifest: Mapping[str, object],
    *,
    device: int = 0,
    hybrid_cache_reservation_bytes: int | None = None,
) -> OrnithMemoryEstimate:
    import torch

    with torch.cuda.device(device):
        available, _ = torch.cuda.mem_get_info()
    estimate = estimate_ornith_memory(
        manifest,
        available_bytes=int(available),
        hybrid_cache_reservation_bytes=hybrid_cache_reservation_bytes,
    )
    if not estimate.fits:
        raise MemoryError(
            "Ornith artifact memory preflight failed: "
            f"requires {estimate.required_bytes} bytes including headroom, "
            f"but only {estimate.available_bytes} bytes are free"
        )
    return estimate
