"""Strict streaming loader for Ornith 1.5 Quark MXFP4 constants."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, Mapping

import torch


_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<component>weight|weight_scale)$"
)
_SHARED_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.shared_expert\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<component>weight|weight_scale)$"
)


def _exact_shape(record: Mapping[str, object]) -> tuple[int, ...] | None:
    values = record.get("shape_values")
    if not isinstance(values, list) or any(
        not isinstance(dim, list)
        or len(dim) != 1
        or not isinstance(dim[0], int)
        or dim[0] <= 0
        for dim in values
    ):
        return None
    return tuple(dim[0] for dim in values)


@dataclass(frozen=True)
class OrnithTensorSpec:
    source_name: str
    target_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    add_one: bool = False


def _direct_specs(
    *, num_layers: int, layer_types: tuple[str, ...]
) -> tuple[OrnithTensorSpec, ...]:
    result: list[OrnithTensorSpec] = []
    for index, layer_type in enumerate(layer_types):
        source = f"model.language_model.layers.{index}"
        target = f"layers_{index}"
        result.extend(
            (
                OrnithTensorSpec(
                    f"{source}.input_layernorm.weight",
                    f"{target}_input_layernorm_weight",
                    (2048,),
                    torch.bfloat16,
                    True,
                ),
                OrnithTensorSpec(
                    f"{source}.post_attention_layernorm.weight",
                    f"{target}_post_attention_layernorm_weight",
                    (2048,),
                    torch.bfloat16,
                    True,
                ),
                OrnithTensorSpec(
                    f"{source}.mlp.gate.weight",
                    f"{target}_mlp_gate_weight",
                    (256, 2048),
                    torch.bfloat16,
                ),
                OrnithTensorSpec(
                    f"{source}.mlp.shared_expert_gate.weight",
                    f"{target}_mlp_shared_expert_gate_weight",
                    (1, 2048),
                    torch.bfloat16,
                ),
            )
        )
        if layer_type == "linear_attention":
            base = f"{source}.linear_attn"
            out = f"{target}_linear_attn"
            result.extend(
                (
                    OrnithTensorSpec(
                        f"{base}.A_log", f"{out}_A_log", (32,), torch.bfloat16
                    ),
                    OrnithTensorSpec(
                        f"{base}.conv1d.weight",
                        f"{out}_conv1d",
                        (8192, 1, 4),
                        torch.bfloat16,
                    ),
                    OrnithTensorSpec(
                        f"{base}.dt_bias", f"{out}_dt_bias", (32,), torch.bfloat16
                    ),
                    OrnithTensorSpec(
                        f"{base}.norm.weight",
                        f"{out}_norm_weight",
                        (128,),
                        torch.bfloat16,
                    ),
                )
            )
            dense_shapes = {
                "in_proj_qkv": ((8192, 1024), (8192, 64)),
                "in_proj_z": ((4096, 1024), (4096, 64)),
                "in_proj_b": ((32, 1024), (32, 64)),
                "in_proj_a": ((32, 1024), (32, 64)),
                "out_proj": ((2048, 2048), (2048, 128)),
            }
            for name, (weight_shape, scale_shape) in dense_shapes.items():
                result.extend(
                    (
                        OrnithTensorSpec(
                            f"{base}.{name}.weight",
                            f"{out}_{name}_weight",
                            weight_shape,
                            torch.uint8,
                        ),
                        OrnithTensorSpec(
                            f"{base}.{name}.weight_scale",
                            f"{out}_{name}_weight_scale",
                            scale_shape,
                            torch.uint8,
                        ),
                    )
                )
        elif layer_type == "full_attention":
            base = f"{source}.self_attn"
            out = f"{target}_self_attn"
            result.extend(
                (
                    OrnithTensorSpec(
                        f"{base}.q_proj.weight",
                        f"{out}_q_proj_weight",
                        (8192, 2048),
                        torch.bfloat16,
                    ),
                    OrnithTensorSpec(
                        f"{base}.k_proj.weight",
                        f"{out}_k_proj_weight",
                        (512, 2048),
                        torch.bfloat16,
                    ),
                    OrnithTensorSpec(
                        f"{base}.v_proj.weight",
                        f"{out}_v_proj_weight",
                        (512, 2048),
                        torch.bfloat16,
                    ),
                    OrnithTensorSpec(
                        f"{base}.o_proj.weight",
                        f"{out}_o_proj_weight",
                        (2048, 4096),
                        torch.bfloat16,
                    ),
                    OrnithTensorSpec(
                        f"{base}.q_norm.weight",
                        f"{out}_q_norm_weight",
                        (256,),
                        torch.bfloat16,
                        True,
                    ),
                    OrnithTensorSpec(
                        f"{base}.k_norm.weight",
                        f"{out}_k_norm_weight",
                        (256,),
                        torch.bfloat16,
                        True,
                    ),
                )
            )
        else:
            raise ValueError(f"unsupported Ornith layer type {layer_type!r}")
    result.append(
        OrnithTensorSpec(
            "model.language_model.norm.weight",
            "norm_weight",
            (2048,),
            torch.bfloat16,
            True,
        )
    )
    return tuple(result)


def _moe_target_shapes(index: int) -> dict[str, tuple[tuple[int, ...], str]]:
    prefix = f"layers_{index}_mlp"
    return {
        f"{prefix}_experts_w13_weight": ((256, 1024, 1024), "uint8"),
        f"{prefix}_experts_w13_weight_scale": ((256, 1024, 64), "uint8"),
        f"{prefix}_experts_w2_weight": ((256, 2048, 256), "uint8"),
        f"{prefix}_experts_w2_weight_scale": ((256, 2048, 16), "uint8"),
        f"{prefix}_shared_expert_w13_weight": ((1, 1024, 1024), "uint8"),
        f"{prefix}_shared_expert_w13_weight_scale": ((1, 1024, 64), "uint8"),
        f"{prefix}_shared_expert_w2_weight": ((1, 2048, 256), "uint8"),
        f"{prefix}_shared_expert_w2_weight_scale": ((1, 2048, 16), "uint8"),
    }


def ornith_specs_from_manifest(manifest: Mapping[str, object]):
    contract = manifest.get("paiton_ornith15_contract")
    if not isinstance(contract, Mapping) or contract.get("version") not in (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ):
        raise ValueError("Ornith artifact requires contract version 1 through 11")
    version = contract.get("version")
    dflash_fields = (
        "dflash_aux_hidden_states",
        "dflash_aux_hidden_state_layers",
        "dflash_aux_hidden_state_outputs",
        "dflash_max_speculative_tokens",
        "dflash_state_index_widths",
        "dflash_full_attention_metadata",
        "dflash_full_attention_metadata_layers",
    )
    if version not in (10, 11):
        present = [field for field in dflash_fields if field in contract]
        if present:
            raise ValueError(
                f"Ornith contract version {version} cannot declare DFlash fields: "
                + ", ".join(present)
            )
    optimization_fields = (
        "grouped_mxfp4_prefill",
        "dense_mxfp4_prefill",
        "wave_mxfp4_decode",
        "merged_mxfp4_projections",
        "dot2_mxfp4_decode",
        "dot2_mxfp4_moe_decode",
        "fused_shared_moe_decode",
        "fused_shared_gate_decode",
    )
    if version == 1:
        present = [field for field in optimization_fields if field in contract]
        if present:
            raise ValueError(
                "Ornith contract version 1 cannot declare optimization fields: "
                + ", ".join(present)
            )
    else:
        if not isinstance(contract.get("grouped_mxfp4_prefill"), bool):
            raise ValueError(
                "Ornith contract version 2 through 11 requires boolean "
                "grouped_mxfp4_prefill"
            )
        if version == 2:
            present = [
                field
                for field in optimization_fields[1:]
                if field in contract
            ]
            if present:
                raise ValueError(
                    "Ornith contract version 2 cannot declare later optimization "
                    "fields: " + ", ".join(present)
                )
        else:
            if not isinstance(contract.get("dense_mxfp4_prefill"), bool):
                raise ValueError(
                    "Ornith contract version 3 through 11 requires boolean "
                    "dense_mxfp4_prefill"
                )
            if version == 3:
                present = [
                    field
                    for field in optimization_fields[2:]
                    if field in contract
                ]
                if present:
                    raise ValueError(
                        "Ornith contract version 3 cannot declare later optimization "
                        "fields: " + ", ".join(present)
                    )
            else:
                if not isinstance(contract.get("wave_mxfp4_decode"), bool):
                    raise ValueError(
                        "Ornith contract version 4 through 11 requires boolean "
                        "wave_mxfp4_decode"
                    )
                if contract.get("version") == 4:
                    if "merged_mxfp4_projections" in contract:
                        raise ValueError(
                            "Ornith contract version 4 cannot declare merged "
                            "MXFP4 projections"
                        )
                    present = [
                        field
                        for field in (
                            "dot2_mxfp4_decode",
                            "dot2_mxfp4_moe_decode",
                            "fused_shared_moe_decode",
                            "fused_shared_gate_decode",
                        )
                        if field in contract
                    ]
                    if present:
                        raise ValueError(
                            "Ornith contract version 4 cannot declare later "
                            "optimization fields: " + ", ".join(present)
                        )
                elif not isinstance(
                    contract.get("merged_mxfp4_projections"), bool
                ):
                    raise ValueError(
                        "Ornith contract version 5 through 11 requires boolean "
                        "merged_mxfp4_projections"
                    )
                elif (
                    contract["merged_mxfp4_projections"]
                    and not contract["dense_mxfp4_prefill"]
                ):
                    raise ValueError(
                        "Ornith merged MXFP4 projections require dense MXFP4 prefill"
                    )
                if version == 5:
                    present = [
                        field
                        for field in (
                            "dot2_mxfp4_decode",
                            "dot2_mxfp4_moe_decode",
                            "fused_shared_moe_decode",
                            "fused_shared_gate_decode",
                        )
                        if field in contract
                    ]
                    if present:
                        raise ValueError(
                            "Ornith contract version 5 cannot declare paired-dot "
                            "MXFP4 decode fields: " + ", ".join(present)
                        )
                elif version in (6, 7):
                    present = [
                        field
                        for field in (
                            "fused_shared_moe_decode",
                            "fused_shared_gate_decode",
                        )
                        if field in contract
                    ]
                    if present:
                        raise ValueError(
                            f"Ornith contract version {version} cannot declare "
                            "fused shared decode fields: " + ", ".join(present)
                        )
                    if not isinstance(contract.get("dot2_mxfp4_decode"), bool):
                        raise ValueError(
                            f"Ornith contract version {version} requires boolean "
                            "dot2_mxfp4_decode"
                        )
                    if version == 6:
                        if "dot2_mxfp4_moe_decode" in contract:
                            raise ValueError(
                                "Ornith contract version 6 cannot declare split "
                                "paired-dot MoE decode"
                            )
                        if contract["dot2_mxfp4_decode"] and not (
                            contract["wave_mxfp4_decode"]
                            and contract["merged_mxfp4_projections"]
                        ):
                            raise ValueError(
                                "Ornith paired-dot MXFP4 decode requires wave decode "
                                "and merged projections"
                            )
                    else:
                        if contract["dot2_mxfp4_decode"] and not contract[
                            "merged_mxfp4_projections"
                        ]:
                            raise ValueError(
                                "Ornith paired-dot MXFP4 projection decode requires "
                                "merged projections"
                            )
                        if not isinstance(
                            contract.get("dot2_mxfp4_moe_decode"), bool
                        ):
                            raise ValueError(
                                "Ornith contract version 7 requires boolean "
                                "dot2_mxfp4_moe_decode"
                            )
                        if contract["dot2_mxfp4_moe_decode"] and not contract[
                            "wave_mxfp4_decode"
                        ]:
                            raise ValueError(
                                "Ornith paired-dot MXFP4 MoE decode requires wave decode"
                            )
                elif version == 8:
                    if contract.get("fused_shared_moe_decode") is not True:
                        raise ValueError(
                            "Ornith contract version 8 requires fused shared MoE decode"
                        )
                    present = [
                        field
                        for field in (
                            "dot2_mxfp4_decode",
                            "dot2_mxfp4_moe_decode",
                            "fused_shared_gate_decode",
                        )
                        if field in contract
                    ]
                    if present:
                        raise ValueError(
                            "Ornith contract version 8 cannot declare rejected "
                            "paired-dot fields: " + ", ".join(present)
                        )
                    if not (
                        contract["wave_mxfp4_decode"]
                        and contract["grouped_mxfp4_prefill"]
                    ):
                        raise ValueError(
                            "Ornith fused shared MoE decode requires grouped prefill "
                            "and wave decode"
                        )
                elif version in (9, 10, 11):
                    if contract.get("fused_shared_moe_decode") is not True:
                        raise ValueError(
                            f"Ornith contract version {version} requires fused shared "
                            "MoE decode"
                        )
                    if contract.get("fused_shared_gate_decode") is not True:
                        raise ValueError(
                            f"Ornith contract version {version} requires fused shared "
                            "gate decode"
                        )
                    present = [
                        field
                        for field in (
                            "dot2_mxfp4_decode",
                            "dot2_mxfp4_moe_decode",
                        )
                        if field in contract
                    ]
                    if present:
                        raise ValueError(
                            f"Ornith contract version {version} cannot declare rejected "
                            "paired-dot fields: " + ", ".join(present)
                        )
                    if not (
                        contract["wave_mxfp4_decode"]
                        and contract["grouped_mxfp4_prefill"]
                    ):
                        raise ValueError(
                            "Ornith fused shared gate decode requires grouped prefill "
                            "and wave decode"
                        )
                    if version in (10, 11):
                        expected_layers = [2, 7, 12, 17, 23, 28, 33, 38]
                        expected_outputs = [
                            f"dflash_aux_hidden_state_layer_{layer:02d}"
                            for layer in expected_layers
                        ]
                        if contract.get("dflash_aux_hidden_states") is not True:
                            raise ValueError(
                                f"Ornith contract version {version} requires DFlash auxiliary "
                                "hidden states"
                            )
                        if (
                            contract.get("dflash_aux_hidden_state_layers")
                            != expected_layers
                        ):
                            raise ValueError(
                                f"Ornith contract version {version} requires exact DFlash "
                                f"auxiliary layers {expected_layers}"
                            )
                        if (
                            contract.get("dflash_aux_hidden_state_outputs")
                            != expected_outputs
                        ):
                            raise ValueError(
                                f"Ornith contract version {version} requires exact DFlash "
                                "auxiliary output names"
                            )
                        if contract.get("dflash_max_speculative_tokens") != 16:
                            raise ValueError(
                                f"Ornith contract version {version} requires exactly 16 "
                                "DFlash speculative tokens"
                            )
                        if contract.get("dflash_state_index_widths") != [1, 17]:
                            raise ValueError(
                                f"Ornith contract version {version} requires DFlash state "
                                "index widths [1, 17]"
                            )
                        if contract.get("gdn_conv_state_shape") != [19, 8192]:
                            raise ValueError(
                                f"Ornith contract version {version} requires expanded GDN "
                                "convolution state shape [19, 8192]"
                            )
                        if version == 11:
                            expected_metadata_layers = [
                                3, 7, 11, 15, 19, 23, 27, 31, 35, 39
                            ]
                            if (
                                contract.get("dflash_full_attention_metadata")
                                != "per_layer"
                            ):
                                raise ValueError(
                                    "Ornith contract version 11 requires DFlash "
                                    "per-layer full-attention metadata"
                                )
                            if (
                                contract.get(
                                    "dflash_full_attention_metadata_layers"
                                )
                                != expected_metadata_layers
                            ):
                                raise ValueError(
                                    "Ornith contract version 11 requires exact "
                                    "DFlash full-attention metadata layers "
                                    f"{expected_metadata_layers}"
                                )
    required = {
        "product_model_type": "ornith_1_5_qwen3_5_moe",
        "scope": "text-only",
        "tp_size": 1,
        "source_num_hidden_layers": 40,
        "max_batch_size": 1,
        "quark_format": "mxfp4_e2m1",
        "quark_group_size": 32,
        "quark_checkpoint_weight_layout": "Nx(K/2)_packed_e2m1_u8",
        "quark_checkpoint_scale_layout": "Nx(K/32)_ue8m0_u8",
        "quark_kernel_layout": "quark_mxfp4_e2m1_ue8m0_g32_v1",
        "num_experts": 256,
        "num_experts_per_token": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "zero_centered_norm_transform": "gamma=1+checkpoint_bf16",
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"Ornith contract {key} must be {expected!r}, got {contract.get(key)!r}"
            )
    num_layers = contract.get("num_hidden_layers")
    if not isinstance(num_layers, int) or not 4 <= num_layers <= 40:
        raise ValueError("Ornith compiled layer count must be in [4,40]")
    if version in (10, 11) and num_layers != 40:
        raise ValueError("Ornith DFlash artifacts require all 40 target layers")
    layer_types = tuple(
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(num_layers)
    )
    if contract.get("num_gdn_layers") != sum(x == "linear_attention" for x in layer_types):
        raise ValueError("Ornith GDN layer count does not match schedule")
    if contract.get("num_full_attention_layers") != sum(
        x == "full_attention" for x in layer_types
    ):
        raise ValueError("Ornith full-attention layer count does not match schedule")
    schedule = ",".join(layer_types)
    if contract.get("layer_schedule_sha256") != hashlib.sha256(
        schedule.encode()
    ).hexdigest():
        raise ValueError("Ornith layer schedule hash is invalid")
    expected_counts = {
        "quark_dense_linear_count": sum(x == "linear_attention" for x in layer_types) * 5,
        "quark_routed_expert_projection_count": num_layers * 256 * 3,
        "quark_shared_expert_projection_count": num_layers * 3,
    }
    for key, expected_count in expected_counts.items():
        if contract.get(key) != expected_count:
            raise ValueError(
                f"Ornith contract {key} must be {expected_count}, got {contract.get(key)!r}"
            )

    direct = _direct_specs(num_layers=num_layers, layer_types=layer_types)
    expected: dict[str, tuple[tuple[int, ...], str]] = {
        spec.target_name: (spec.shape, str(spec.dtype).removeprefix("torch."))
        for spec in direct
    }
    for index in range(num_layers):
        expected.update(_moe_target_shapes(index))

    interface = manifest.get("interface")
    records = interface.get("tensors") if isinstance(interface, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("Ornith manifest is missing its tensor interface")
    declared: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("name"), str):
            raise ValueError("Ornith manifest tensor records must be named objects")
        if "param" in record.get("roles", ()):
            declared[record["name"]] = record
    compiler_owned = {"rotary_emb_inv_freq"}
    if set(declared) != set(expected) | compiler_owned:
        missing = sorted((set(expected) | compiler_owned) - set(declared))
        extra = sorted(set(declared) - set(expected) - compiler_owned)
        raise ValueError(
            f"Ornith manifest parameter mismatch; missing={missing[:8]}, extra={extra[:8]}"
        )
    for name, (shape, dtype) in expected.items():
        record = declared[name]
        if (
            record.get("binding") != "unbound"
            or record.get("dtype") != dtype
            or _exact_shape(record) != shape
        ):
            raise ValueError(
                f"Ornith manifest constant {name} must be unbound {dtype} {shape}"
            )
    rotary = declared["rotary_emb_inv_freq"]
    if (
        rotary.get("binding") != "compiler_owned"
        or rotary.get("dtype") != "float32"
        or _exact_shape(rotary) != (32,)
    ):
        raise ValueError("Ornith rotary inv-freq constant is invalid")
    return num_layers, layer_types, direct, frozenset(expected)


class _ExpertLayerBuffer:
    def __init__(self) -> None:
        self.outputs = {
            "w13_weight": torch.empty((256, 1024, 1024), dtype=torch.uint8),
            "w13_weight_scale": torch.empty((256, 1024, 64), dtype=torch.uint8),
            "w2_weight": torch.empty((256, 2048, 256), dtype=torch.uint8),
            "w2_weight_scale": torch.empty((256, 2048, 16), dtype=torch.uint8),
        }
        self.seen: set[tuple[int, str, str]] = set()

    @staticmethod
    def _destination(projection: str, component: str):
        expected = {
            ("gate_proj", "weight"): ((512, 1024), "w13_weight", slice(0, 512)),
            ("up_proj", "weight"): ((512, 1024), "w13_weight", slice(512, 1024)),
            ("down_proj", "weight"): ((2048, 256), "w2_weight", slice(None)),
            ("gate_proj", "weight_scale"): (
                (512, 64), "w13_weight_scale", slice(0, 512)
            ),
            ("up_proj", "weight_scale"): (
                (512, 64), "w13_weight_scale", slice(512, 1024)
            ),
            ("down_proj", "weight_scale"): (
                (2048, 16), "w2_weight_scale", slice(None)
            ),
        }
        try:
            return expected[(projection, component)]
        except KeyError as error:
            raise ValueError(
                f"unsupported Ornith expert component {(projection, component)}"
            ) from error

    @classmethod
    def validate(cls, projection: str, component: str, tensor: torch.Tensor) -> None:
        shape, _, _ = cls._destination(projection, component)
        if tensor.dtype is not torch.uint8 or tuple(tensor.shape) != shape:
            raise ValueError(
                f"Ornith expert {(projection, component)} must be uint8 {shape}, got "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )

    def consume(
        self, expert: int, projection: str, component: str, tensor: torch.Tensor
    ) -> None:
        key = (expert, projection, component)
        if key in self.seen:
            raise ValueError(f"duplicate Ornith expert tensor {key}")
        self.validate(projection, component, tensor)
        _, output_name, row_slice = self._destination(projection, component)
        self.outputs[output_name][expert, row_slice].copy_(tensor)
        self.seen.add(key)

    @property
    def complete(self) -> bool:
        return len(self.seen) == 256 * 6


class _SharedExpertBuffer:
    def __init__(self) -> None:
        self.parts: dict[tuple[str, str], torch.Tensor] = {}

    def consume(self, projection: str, component: str, tensor: torch.Tensor) -> None:
        key = (projection, component)
        if key in self.parts:
            raise ValueError(f"duplicate Ornith shared-expert tensor {key}")
        shapes = {
            ("gate_proj", "weight"): (512, 1024),
            ("up_proj", "weight"): (512, 1024),
            ("down_proj", "weight"): (2048, 256),
            ("gate_proj", "weight_scale"): (512, 64),
            ("up_proj", "weight_scale"): (512, 64),
            ("down_proj", "weight_scale"): (2048, 16),
        }
        shape = shapes[key]
        if tensor.dtype is not torch.uint8 or tuple(tensor.shape) != shape:
            raise ValueError(
                f"Ornith shared expert {key} must be uint8 {shape}, got "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )
        self.parts[key] = tensor.detach().to(device="cpu").contiguous()

    @property
    def complete(self) -> bool:
        return len(self.parts) == 6

    def outputs(self) -> dict[str, torch.Tensor]:
        if not self.complete:
            raise RuntimeError("cannot finalize incomplete Ornith shared expert")
        return {
            "w13_weight": torch.cat(
                (self.parts[("gate_proj", "weight")], self.parts[("up_proj", "weight")])
            ).unsqueeze(0),
            "w13_weight_scale": torch.cat(
                (
                    self.parts[("gate_proj", "weight_scale")],
                    self.parts[("up_proj", "weight_scale")],
                )
            ).unsqueeze(0),
            "w2_weight": self.parts[("down_proj", "weight")].unsqueeze(0),
            "w2_weight_scale": self.parts[("down_proj", "weight_scale")].unsqueeze(0),
        }


class OrnithMxfp4StreamingLoader:
    """Consume vLLM's checkpoint iterator with one routed-expert layer staged."""

    def __init__(self, manifest: Mapping[str, object], *, max_pending_layers: int = 2):
        if max_pending_layers <= 0:
            raise ValueError("max_pending_layers must be positive")
        (
            self.num_layers,
            self.layer_types,
            direct,
            self.expected_target_names,
        ) = ornith_specs_from_manifest(manifest)
        self.direct = {spec.source_name: spec for spec in direct}
        self.max_pending_layers = max_pending_layers
        self._direct_seen: set[str] = set()
        self._expert_pending: dict[int, _ExpertLayerBuffer] = {}
        self._shared_pending: dict[int, _SharedExpertBuffer] = {}
        self._expert_completed: set[int] = set()
        self._shared_completed: set[int] = set()
        self.emitted_target_names: set[str] = set()

    @staticmethod
    def _prepared(spec: OrnithTensorSpec, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype is not spec.dtype or tuple(tensor.shape) != spec.shape:
            raise ValueError(
                f"{spec.source_name} must be {spec.dtype} {spec.shape}, got "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )
        result = tensor.detach().to(device="cpu").contiguous()
        if spec.add_one:
            result = result.add(torch.ones_like(result))
        return result

    def _emit(self, values: Iterable[tuple[str, torch.Tensor]]):
        result = []
        for name, tensor in values:
            if name in self.emitted_target_names:
                raise ValueError(f"duplicate Ornith target constant {name}")
            self.emitted_target_names.add(name)
            result.append((name, tensor))
        return result

    def consume(self, name: str, tensor: torch.Tensor):
        spec = self.direct.get(name)
        if spec is not None:
            if name in self._direct_seen:
                raise ValueError(f"duplicate Ornith checkpoint tensor {name}")
            self._direct_seen.add(name)
            return self._emit(((spec.target_name, self._prepared(spec, tensor)),))

        match = _EXPERT_RE.match(name)
        if match is not None:
            layer = int(match.group("layer"))
            expert = int(match.group("expert"))
            if layer >= self.num_layers:
                return []
            if not 0 <= expert < 256:
                raise ValueError(f"invalid Ornith expert index in {name}")
            if layer in self._expert_completed:
                raise ValueError(f"late duplicate Ornith expert tensor {name}")
            pending = self._expert_pending.get(layer)
            if pending is None:
                if len(self._expert_pending) >= self.max_pending_layers:
                    raise MemoryError(
                        "Ornith expert checkpoint order exceeded staging bound"
                    )
                _ExpertLayerBuffer.validate(
                    match.group("projection"), match.group("component"), tensor
                )
                pending = _ExpertLayerBuffer()
                self._expert_pending[layer] = pending
            pending.consume(
                expert,
                match.group("projection"),
                match.group("component"),
                tensor,
            )
            if not pending.complete:
                return []
            del self._expert_pending[layer]
            self._expert_completed.add(layer)
            prefix = f"layers_{layer}_mlp_experts_"
            return self._emit(
                (prefix + suffix, value.contiguous())
                for suffix, value in pending.outputs.items()
            )

        match = _SHARED_RE.match(name)
        if match is not None:
            layer = int(match.group("layer"))
            if layer >= self.num_layers:
                return []
            if layer in self._shared_completed:
                raise ValueError(f"late duplicate Ornith shared-expert tensor {name}")
            pending = self._shared_pending.get(layer)
            if pending is None:
                if len(self._shared_pending) >= self.max_pending_layers:
                    raise MemoryError(
                        "Ornith shared-expert order exceeded staging bound"
                    )
                pending = _SharedExpertBuffer()
                self._shared_pending[layer] = pending
            pending.consume(
                match.group("projection"), match.group("component"), tensor
            )
            if not pending.complete:
                return []
            del self._shared_pending[layer]
            self._shared_completed.add(layer)
            prefix = f"layers_{layer}_mlp_shared_expert_"
            return self._emit(
                (prefix + suffix, value.contiguous())
                for suffix, value in pending.outputs().items()
            )

        language_prefix = "model.language_model.layers."
        if name.startswith(language_prefix):
            rest = name[len(language_prefix) :]
            try:
                layer = int(rest.split(".", 1)[0])
            except ValueError as error:
                raise ValueError(f"invalid Ornith checkpoint tensor name {name}") from error
            if layer >= self.num_layers:
                return []
            raise ValueError(f"unsupported Ornith text tensor {name}")
        if name.startswith("model.visual.") or name.startswith("mtp."):
            return []
        return None

    def finish(self) -> None:
        missing_direct = sorted(set(self.direct) - self._direct_seen)
        missing_experts = sorted(set(range(self.num_layers)) - self._expert_completed)
        missing_shared = sorted(set(range(self.num_layers)) - self._shared_completed)
        missing_targets = sorted(self.expected_target_names - self.emitted_target_names)
        if self._expert_pending or self._shared_pending:
            raise ValueError("Ornith checkpoint ended with incomplete expert layers")
        if missing_direct or missing_experts or missing_shared or missing_targets:
            raise ValueError(
                "Ornith checkpoint is incomplete: "
                f"direct={missing_direct[:8]}, experts={missing_experts[:8]}, "
                f"shared={missing_shared[:8]}, targets={missing_targets[:8]}"
            )
