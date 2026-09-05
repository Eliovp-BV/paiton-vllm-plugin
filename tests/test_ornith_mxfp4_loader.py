import hashlib
import unittest

import torch

from paiton_vllm_plugin.runtime.core.utils.ornith_mxfp4_loader import (
    OrnithMxfp4StreamingLoader,
    _SharedExpertBuffer,
    _direct_specs,
    _moe_target_shapes,
    ornith_specs_from_manifest,
)


def manifest(
    num_layers=4,
    *,
    contract_version=1,
    grouped_mxfp4_prefill=None,
    dense_mxfp4_prefill=None,
    wave_mxfp4_decode=None,
    merged_mxfp4_projections=None,
    dot2_mxfp4_decode=None,
    dot2_mxfp4_moe_decode=None,
    fused_shared_moe_decode=None,
    fused_shared_gate_decode=None,
    dflash_aux_hidden_states=None,
    dflash_aux_hidden_state_layers=None,
    dflash_aux_hidden_state_outputs=None,
    dflash_max_speculative_tokens=None,
    dflash_state_index_widths=None,
    dflash_full_attention_metadata=None,
    dflash_full_attention_metadata_layers=None,
):
    layer_types = tuple(
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(num_layers)
    )
    tensors = []
    for spec in _direct_specs(num_layers=num_layers, layer_types=layer_types):
        tensors.append(
            {
                "name": spec.target_name,
                "roles": ["param"],
                "binding": "unbound",
                "dtype": str(spec.dtype).removeprefix("torch."),
                "shape_values": [[dim] for dim in spec.shape],
            }
        )
    for index in range(num_layers):
        for name, (shape, dtype) in _moe_target_shapes(index).items():
            tensors.append(
                {
                    "name": name,
                    "roles": ["param"],
                    "binding": "unbound",
                    "dtype": dtype,
                    "shape_values": [[dim] for dim in shape],
                }
            )
    tensors.append(
        {
            "name": "rotary_emb_inv_freq",
            "roles": ["param"],
            "binding": "compiler_owned",
            "dtype": "float32",
            "shape_values": [[32]],
        }
    )
    schedule = ",".join(layer_types)
    result = {
        "interface": {"tensors": tensors},
        "paiton_ornith15_contract": {
            "version": contract_version,
            "product_model_type": "ornith_1_5_qwen3_5_moe",
            "scope": "text-only",
            "tp_size": 1,
            "source_num_hidden_layers": 40,
            "num_hidden_layers": num_layers,
            "num_gdn_layers": sum(x == "linear_attention" for x in layer_types),
            "num_full_attention_layers": sum(x == "full_attention" for x in layer_types),
            "layer_schedule_sha256": hashlib.sha256(schedule.encode()).hexdigest(),
            "max_batch_size": 1,
            "max_num_batched_tokens": 8192,
            "max_context_length": 8192,
            "kv_cache_block_size": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "gdn_conv_state_shape": (
                [19, 8192] if contract_version in (10, 11) else [3, 8192]
            ),
            "gdn_recurrent_state_shape": [32, 128, 128],
            "runtime_shell_parameters": [
                {
                    "name": "model.language_model.embed_tokens.weight",
                    "dtype": "bfloat16",
                    "shape": [248320, 2048],
                },
                {
                    "name": "lm_head.weight",
                    "dtype": "bfloat16",
                    "shape": [248320, 2048],
                },
            ],
            "quark_format": "mxfp4_e2m1",
            "quark_group_size": 32,
            "quark_checkpoint_weight_layout": "Nx(K/2)_packed_e2m1_u8",
            "quark_checkpoint_scale_layout": "Nx(K/32)_ue8m0_u8",
            "quark_kernel_layout": "quark_mxfp4_e2m1_ue8m0_g32_v1",
            "quark_dense_linear_count": sum(
                x == "linear_attention" for x in layer_types
            ) * 5,
            "quark_routed_expert_projection_count": num_layers * 256 * 3,
            "quark_shared_expert_projection_count": num_layers * 3,
            "num_experts": 256,
            "num_experts_per_token": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "zero_centered_norm_transform": "gamma=1+checkpoint_bf16",
        },
    }
    if grouped_mxfp4_prefill is not None:
        result["paiton_ornith15_contract"]["grouped_mxfp4_prefill"] = (
            grouped_mxfp4_prefill
        )
    if dense_mxfp4_prefill is not None:
        result["paiton_ornith15_contract"]["dense_mxfp4_prefill"] = (
            dense_mxfp4_prefill
        )
    if wave_mxfp4_decode is not None:
        result["paiton_ornith15_contract"]["wave_mxfp4_decode"] = (
            wave_mxfp4_decode
        )
    if merged_mxfp4_projections is not None:
        result["paiton_ornith15_contract"]["merged_mxfp4_projections"] = (
            merged_mxfp4_projections
        )
    if dot2_mxfp4_decode is not None:
        result["paiton_ornith15_contract"]["dot2_mxfp4_decode"] = (
            dot2_mxfp4_decode
        )
    if dot2_mxfp4_moe_decode is not None:
        result["paiton_ornith15_contract"]["dot2_mxfp4_moe_decode"] = (
            dot2_mxfp4_moe_decode
        )
    if fused_shared_moe_decode is not None:
        result["paiton_ornith15_contract"]["fused_shared_moe_decode"] = (
            fused_shared_moe_decode
        )
    if fused_shared_gate_decode is not None:
        result["paiton_ornith15_contract"]["fused_shared_gate_decode"] = (
            fused_shared_gate_decode
        )
    if dflash_aux_hidden_states is not None:
        result["paiton_ornith15_contract"]["dflash_aux_hidden_states"] = (
            dflash_aux_hidden_states
        )
    if dflash_aux_hidden_state_layers is not None:
        result["paiton_ornith15_contract"]["dflash_aux_hidden_state_layers"] = (
            dflash_aux_hidden_state_layers
        )
    if dflash_aux_hidden_state_outputs is not None:
        result["paiton_ornith15_contract"]["dflash_aux_hidden_state_outputs"] = (
            dflash_aux_hidden_state_outputs
        )
    if dflash_max_speculative_tokens is not None:
        result["paiton_ornith15_contract"]["dflash_max_speculative_tokens"] = (
            dflash_max_speculative_tokens
        )
    if dflash_state_index_widths is not None:
        result["paiton_ornith15_contract"]["dflash_state_index_widths"] = (
            dflash_state_index_widths
        )
    if dflash_full_attention_metadata is not None:
        result["paiton_ornith15_contract"]["dflash_full_attention_metadata"] = (
            dflash_full_attention_metadata
        )
    if dflash_full_attention_metadata_layers is not None:
        result["paiton_ornith15_contract"][
            "dflash_full_attention_metadata_layers"
        ] = dflash_full_attention_metadata_layers
    return result


class OrnithMxfp4LoaderTests(unittest.TestCase):
    def test_manifest_exactly_covers_compiled_constants(self):
        num_layers, layer_types, direct, targets = ornith_specs_from_manifest(
            manifest()
        )
        self.assertEqual(num_layers, 4)
        self.assertEqual(layer_types, ("linear_attention",) * 3 + ("full_attention",))
        self.assertEqual(len(targets), len(direct) + 4 * 8)

    def test_contract_v2_requires_boolean_grouped_prefill_flag(self):
        value = manifest(contract_version=2, grouped_mxfp4_prefill=True)
        num_layers, _, _, _ = ornith_specs_from_manifest(value)
        self.assertEqual(num_layers, 4)

        for invalid in (None, 1, "true"):
            value = manifest(contract_version=2)
            if invalid is not None:
                value["paiton_ornith15_contract"][
                    "grouped_mxfp4_prefill"
                ] = invalid
            with self.assertRaisesRegex(
                ValueError, "requires boolean grouped_mxfp4_prefill"
            ):
                ornith_specs_from_manifest(value)

    def test_contract_v1_rejects_grouped_prefill_field(self):
        with self.assertRaisesRegex(
            ValueError, "version 1 cannot declare optimization fields"
        ):
            ornith_specs_from_manifest(
                manifest(contract_version=1, grouped_mxfp4_prefill=False)
            )

    def test_contract_v3_requires_both_prefill_flags(self):
        value = manifest(
            contract_version=3,
            grouped_mxfp4_prefill=True,
            dense_mxfp4_prefill=True,
        )
        num_layers, _, _, _ = ornith_specs_from_manifest(value)
        self.assertEqual(num_layers, 4)

        for missing in ("grouped_mxfp4_prefill", "dense_mxfp4_prefill"):
            value = manifest(
                contract_version=3,
                grouped_mxfp4_prefill=True,
                dense_mxfp4_prefill=True,
            )
            del value["paiton_ornith15_contract"][missing]
            with self.assertRaisesRegex(ValueError, "requires boolean"):
                ornith_specs_from_manifest(value)

    def test_contract_v2_rejects_dense_prefill_field(self):
        with self.assertRaisesRegex(
            ValueError, "version 2 cannot declare later optimization fields"
        ):
            ornith_specs_from_manifest(
                manifest(
                    contract_version=2,
                    grouped_mxfp4_prefill=True,
                    dense_mxfp4_prefill=False,
                )
            )

    def test_contract_v4_requires_all_optimization_flags(self):
        value = manifest(
            contract_version=4,
            grouped_mxfp4_prefill=True,
            dense_mxfp4_prefill=True,
            wave_mxfp4_decode=True,
        )
        num_layers, _, _, _ = ornith_specs_from_manifest(value)
        self.assertEqual(num_layers, 4)

        for missing in (
            "grouped_mxfp4_prefill",
            "dense_mxfp4_prefill",
            "wave_mxfp4_decode",
        ):
            value = manifest(
                contract_version=4,
                grouped_mxfp4_prefill=True,
                dense_mxfp4_prefill=True,
                wave_mxfp4_decode=True,
            )
            del value["paiton_ornith15_contract"][missing]
            with self.assertRaisesRegex(ValueError, "requires boolean"):
                ornith_specs_from_manifest(value)

    def test_contract_v3_rejects_wave_decode_field(self):
        with self.assertRaisesRegex(
            ValueError, "version 3 cannot declare later optimization fields"
        ):
            ornith_specs_from_manifest(
                manifest(
                    contract_version=3,
                    grouped_mxfp4_prefill=True,
                    dense_mxfp4_prefill=True,
                    wave_mxfp4_decode=False,
                )
            )

    def test_contract_v5_requires_dense_prefill_for_merged_projections(self):
        with self.assertRaisesRegex(ValueError, "require dense MXFP4 prefill"):
            ornith_specs_from_manifest(
                manifest(
                    contract_version=5,
                    grouped_mxfp4_prefill=True,
                    dense_mxfp4_prefill=False,
                    wave_mxfp4_decode=True,
                    merged_mxfp4_projections=True,
                )
            )

    def test_contract_v5_requires_all_optimization_flags(self):
        kwargs = {
            "contract_version": 5,
            "grouped_mxfp4_prefill": True,
            "dense_mxfp4_prefill": True,
            "wave_mxfp4_decode": True,
            "merged_mxfp4_projections": True,
        }
        num_layers, _, _, _ = ornith_specs_from_manifest(manifest(**kwargs))
        self.assertEqual(num_layers, 4)
        for missing in (
            "grouped_mxfp4_prefill",
            "dense_mxfp4_prefill",
            "wave_mxfp4_decode",
            "merged_mxfp4_projections",
        ):
            value = manifest(**kwargs)
            del value["paiton_ornith15_contract"][missing]
            with self.assertRaisesRegex(ValueError, "requires boolean"):
                ornith_specs_from_manifest(value)

    def test_contract_v4_rejects_merged_projection_field(self):
        with self.assertRaisesRegex(
            ValueError, "version 4 cannot declare merged MXFP4 projections"
        ):
            ornith_specs_from_manifest(
                manifest(
                    contract_version=4,
                    grouped_mxfp4_prefill=True,
                    dense_mxfp4_prefill=True,
                    wave_mxfp4_decode=True,
                    merged_mxfp4_projections=False,
                )
            )

    def test_contract_v6_requires_paired_dot_dependencies(self):
        kwargs = {
            "contract_version": 6,
            "grouped_mxfp4_prefill": True,
            "dense_mxfp4_prefill": True,
            "wave_mxfp4_decode": True,
            "merged_mxfp4_projections": True,
            "dot2_mxfp4_decode": True,
        }
        num_layers, _, _, _ = ornith_specs_from_manifest(manifest(**kwargs))
        self.assertEqual(num_layers, 4)
        for missing in (
            "grouped_mxfp4_prefill",
            "dense_mxfp4_prefill",
            "wave_mxfp4_decode",
            "merged_mxfp4_projections",
            "dot2_mxfp4_decode",
        ):
            value = manifest(**kwargs)
            del value["paiton_ornith15_contract"][missing]
            with self.assertRaisesRegex(ValueError, "requires boolean"):
                ornith_specs_from_manifest(value)
        for wave, merged in ((False, True), (True, False)):
            value = manifest(
                **{
                    **kwargs,
                    "wave_mxfp4_decode": wave,
                    "merged_mxfp4_projections": merged,
                }
            )
            with self.assertRaisesRegex(ValueError, "requires wave decode"):
                ornith_specs_from_manifest(value)

    def test_contract_v5_rejects_paired_dot_field(self):
        with self.assertRaisesRegex(ValueError, "version 5 cannot declare"):
            ornith_specs_from_manifest(
                manifest(
                    contract_version=5,
                    grouped_mxfp4_prefill=True,
                    dense_mxfp4_prefill=True,
                    wave_mxfp4_decode=True,
                    merged_mxfp4_projections=True,
                    dot2_mxfp4_decode=False,
                )
            )

    def test_contract_v7_splits_projection_and_moe_paired_dot(self):
        kwargs = {
            "contract_version": 7,
            "grouped_mxfp4_prefill": True,
            "dense_mxfp4_prefill": True,
            "wave_mxfp4_decode": True,
            "merged_mxfp4_projections": True,
            "dot2_mxfp4_decode": True,
            "dot2_mxfp4_moe_decode": False,
        }
        num_layers, _, _, _ = ornith_specs_from_manifest(manifest(**kwargs))
        self.assertEqual(num_layers, 4)
        for missing in ("dot2_mxfp4_decode", "dot2_mxfp4_moe_decode"):
            value = manifest(**kwargs)
            del value["paiton_ornith15_contract"][missing]
            with self.assertRaisesRegex(ValueError, "requires boolean"):
                ornith_specs_from_manifest(value)
        value = manifest(**{**kwargs, "wave_mxfp4_decode": False,
                            "dot2_mxfp4_moe_decode": True})
        with self.assertRaisesRegex(ValueError, "MoE decode requires wave"):
            ornith_specs_from_manifest(value)

    def test_contract_v8_requires_fused_shared_dependencies(self):
        kwargs = {
            "contract_version": 8,
            "grouped_mxfp4_prefill": True,
            "dense_mxfp4_prefill": True,
            "wave_mxfp4_decode": True,
            "merged_mxfp4_projections": True,
            "fused_shared_moe_decode": True,
        }
        num_layers, _, _, _ = ornith_specs_from_manifest(manifest(**kwargs))
        self.assertEqual(num_layers, 4)
        for field in ("fused_shared_moe_decode", "wave_mxfp4_decode"):
            value = manifest(**kwargs)
            value["paiton_ornith15_contract"].pop(field)
            with self.assertRaisesRegex(ValueError, "requires"):
                ornith_specs_from_manifest(value)
        for field in ("dot2_mxfp4_decode", "dot2_mxfp4_moe_decode"):
            value = manifest(**kwargs)
            value["paiton_ornith15_contract"][field] = False
            with self.assertRaisesRegex(ValueError, "cannot declare rejected"):
                ornith_specs_from_manifest(value)

    def test_contract_v9_requires_fused_shared_gate_dependencies(self):
        kwargs = {
            "contract_version": 9,
            "grouped_mxfp4_prefill": True,
            "dense_mxfp4_prefill": True,
            "wave_mxfp4_decode": True,
            "merged_mxfp4_projections": True,
            "fused_shared_moe_decode": True,
            "fused_shared_gate_decode": True,
        }
        num_layers, _, _, _ = ornith_specs_from_manifest(manifest(**kwargs))
        self.assertEqual(num_layers, 4)
        for field in (
            "fused_shared_moe_decode",
            "fused_shared_gate_decode",
            "wave_mxfp4_decode",
        ):
            value = manifest(**kwargs)
            value["paiton_ornith15_contract"].pop(field)
            with self.assertRaisesRegex(ValueError, "requires"):
                ornith_specs_from_manifest(value)
        value = manifest(**{**kwargs, "grouped_mxfp4_prefill": False})
        with self.assertRaisesRegex(ValueError, "requires grouped prefill"):
            ornith_specs_from_manifest(value)

    def test_contract_v11_requires_exact_dflash_outputs_and_metadata(self):
        layers = [2, 7, 12, 17, 23, 28, 33, 38]
        outputs = [
            f"dflash_aux_hidden_state_layer_{layer:02d}" for layer in layers
        ]
        kwargs = {
            "num_layers": 40,
            "contract_version": 11,
            "grouped_mxfp4_prefill": True,
            "dense_mxfp4_prefill": True,
            "wave_mxfp4_decode": True,
            "merged_mxfp4_projections": True,
            "fused_shared_moe_decode": True,
            "fused_shared_gate_decode": True,
            "dflash_aux_hidden_states": True,
            "dflash_aux_hidden_state_layers": layers,
            "dflash_aux_hidden_state_outputs": outputs,
            "dflash_max_speculative_tokens": 16,
            "dflash_state_index_widths": [1, 17],
            "dflash_full_attention_metadata": "per_layer",
            "dflash_full_attention_metadata_layers": [
                3, 7, 11, 15, 19, 23, 27, 31, 35, 39
            ],
        }
        num_layers, _, _, _ = ornith_specs_from_manifest(manifest(**kwargs))
        self.assertEqual(num_layers, 40)
        for field in (
            "dflash_aux_hidden_states",
            "dflash_aux_hidden_state_layers",
            "dflash_aux_hidden_state_outputs",
            "dflash_max_speculative_tokens",
            "dflash_state_index_widths",
            "dflash_full_attention_metadata",
            "dflash_full_attention_metadata_layers",
        ):
            value = manifest(**kwargs)
            value["paiton_ornith15_contract"].pop(field)
            with self.assertRaisesRegex(ValueError, "DFlash"):
                ornith_specs_from_manifest(value)
        value = manifest(**{**kwargs, "num_layers": 4})
        with self.assertRaisesRegex(ValueError, "all 40"):
            ornith_specs_from_manifest(value)

    def test_contract_v9_rejects_dflash_fields(self):
        with self.assertRaisesRegex(ValueError, "cannot declare DFlash fields"):
            ornith_specs_from_manifest(
                manifest(
                    contract_version=9,
                    grouped_mxfp4_prefill=True,
                    dense_mxfp4_prefill=True,
                    wave_mxfp4_decode=True,
                    merged_mxfp4_projections=True,
                    fused_shared_moe_decode=True,
                    fused_shared_gate_decode=True,
                    dflash_aux_hidden_states=True,
                )
            )

    def test_direct_zero_centered_norm_transform(self):
        loader = OrnithMxfp4StreamingLoader(manifest())
        source = torch.tensor([-0.5, 0.0, 0.5] + [0.0] * 2045, dtype=torch.bfloat16)
        result = loader.consume(
            "model.language_model.layers.0.input_layernorm.weight", source
        )
        self.assertEqual(result[0][0], "layers_0_input_layernorm_weight")
        self.assertTrue(
            torch.equal(
                result[0][1][:3],
                torch.tensor([0.5, 1.0, 1.5], dtype=torch.bfloat16),
            )
        )

    def test_shared_expert_assembles_gate_then_up(self):
        pending = _SharedExpertBuffer()
        parts = {
            ("gate_proj", "weight"): torch.full((512, 1024), 1, dtype=torch.uint8),
            ("up_proj", "weight"): torch.full((512, 1024), 2, dtype=torch.uint8),
            ("down_proj", "weight"): torch.full((2048, 256), 3, dtype=torch.uint8),
            ("gate_proj", "weight_scale"): torch.full((512, 64), 4, dtype=torch.uint8),
            ("up_proj", "weight_scale"): torch.full((512, 64), 5, dtype=torch.uint8),
            ("down_proj", "weight_scale"): torch.full((2048, 16), 6, dtype=torch.uint8),
        }
        for (projection, component), tensor in parts.items():
            pending.consume(projection, component, tensor)
        values = pending.outputs()
        self.assertEqual(tuple(values["w13_weight"].shape), (1, 1024, 1024))
        self.assertTrue(torch.all(values["w13_weight"][:, :512] == 1))
        self.assertTrue(torch.all(values["w13_weight"][:, 512:] == 2))
        self.assertTrue(torch.all(values["w2_weight"] == 3))

    def test_rejects_unknown_compiled_text_tensor(self):
        loader = OrnithMxfp4StreamingLoader(manifest())
        with self.assertRaisesRegex(ValueError, "unsupported Ornith text tensor"):
            loader.consume(
                "model.language_model.layers.0.unknown.weight",
                torch.empty(1),
            )

    def test_ignores_visual_mtp_and_uncompiled_layers(self):
        loader = OrnithMxfp4StreamingLoader(manifest())
        value = torch.empty(1)
        self.assertEqual(loader.consume("model.visual.foo", value), [])
        self.assertEqual(loader.consume("mtp.layers.0.foo", value), [])
        self.assertEqual(
            loader.consume(
                "model.language_model.layers.39.mlp.experts.0.up_proj.weight",
                value,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
