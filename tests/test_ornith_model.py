import importlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from tests.test_qwen38_model import pinned_api_stubs


class OrnithDFlashModelContractTests(unittest.TestCase):
    def setUp(self):
        for name in (
            "paiton_vllm_plugin.models.paiton_qwen38",
            "paiton_vllm_plugin.models.paiton_ornith15",
            "paiton_vllm_plugin.vllm_compat",
        ):
            sys.modules.pop(name, None)

    def _model_class(self):
        context = SimpleNamespace(attn_metadata=None)
        modules = pinned_api_stubs(context)
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        imported = importlib.import_module(
            "paiton_vllm_plugin.models.paiton_ornith15"
        )
        return imported.PaitonOrnith15ForCausalLM

    def test_exact_dflash_auxiliary_layer_interface(self):
        cls = self._model_class()
        self.assertTrue(cls.supports_eagle3)
        self.assertIs(cls.has_own_lm_head, False)
        self.assertIs(cls.has_own_embed_tokens, False)
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.contract = {"version": 11}
        instance.config = SimpleNamespace(hidden_size=8)
        instance._dflash_output_names = tuple(
            f"dflash_aux_hidden_state_layer_{layer:02d}"
            for layer in cls.DFLASH_AUX_HIDDEN_STATE_LAYERS
        )
        instance._aux_hidden_state_layers = ()

        self.assertEqual(
            instance.get_eagle3_default_aux_hidden_state_layers(),
            (2, 7, 12, 17, 23, 28, 33, 38),
        )
        with self.assertRaisesRegex(ValueError, "requires auxiliary layers"):
            instance.set_aux_hidden_state_layers((2, 20, 37))
        instance.set_aux_hidden_state_layers((2, 7, 12, 17, 23, 28, 33, 38))

        outputs = instance._allocate_compiled_outputs(3, torch.device("cpu"))
        self.assertEqual(len(outputs), 9)
        self.assertEqual(tuple(outputs["hidden_states"].shape), (3, 8))
        hidden_states, auxiliary = instance._format_compiled_outputs(outputs)
        self.assertIs(hidden_states, outputs["hidden_states"])
        self.assertEqual(
            auxiliary,
            [outputs[name] for name in instance._dflash_output_names],
        )

        profile = torch.empty((3, 8), dtype=torch.bfloat16)
        profile_hidden, profile_auxiliary = instance._format_profile_output(profile)
        self.assertIs(profile_hidden, profile)
        self.assertEqual(len(profile_auxiliary), 8)
        self.assertEqual(len({value.data_ptr() for value in profile_auxiliary}), 8)

    def test_v9_artifact_rejects_dflash_auxiliary_layer_request(self):
        cls = self._model_class()
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.contract = {"version": 9}
        instance._aux_hidden_state_layers = ()
        with self.assertRaisesRegex(ValueError, "contract v11"):
            instance.set_aux_hidden_state_layers(
                cls.DFLASH_AUX_HIDDEN_STATE_LAYERS
            )

    def test_exact_dflash_draft_and_speculative_metadata_contract(self):
        cls = self._model_class()
        self.assertIs(cls.is_neox_style, True)
        dflash = {
            "block_size": 16,
            "mask_token_id": 248077,
            "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37],
        }
        hf_config = SimpleNamespace(
            architectures=["DFlashDraftModel"],
            model_type="qwen3",
            num_hidden_layers=6,
            hidden_size=2048,
            intermediate_size=6144,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            sliding_window=4096,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            dflash_config=dflash,
        )
        speculative_config = SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_config)
        )
        cls._validate_dflash_draft_config(speculative_config)
        wrapped_config = SimpleNamespace(
            model_type="eagle",
            architectures=["DFlashDraftModel"],
            model=hf_config,
        )
        cls._validate_dflash_draft_config(
            SimpleNamespace(
                draft_model_config=SimpleNamespace(hf_config=wrapped_config)
            )
        )
        hf_config.dflash_config = {**dflash, "block_size": 8}
        with self.assertRaisesRegex(ValueError, "metadata is incompatible"):
            cls._validate_dflash_draft_config(speculative_config)

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.contract = {"version": 11}
        instance._metadata_inputs = {}
        spec_indices = torch.arange(17, dtype=torch.int32).reshape(1, 17)
        accepted = torch.ones(1, dtype=torch.int32)
        starts = torch.tensor([0, 17], dtype=torch.int32)
        spec_metadata = SimpleNamespace(
            num_spec_decodes=1,
            num_prefills=0,
            num_decodes=0,
            spec_query_start_loc=starts,
            spec_state_indices_tensor=spec_indices,
            num_accepted_tokens=accepted,
            has_initial_state=torch.ones(1, dtype=torch.int32),
        )
        selected = instance._compiled_gdn_metadata(
            spec_metadata, torch.device("cpu")
        )
        self.assertIs(selected[0], starts)
        self.assertIs(selected[1], spec_indices)
        self.assertIs(selected[3], accepted)

        non_spec_metadata = SimpleNamespace(
            num_spec_decodes=0,
            non_spec_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            non_spec_state_indices_tensor=torch.tensor([4], dtype=torch.int32),
            has_initial_state=None,
        )
        selected = instance._compiled_gdn_metadata(
            non_spec_metadata, torch.device("cpu")
        )
        self.assertEqual(tuple(selected[1].shape), (1, 1))
        self.assertTrue(torch.equal(selected[3], torch.ones(1, dtype=torch.int32)))

    def test_dflash_metadata_rejects_mixed_or_wrong_width(self):
        cls = self._model_class()
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.contract = {"version": 11}
        instance._metadata_inputs = {}
        metadata = SimpleNamespace(
            num_spec_decodes=1,
            num_prefills=1,
            num_decodes=0,
            spec_query_start_loc=torch.tensor([0, 17], dtype=torch.int32),
            spec_state_indices_tensor=torch.zeros(1, 17, dtype=torch.int32),
            num_accepted_tokens=torch.ones(1, dtype=torch.int32),
            has_initial_state=torch.ones(1, dtype=torch.int32),
        )
        with self.assertRaisesRegex(ValueError, "isolated speculative"):
            instance._compiled_gdn_metadata(metadata, torch.device("cpu"))
        metadata.num_prefills = 0
        metadata.spec_state_indices_tensor = torch.zeros(1, 16, dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, r"\[1, 17\]"):
            instance._compiled_gdn_metadata(metadata, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
