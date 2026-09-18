"""CPU checks for opt-in binding; native arithmetic requires GPU validation."""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


PACKAGE = "_paiton_stock_gdn_test"
SOURCE = Path(__file__).resolve().parents[1] / "paiton_vllm_plugin"
spec = importlib.util.spec_from_file_location(
    PACKAGE + ".gdn_native_stock_prefill", SOURCE / "gdn_native_stock_prefill.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class StockLayer:
    def __init__(self, index):
        self.prefix = f"model.layers.{index}.linear_attn"
        self.tp_size, self.num_spec = 1, 7
        self.head_k_dim = self.head_v_dim = 128
        self.num_v_heads, self.num_k_heads = 48, 16
        self.gqa_interleaved_layout = self.enable_fused_gdn_decode = False
        self.chunk_gated_delta_rule = object()
        self.kv_cache = object()
        self.conv1d = object()


class PrefillWrapper:
    def __init__(self, original, scope, prefix):
        self.original, self.scope, self.prefix = original, scope, prefix


def config():
    return SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True, mamba_cache_mode="align"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=8, async_scheduling=False),
    )


class StockPrefillBindingTests(unittest.TestCase):
    def setUp(self):
        self.layers = [StockLayer(i) for i in range(48)]
        self.model = SimpleNamespace(modules=lambda: iter(self.layers))
        self.scope = object()
        upstream = ModuleType("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
        upstream.QwenGatedDeltaNetAttention = StockLayer
        native = ModuleType(PACKAGE + ".gdn_native_prefill")
        native.PaitonGDNPrefill = PrefillWrapper
        self.native = native
        self.patches = patch.dict(sys.modules, {
            upstream.__name__: upstream, native.__name__: native,
        })
        self.patches.start()
        self.addCleanup(self.patches.stop)

    def test_accepts_only_opt_in_apc_configuration(self):
        with patch.dict(os.environ, {"PAITON_EXPERIMENTAL_GDN_REPLAY": "0"}):
            adapter.validate_config(config())
            for section, name, value in (
                ("cache_config", "enable_prefix_caching", False),
                ("cache_config", "mamba_cache_mode", "all"),
                ("parallel_config", "tensor_parallel_size", 2),
                ("parallel_config", "pipeline_parallel_size", 2),
                ("scheduler_config", "max_num_seqs", 9),
            ):
                with self.subTest(name=name, value=value):
                    bad = config()
                    setattr(getattr(bad, section), name, value)
                    with self.assertRaises(ValueError):
                        adapter.validate_config(bad)
            asynchronous = config()
            asynchronous.scheduler_config.async_scheduling = True
            with self.assertRaisesRegex(ValueError, "synchronous scheduling for shared workspace"):
                adapter.validate_config(asynchronous)

    def test_compact_replay_conflict_fails_before_binding(self):
        with patch.dict(os.environ, {"PAITON_EXPERIMENTAL_GDN_REPLAY": "1"}):
            with self.assertRaisesRegex(ValueError, "compact GDN replay"):
                adapter.validate_config(config())

    def test_only_prefill_callable_changes(self):
        previous = [vars(layer).copy() for layer in self.layers]
        prefixes = adapter.bind(self.model, self.scope)
        self.assertEqual(prefixes, tuple(layer.prefix for layer in self.layers))
        for layer, before in zip(self.layers, previous):
            wrapped = layer.chunk_gated_delta_rule
            self.assertIsInstance(wrapped, PrefillWrapper)
            self.assertIs(wrapped.original, before["chunk_gated_delta_rule"])
            self.assertIs(wrapped.scope, self.scope)
            for key, value in before.items():
                if key != "chunk_gated_delta_rule":
                    self.assertIs(vars(layer)[key], value)
            self.assertEqual(set(vars(layer)), set(before))

    def test_bad_last_layer_does_not_partially_bind(self):
        originals = [layer.chunk_gated_delta_rule for layer in self.layers]
        self.layers[-1].num_spec = 3
        with self.assertRaisesRegex(ValueError, "geometry"):
            adapter.bind(self.model, self.scope)
        self.assertEqual(originals, [layer.chunk_gated_delta_rule for layer in self.layers])

    def test_preexisting_compact_layer_is_rejected(self):
        self.layers[-1].paiton_replay = object()
        with self.assertRaisesRegex(ValueError, "geometry"):
            adapter.bind(self.model, self.scope)

    def test_wrong_layer_count_is_rejected(self):
        self.layers.pop()
        with self.assertRaisesRegex(ValueError, "48 stock"):
            adapter.bind(self.model, self.scope)

    def test_native_initialization_failure_does_not_partially_bind(self):
        originals = [layer.chunk_gated_delta_rule for layer in self.layers]

        class FailingWrapper(PrefillWrapper):
            def __init__(self, original, scope, prefix):
                if prefix == self_last_prefix:
                    raise RuntimeError("native initialization failure")
                super().__init__(original, scope, prefix)

        self_last_prefix = self.layers[-1].prefix
        self.native.PaitonGDNPrefill = FailingWrapper
        with self.assertRaisesRegex(RuntimeError, "initialization failure"):
            adapter.bind(self.model, self.scope)
        self.assertEqual(originals, [layer.chunk_gated_delta_rule for layer in self.layers])

    def test_double_binding_is_rejected(self):
        adapter.bind(self.model, self.scope)
        with self.assertRaisesRegex(ValueError, "already bound"):
            adapter.bind(self.model, self.scope)


if __name__ == "__main__":
    unittest.main()
