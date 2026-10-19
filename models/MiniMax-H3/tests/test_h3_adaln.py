"""CPU checks for the cached H3 Ref2VA AdaLN table contract."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE))

from paiton_video import adaln, refpack


def _tables(t=5, hidden=8, blocks=50):
    values = torch.tensor([0.0, 0.2, 0.4, 0.999, 1.0], dtype=torch.float32)[:t]
    tensors = {"adaln_t_values_turbo4": values}
    for index in range(blocks):
        tensors[f"blocks.{index}.adaln_t_table_turbo4"] = torch.arange(
            t*3*6*hidden, dtype=torch.float32).reshape(t*3, 6*hidden).to(torch.bfloat16)
    tensors["final_layer.adaln_t_table_turbo4"] = torch.arange(
        t*2*hidden, dtype=torch.float32).reshape(t, 2*hidden).to(torch.bfloat16)
    return tensors


def _quantizer_manifest(t=5, hidden=8):
    values = [0.0, 0.2, 0.4, 0.999, 1.0][:t]
    return {
        "h3_adaln": "per_modality_cached",
        "h3_adaln_tables": {
            "index": "timestep_index*3+token_tag",
            "token_tags": adaln.TOKEN_TAGS,
            "row_order": list(adaln.ROW_ORDER),
            "row_width": 6*hidden,
            "final_row_width": 2*hidden,
            "presets": {
                "turbo4": {"num_timesteps": t, "values": values}
            },
        },
    }


def _manifest(t=5, hidden=8):
    return {
        "adaln_tables": {
            "format": adaln.FORMAT,
            "abi_version": adaln.ABI_VERSION,
            "hidden": hidden,
            "row_index": "timestep_index*3+token_tag",
            "token_tags": adaln.TOKEN_TAGS,
            "row_order": list(adaln.ROW_ORDER),
            "dtype": "bfloat16",
            "presets": {
                "turbo4": {
                    "steps": 4,
                    "shift_video": 6.0,
                    "shift_audio": 3.0,
                    "values": "adaln_t_values_turbo4",
                    "blocks": "blocks.{block}.adaln_t_table_turbo4",
                    "final": "final_layer.adaln_t_table_turbo4",
                }
            },
        }
    }


class AdalnTableTest(unittest.TestCase):
    def _directory(self, tensors=None, manifest=True):
        directory = tempfile.TemporaryDirectory(prefix="h3-adaln-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        save_file(tensors or _tables(), str(root/"adaln_tables.safetensors"),
                  metadata={"format": adaln.FORMAT, "abi_version": str(adaln.ABI_VERSION)})
        if manifest:
            (root/"manifest.json").write_text(json.dumps(_manifest()))
        return root

    def test_loads_the_cached_table_contract(self):
        tables = adaln.load_tables(self._directory(), "turbo4")
        self.assertEqual(tables.preset, "turbo4")
        self.assertEqual(tables.hidden, 8)
        torch.testing.assert_close(tables.values,
                                   torch.tensor([0.0, 0.2, 0.4, 0.999, 1.0]),
                                   atol=0, rtol=0)
        self.assertEqual(len(tables.blocks), 50)
        self.assertEqual(tuple(tables.blocks[0].shape), (15, 48))
        self.assertEqual(tuple(tables.final.shape), (5, 16))
        self.assertEqual(tables.blocks[0].dtype, torch.bfloat16)

    def test_quantizer_manifest_format_loads(self):
        directory = tempfile.TemporaryDirectory(prefix="h3-adaln-h3manifest-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        save_file(_tables(), str(root/"adaln_tables.safetensors"))
        (root/"manifest.json").write_text(json.dumps(_quantizer_manifest()))
        tables = adaln.load_tables(root, "turbo4")
        self.assertEqual(tables.hidden, 8)
        self.assertEqual(tables.values.numel(), 5)
        broken = _quantizer_manifest()
        broken["h3_adaln_tables"]["presets"]["turbo4"]["values"] = [0.0, 0.1, 0.4, 0.999, 1.0]
        (root/"manifest.json").write_text(json.dumps(broken))
        with self.assertRaisesRegex(ValueError, "values tensor"):
            adaln.load_tables(root, "turbo4")

    def test_manifest_contract_is_checked(self):
        root = self._directory()
        manifest = _manifest()
        manifest["adaln_tables"]["row_order"] = ["scale_msa", "shift_msa"]
        (root/"manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "row order"):
            adaln.load_tables(root, "turbo4")

        manifest = _manifest()
        manifest["adaln_tables"]["presets"]["turbo4"]["steps"] = 8
        (root/"manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "steps"):
            adaln.load_tables(root, "turbo4")

    def test_reads_tables_from_checkpoint_shards(self):
        directory = tempfile.TemporaryDirectory(prefix="h3-adaln-shards-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        tensors = _tables()
        names = sorted(tensors)
        first = {name: tensors[name] for name in names[::2]}
        second = {name: tensors[name] for name in names[1::2]}
        save_file(first, str(root/"model-00001-of-00002.safetensors"))
        save_file(second, str(root/"model-00002-of-00002.safetensors"))
        weight_map = {name: "model-00001-of-00002.safetensors" for name in first}
        weight_map.update({name: "model-00002-of-00002.safetensors" for name in second})
        (root/"model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        (root/"manifest.json").write_text(json.dumps(_manifest()))
        tables = adaln.load_tables(root, "turbo4")
        self.assertEqual(len(tables.blocks), 50)
        self.assertEqual(tuple(tables.final.shape), (5, 16))

    def test_bad_table_shapes_are_rejected(self):
        tensors = _tables()
        tensors["blocks.7.adaln_t_table_turbo4"] = tensors["blocks.7.adaln_t_table_turbo4"][:-1]
        with self.assertRaisesRegex(ValueError, "blocks.7"):
            adaln.load_tables(self._directory(tensors), "turbo4")

    def test_install_replaces_block_and_final_projections(self):
        tables = adaln.load_tables(self._directory(), "turbo4")

        class Block:
            def __init__(self):
                self.adaln_proj = lambda t_emb: None
        class Final:
            def __init__(self):
                self.adaln_proj = lambda t_emb: None
        class Model:
            def __init__(self):
                self.blocks = [Block() for _ in range(50)]
                self.final_layer = Final()
        model = Model()
        adaln.install(model, tables)
        local = torch.tensor([[2.0], [3.0]])
        chunks = model.blocks[0].adaln_proj(local)
        self.assertEqual(len(chunks), 6)
        rows = torch.tensor([6, 7, 8, 9, 10, 11])
        expected = tables.blocks[0][rows].chunk(6, dim=-1)
        for index, chunk in enumerate(chunks):
            torch.testing.assert_close(chunk, expected[index], atol=0, rtol=0)
        final = model.final_layer.adaln_proj(torch.tensor([[1.0], [2.0]]))
        self.assertEqual(len(final), 2)
        expected_final = tables.final[torch.tensor([1, 2])].chunk(2, dim=-1)
        torch.testing.assert_close(final[0], expected_final[0], atol=0, rtol=0)
        self.assertIs(model._paiton_adaln_tables, tables)
        self.assertFalse(model.use_adaln_curves)
        index = model.time_embedder(torch.tensor([0.2, 0.4]))
        self.assertEqual(tuple(index.shape), (2, 1))
        self.assertEqual(index.reshape(-1).tolist(), [1.0, 2.0])

    def test_cached_checkpoint_placeholders_satisfy_stock_detection(self):
        state = {"video_patch_proj.weight": torch.zeros(5376, 96),
                 "adaln_t_values_base50": torch.zeros(99)}
        prepared = adaln.prepare_state_dict(state)
        self.assertIn("adaln_t_table", prepared)
        self.assertEqual(tuple(prepared["adaln_t_table"].shape), (1, 1))
        self.assertEqual(tuple(prepared["blocks.0.adaln_proj.linear.weight"].shape), (6*5376*3, 1))
        self.assertEqual(tuple(prepared["blocks.49.adaln_proj.linear.bias"].shape), (6*5376*3,))
        self.assertEqual(tuple(prepared["final_layer.adaln_proj.linear.weight"].shape), (2*5376, 1))
        self.assertNotIn("adaln_t_table", state)
        again = adaln.prepare_state_dict(prepared)
        self.assertIs(again, prepared)
        plain = {"video_patch_proj.weight": torch.zeros(1)}
        self.assertIs(adaln.prepare_state_dict(plain), plain)
        pruned = dict(state)
        pruned["adaln_t_table"] = torch.zeros(1025, 8)
        self.assertIs(adaln.prepare_state_dict(pruned), pruned)

    def test_install_pruned_adaln_copies_table_and_weights(self):
        class Linear(torch.nn.Module):
            def __init__(self, out):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(out, 8))
                self.bias = torch.nn.Parameter(torch.zeros(out))
        class Adaln(torch.nn.Module):
            def __init__(self, out):
                super().__init__()
                self.linear = Linear(out)
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adaln_proj = Adaln(48)
        class Final(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adaln_proj = Adaln(16)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = 8
                self.blocks = torch.nn.ModuleList([Block(), Block()])
                self.final_layer = Final()
                self.use_adaln_curves = False
                self.register_buffer("adaln_t_table", torch.zeros(5, 8))
        model = Model()
        table = torch.arange(5*8, dtype=torch.float32).reshape(5, 8)
        blocks = tuple((torch.arange(48*8, dtype=torch.float32).reshape(48, 8) + index,
                        torch.arange(48, dtype=torch.float32) + index)
                       for index in range(2))
        final = (torch.arange(16*8, dtype=torch.float32).reshape(16, 8),
                 torch.arange(16, dtype=torch.float32))
        pruned = adaln.PrunedAdaln(table, blocks, final)
        adaln.install_pruned(model, pruned)
        self.assertTrue(model.use_adaln_curves)
        torch.testing.assert_close(model.adaln_t_table, table, atol=0, rtol=0)
        for index, block in enumerate(model.blocks):
            torch.testing.assert_close(block.adaln_proj.linear.weight,
                                       blocks[index][0], atol=0, rtol=0)
            torch.testing.assert_close(block.adaln_proj.linear.bias,
                                       blocks[index][1], atol=0, rtol=0)
        torch.testing.assert_close(model.final_layer.adaln_proj.linear.weight,
                                   final[0], atol=0, rtol=0)

    def test_load_pruned_adaln_from_starsfriday_if_available(self):
        path = os.environ.get("PAITON_H3_REF2VA_CHECKPOINT")
        if not path or not Path(path).is_file():
            self.skipTest("PAITON_H3_REF2VA_CHECKPOINT is not set to the checkpoint")
        pruned = adaln.load_pruned_adaln(path)
        self.assertEqual(tuple(pruned.table.shape), (1025, 8))
        self.assertEqual(len(pruned.blocks), 50)
        self.assertEqual(tuple(pruned.blocks[0][0].shape), (96768, 8))
        self.assertEqual(tuple(pruned.final[0].shape), (10752, 8))

    def test_layout_global_indices_address_the_table(self):
        tables = adaln.load_tables(self._directory(), "turbo4")
        layout = refpack.build_packed_layout(
            8, 5, 4, 4, 5, [], video_timestep=0.4, audio_timestep=0.2,
            adaln_values=tables.values)
        adaln.check_layout(layout, tables)
        rows = tables.blocks[0][layout.adaln_indices]
        self.assertEqual(tuple(rows.shape), (layout.seq_len, 48))
        bad = refpack.build_packed_layout(8, 5, 4, 4, 5, [], video_timestep=0.4,
                                          audio_timestep=0.2)
        with self.assertRaisesRegex(ValueError, "timestep values"):
            adaln.check_layout(bad, tables)


@unittest.skipUnless(os.environ.get("PAITON_H3_ADALN_TABLES"),
                     "PAITON_H3_ADALN_TABLES required")
class AdalnPackageTest(unittest.TestCase):
    def test_ready_values_match_the_comfyui_release_schedule(self):
        try:
            import comfy.model_sampling as model_sampling
            from comfy.samplers import simple_scheduler
            from comfy.ldm.minimax.model import time_shift_sigma
        except ImportError as error:
            self.skipTest(f"ComfyUI is not importable: {error}")
        root = os.environ["PAITON_H3_ADALN_TABLES"]
        specs = {"base50": (50, 12.0), "turbo4": (4, 6.0), "turbo8": (8, 6.0)}
        for preset, (steps, shift) in specs.items():
            sampling = model_sampling.ModelSamplingAV(None)
            sampling.set_parameters(shift=shift, audio_shift=3.0)
            sigmas = simple_scheduler(sampling, steps)[:-1]
            video = 1.0 - sigmas
            audio = 1.0 - torch.tensor(
                [time_shift_sigma(float(value), shift, 3.0) for value in sigmas])
            expected = torch.unique(
                torch.cat([video, audio, torch.tensor([0.999, 1.0])]), sorted=True)
            tables = adaln.load_tables(root, preset)
            torch.testing.assert_close(tables.values, expected, atol=1e-7, rtol=0)

    def test_ready_package_loads_all_presets_and_indexes_the_layout(self):
        root = os.environ["PAITON_H3_ADALN_TABLES"]
        for preset in ("base50", "turbo4", "turbo8"):
            tables = adaln.load_tables(root, preset)
            rows = tables.values.numel()
            self.assertEqual(len(tables.blocks), 50)
            self.assertEqual(tuple(tables.blocks[0].shape), (rows*3, 6*5376))
            self.assertEqual(tuple(tables.final.shape), (rows, 2*5376))
        tables = adaln.load_tables(root, "base50")
        refs = [{"kind": "image", "latent_h": 48, "latent_w": 86} for _ in range(9)]
        layout = refpack.build_packed_layout(
            56, 37, 30, 54, 207, refs,
            video_timestep=float(tables.values[10]),
            audio_timestep=float(tables.values[5]), adaln_values=tables.values)
        adaln.check_layout(layout, tables)
        self.assertLess(int(layout.adaln_indices.max()), tables.values.numel()*3)


class AdalnNodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        import types
        if "comfy" not in sys.modules:
            comfy = types.ModuleType("comfy")
            comfy.sd = types.SimpleNamespace(VAE=object)
            comfy.utils = types.SimpleNamespace(load_torch_file=None)
            sys.modules["comfy"] = comfy
            sys.modules["comfy.sd"] = comfy.sd
            sys.modules["comfy.utils"] = comfy.utils
        if "folder_paths" not in sys.modules:
            sys.modules["folder_paths"] = types.SimpleNamespace(
                get_full_path_or_raise=lambda *args: None)
        spec = importlib.util.spec_from_file_location(
            "paiton_h3_adaln_node", RELEASE/"comfyui/paiton_h3/__init__.py")
        cls.node = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.node)

    def test_node_registers_the_three_presets(self):
        self.assertIn("PaitonH3AdaLN", self.node.NODE_CLASS_MAPPINGS)
        schema = self.node.PaitonH3AdaLN.INPUT_TYPES()
        self.assertEqual(schema["required"]["preset"][0], ["base50", "turbo4", "turbo8"])
        self.assertEqual(schema["optional"]["mode"][0], ["pruned_8dim", "cached_tables"])
        self.assertEqual(schema["optional"]["mode"][1]["default"], "pruned_8dim")

    def test_node_installs_the_selected_preset(self):
        import types
        directory = tempfile.TemporaryDirectory(prefix="h3-adaln-node-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        save_file(_tables(), str(root/"adaln_tables.safetensors"))
        (root/"manifest.json").write_text(json.dumps(_manifest()))

        class Block:
            def __init__(self):
                self.adaln_proj = lambda t_emb: None
        class Final:
            def __init__(self):
                self.adaln_proj = lambda t_emb: None
        class FakeH3:
            def __init__(self):
                self.blocks = [Block() for _ in range(50)]
                self.final_layer = Final()
        sys.modules["comfy.ldm.minimax.model"] = types.SimpleNamespace(MiniMaxH3Model=FakeH3)
        diffusion = FakeH3()
        model = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=diffusion))
        result = self.node.PaitonH3AdaLN().apply(model, str(root), "turbo4",
                                                 mode="cached_tables")
        self.assertEqual(result, (model,))
        self.assertIsInstance(diffusion.blocks[0].adaln_proj, adaln.CachedAdalnBlock)
        self.assertIsInstance(diffusion.final_layer.adaln_proj, adaln.CachedAdalnOut)


if __name__ == "__main__":
    unittest.main()
