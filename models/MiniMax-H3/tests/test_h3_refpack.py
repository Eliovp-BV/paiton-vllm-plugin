"""CPU checks for the H3 Ref2VA packed layout and reference policy."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE))

from paiton_video import refpack


def _image_block(h=48, w=86):
    return {"kind": "image", "latent_h": h, "latent_w": w}


def _video_block(t=22, h=48, w=84, audio_t=200):
    return {"kind": "video_audio", "latent_t": t, "latent_h": h, "latent_w": w,
            "ref_audio_t": audio_t}


def _audio_block(t=150):
    return {"kind": "audio", "ref_audio_t": t}


class RefPackTest(unittest.TestCase):
    def test_nine_image_inputs_keep_autogrow_order_and_first_frame(self):
        images = {f"ref_image_{i}": torch.full((2, 4, 4, 3), float(i)) for i in range(9)}
        images["ref_image_3"] = torch.stack((torch.zeros(4, 4, 3), torch.ones(4, 4, 3)))
        request = refpack.parse_refs(ref_images=images)
        self.assertEqual(len(request.images), 9)
        self.assertEqual([item.index for item in request.images], list(range(9)))
        for item in request.images:
            self.assertEqual(tuple(item.image.shape), (1, 4, 4, 3))
        self.assertEqual(float(request.images[3].image[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(request.images[8].image[0, 0, 0, 0]), 8.0)

    def test_reference_counts_are_capped(self):
        with self.assertRaisesRegex(ValueError, "at most 9"):
            refpack.parse_refs(ref_images={f"ref_image_{i}": torch.zeros(1, 4, 4, 3)
                                           for i in range(10)})
        with self.assertRaisesRegex(ValueError, "at most 3"):
            refpack.parse_refs(ref_videos={f"ref_video_{i}": torch.zeros(5, 4, 4, 3)
                                           for i in range(4)})

    def test_presentation_order_is_images_then_video_soundtrack_then_audio(self):
        images = {"ref_image_1": torch.zeros(1, 4, 4, 3),
                  "ref_image_0": torch.zeros(1, 4, 4, 3)}
        videos = {"ref_video_1": torch.zeros(5, 4, 4, 3),
                  "ref_video_2": torch.zeros(5, 4, 4, 3)}
        soundtracks = {"ref_video_audio_2": {"waveform": torch.zeros(1, 2, 800),
                                             "sample_rate": 32000}}
        audios = {"ref_audio_0": {"waveform": torch.zeros(1, 2, 400), "sample_rate": 32000}}
        request = refpack.parse_refs(images, videos, soundtracks, audios)
        self.assertEqual([item.index for item in request.images], [0, 1])
        self.assertEqual([item.index for item in request.videos], [1, 2])
        self.assertIsNone(request.videos[0].audio)
        self.assertIsNotNone(request.videos[1].audio)
        self.assertEqual([item.kind for item in request.order],
                         ["image", "image", "video", "audio", "video", "audio"])
        self.assertEqual([item["kind"] for item in request.blocks()],
                         ["image", "image", "video", "video_audio", "audio"])

    def test_match_geometry_rounds_to_32_and_never_upscales(self):
        self.assertEqual(refpack.plan_ref_geometry(864, 480, "match", 1920, 1080), (864, 480))
        self.assertEqual(refpack.plan_ref_geometry(864, 480, "match", 4000, 3000), (736, 544))
        self.assertEqual(refpack.plan_ref_geometry(864, 480, "match", 100, 100), (96, 96))
        self.assertEqual(refpack.plan_ref_geometry(864, 480, "match", 320, 240), (320, 256))
        with self.assertRaisesRegex(ValueError, "ref_image_size"):
            refpack.plan_ref_geometry(864, 480, "stretch", 1920, 1080)

    def test_max_geometry_caps_the_short_edge_at_2048(self):
        self.assertEqual(refpack.plan_ref_geometry(864, 480, "max", 4000, 3000), (2720, 2048))
        self.assertEqual(refpack.plan_ref_geometry(864, 480, "max", 1000, 500), (992, 512))
        self.assertEqual(refpack.plan_ref_geometry(864, 480, "max", 320, 240), (320, 256))

    def test_reference_video_canvas_ignores_the_image_mode(self):
        self.assertEqual(refpack.plan_video_geometry(1920, 1080), (1344, 768))
        self.assertEqual(refpack.plan_video_geometry(320, 240), (320, 256))
        self.assertEqual(refpack.trim_video_frames(124, 124), 124)
        self.assertEqual(refpack.trim_video_frames(130, 124), 124)
        self.assertEqual(refpack.trim_video_frames(20, 124), 5)
        self.assertEqual(refpack.trim_video_frames(21, 124), 5)
        self.assertEqual(refpack.trim_video_frames(23, 124), 22)
        self.assertEqual(refpack.trim_video_frames(39, 124), 39)
        with self.assertRaisesRegex(ValueError, "at least 5 frames"):
            refpack.trim_video_frames(4, 124)

    def test_layout_segments_and_rope_clock_match_the_reference(self):
        refs = [_image_block(), _image_block(), _video_block(), _audio_block()]
        layout = refpack.build_packed_layout(8, 37, 30, 54, 207, refs,
                                             video_timestep=0.4, audio_timestep=0.2)
        self.assertEqual(layout.seq_len, 40347)
        self.assertEqual(layout.segments, (
            (0, 8, "text"), (8, 1040, "ref_img"), (1040, 2072, "ref_img"),
            (2072, 2472, "ref_audio"), (2472, 24648, "ref_img"),
            (24648, 24948, "ref_audio"), (24948, 25362, "audio"),
            (25362, 40347, "video")))
        self.assertEqual(layout.position_ids.dtype, torch.float64)
        self.assertEqual(tuple(layout.position_ids.shape), (40347, 3))
        self.assertEqual(layout.position_ids[:8, 0].tolist(), list(range(8)))
        self.assertEqual(layout.position_ids[8, 0].item(), 8.0)
        self.assertEqual(layout.position_ids[1040, 0].item(), 9.0)
        self.assertEqual(layout.position_ids[2072, 0].item(), 10.0)
        self.assertEqual(layout.position_ids[24948, 0].item(), 360.0)
        self.assertEqual(layout.position_ids[25362, 0].item(), 360.0)
        self.assertAlmostEqual(layout.position_ids[-1, 0].item(), 560.0, places=9)

    def test_layout_position_and_update_hashes_match_the_reference(self):
        # Oracle hashes from ComfyUI efa6c8f8 PackedLayout for the case above.
        refs = [_image_block(), _image_block(), _video_block(), _audio_block()]
        layout = refpack.build_packed_layout(8, 37, 30, 54, 207, refs,
                                             video_timestep=0.4, audio_timestep=0.2)
        def digest(tensor):
            return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        self.assertEqual(digest(layout.position_ids),
                         "5d2e7f61ceb7207c9a7c8436407e03f16be6fc624aa751e8ad312204531e37d1")
        self.assertEqual(digest(layout.img_pos),
                         "5190e6f7704452c412183215cfe281fd0840ba2db8131a92f79cd6f659ce2494")
        self.assertEqual(digest(layout.img_update),
                         "354e044663375a3c5f11d63c8e174e535781d0da9bf3a69fa6d92783b43a978a")
        self.assertEqual(digest(layout.audio_pos),
                         "360fbe221b3ade70020a154a04683b260807924c66f9b25cf16c2019490c58be")
        self.assertEqual(digest(layout.audio_update),
                         "437862da1c39ad76405b84a6c8fcd7055f691b67c6586d353716061ac214924b")
        self.assertEqual(layout.img_pos[:3].tolist(), [8, 9, 10])
        self.assertEqual(layout.audio_pos[:3].tolist(), [2072, 2073, 2074])
        self.assertTrue(layout.img_update[-1].item())
        self.assertFalse(layout.img_update[0].item())
        self.assertTrue(layout.audio_update[-1].item())
        self.assertFalse(layout.audio_update[0].item())

    def test_nine_reference_layout_matches_the_reference_hash(self):
        layout = refpack.build_packed_layout(
            56, 37, 30, 54, 207, [_image_block() for _ in range(9)])
        self.assertEqual(layout.seq_len, 24743)
        self.assertEqual(layout.segments[0], (0, 56, "text"))
        self.assertEqual(layout.segments[-2], (9344, 9758, "audio"))
        self.assertEqual(hashlib.sha256(layout.position_ids.numpy().tobytes()).hexdigest(),
                         "4a33c1f2eb48788ab1fb521175924078a1eb5257328f3d95ba2a6a2b856d9000")

    def test_timestep_classes_and_adaln_indices(self):
        refs = [_image_block(), _video_block(), _audio_block()]
        layout = refpack.build_packed_layout(8, 37, 30, 54, 207, refs,
                                             video_timestep=0.4, audio_timestep=0.2)
        torch.testing.assert_close(layout.timestep_values,
                                   torch.tensor([0.2, 0.4, 0.999, 1.0]), atol=0, rtol=0)
        text = layout.segments[0]
        ref_img = layout.segments[1]
        ref_audio = layout.segments[2]
        target_audio = layout.segments[-2]
        target_video = layout.segments[-1]
        self.assertTrue(torch.all(layout.timestep_indices[text[0]:text[1]] == 1))
        self.assertTrue(torch.all(layout.timestep_indices[ref_img[0]:ref_img[1]] == 2))
        self.assertTrue(torch.all(layout.timestep_indices[ref_audio[0]:ref_audio[1]] == 3))
        self.assertTrue(torch.all(layout.timestep_indices[target_audio[0]:target_audio[1]] == 0))
        self.assertTrue(torch.all(layout.timestep_indices[target_video[0]:target_video[1]] == 1))
        self.assertTrue(torch.all(layout.token_tags[text[0]:text[1]] == 1))
        self.assertTrue(torch.all(layout.token_tags[ref_img[0]:ref_img[1]] == 0))
        self.assertTrue(torch.all(layout.token_tags[ref_audio[0]:ref_audio[1]] == 2))
        self.assertTrue(torch.all(layout.token_tags[target_audio[0]:target_audio[1]] == 2))
        self.assertTrue(torch.equal(layout.adaln_indices,
                                    layout.timestep_indices*3 + layout.token_tags))
        self.assertEqual(layout.adaln_indices[0].item(), 4)
        self.assertEqual(layout.adaln_indices[ref_img[0]].item(), 6)
        self.assertEqual(layout.adaln_indices[ref_audio[0]].item(), 11)
        self.assertEqual(layout.adaln_indices[target_audio[0]].item(), 2)

    def test_global_adaln_values_index_the_cached_table(self):
        refs = [_image_block(), _video_block(), _audio_block()]
        values = torch.tensor([0.0, 0.2, 0.4, 0.999, 1.0], dtype=torch.float32)
        layout = refpack.build_packed_layout(8, 37, 30, 54, 207, refs,
                                             video_timestep=0.4, audio_timestep=0.2,
                                             adaln_values=values)
        torch.testing.assert_close(layout.timestep_values, values, atol=0, rtol=0)
        text = layout.segments[0]
        ref_img = layout.segments[1]
        ref_audio = layout.segments[2]
        target_audio = layout.segments[-2]
        target_video = layout.segments[-1]
        self.assertTrue(torch.all(layout.timestep_indices[text[0]:text[1]] == 2))
        self.assertTrue(torch.all(layout.timestep_indices[ref_img[0]:ref_img[1]] == 3))
        self.assertTrue(torch.all(layout.timestep_indices[ref_audio[0]:ref_audio[1]] == 4))
        self.assertTrue(torch.all(layout.timestep_indices[target_audio[0]:target_audio[1]] == 1))
        self.assertTrue(torch.all(layout.timestep_indices[target_video[0]:target_video[1]] == 2))
        self.assertLess(int(layout.adaln_indices.max()), len(values)*3)
        table = torch.arange(len(values)*3*48, dtype=torch.float32).reshape(len(values)*3, 48)
        rows = table[layout.adaln_indices]
        self.assertEqual(tuple(rows.shape), (layout.seq_len, 48))
        self.assertTrue(torch.equal(rows[0], table[2*3+1]))

    def test_global_adaln_values_must_be_sorted_and_complete(self):
        with self.assertRaisesRegex(ValueError, "sorted"):
            refpack.build_packed_layout(8, 5, 4, 4, 5, [], video_timestep=0.4,
                                         audio_timestep=0.2,
                                         adaln_values=torch.tensor([0.4, 0.2, 0.999, 1.0]))
        with self.assertRaisesRegex(ValueError, "missing"):
            refpack.build_packed_layout(8, 5, 4, 4, 5, [], video_timestep=0.4,
                                         audio_timestep=0.2,
                                         adaln_values=torch.tensor([0.0, 0.2, 0.999, 1.0]))

    def test_text_token_tags_split_vision_rows(self):
        tags = torch.tensor([1, 0, 0, 2, 1, 0, 1, 1])
        layout = refpack.build_packed_layout(8, 5, 4, 4, 5, [], text_token_tags=tags,
                                             video_timestep=0.6, audio_timestep=0.3)
        self.assertTrue(torch.equal(layout.adaln_indices[:8],
                                    layout.timestep_indices[:8]*3 + tags))

    def test_ref_timestep_pins_references(self):
        self.assertEqual(refpack.ref_timestep(0.0), 0.999)
        self.assertEqual(refpack.ref_timestep(0.5), 0.999)
        self.assertEqual(refpack.ref_timestep(0.999), 0.999)
        self.assertEqual(refpack.ref_timestep(1.0), 1.0)
        self.assertEqual(refpack.ref_audio_timestep(0.7), 1.0)
        self.assertEqual(refpack.ref_audio_timestep(1.0), 1.0)

    def test_rope_is_float32_with_the_reference_half_split(self):
        inv = torch.arange(1, 17, dtype=torch.float32)
        layout = refpack.build_packed_layout(4, 5, 4, 4, 5, [],
                                             video_timestep=0.5, audio_timestep=0.25,
                                             inv_freq=inv)
        self.assertEqual(layout.rope.dtype, torch.float32)
        self.assertEqual(tuple(layout.rope.shape), (layout.seq_len, 96))
        per_axis = layout.position_ids.to(torch.float32).unsqueeze(-1)*inv
        t, h, w = per_axis.unbind(dim=1)
        half = torch.cat((t, h, w), dim=-1)
        torch.testing.assert_close(layout.rope, torch.cat((half, half), dim=-1),
                                   atol=0, rtol=0)

    def test_signature_keys_on_the_reference_set(self):
        first = refpack.build_packed_layout(8, 37, 30, 54, 207, [_image_block()])
        same = refpack.build_packed_layout(8, 37, 30, 54, 207, [_image_block()])
        other = refpack.build_packed_layout(8, 37, 30, 54, 207, [_video_block()])
        self.assertEqual(first.signature, same.signature)
        self.assertNotEqual(first.signature, other.signature)
        tagged = refpack.build_packed_layout(8, 37, 30, 54, 207, [_image_block()],
                                             video_timestep=0.5, audio_timestep=0.2,
                                             text_token_tags=torch.zeros(8, dtype=torch.long))
        self.assertNotEqual(first.signature, tagged.signature)

    def test_the_49152_token_cap_is_enforced(self):
        refs = [_image_block() for _ in range(9)]
        cap = refpack.build_packed_layout(56, 37, 30, 54, 207, refs)
        self.assertEqual(cap.seq_len, 24743)
        text_len = 49152 - cap.seq_len + 56
        exact = refpack.build_packed_layout(text_len, 37, 30, 54, 207, refs)
        self.assertEqual(exact.seq_len, 49152)
        with self.assertRaisesRegex(ValueError, "exceeds the 49152-token cap"):
            refpack.build_packed_layout(text_len+1, 37, 30, 54, 207, refs)
        with self.assertRaisesRegex(ValueError, "exceeds the 1000-token cap"):
            refpack.build_packed_layout(56, 37, 30, 54, 207, refs, max_tokens=1000)

    def test_768p_nine_reference_match_is_over_cap(self):
        # Synthetic layout matching the measured 1344x768 nine-ref match
        # total: 37,296 video + 414 audio + 9*1032 refs + 6,067 text.
        refs = [_image_block() for _ in range(9)]
        layout = refpack.build_packed_layout(6067, 37, 48, 84, 207, refs,
                                             max_tokens=60000)
        self.assertEqual(layout.seq_len, 53065)
        with self.assertRaisesRegex(ValueError, "exceeds the 49152-token cap"):
            refpack.build_packed_layout(6067, 37, 48, 84, 207, refs)

    def test_gate_sizes_are_constructible(self):
        expected = {
            20000: "8b73aee8bf5862027d4303b3c4bd7de10adead5dcd4cab4ec6b4567e235b8279",
            24743: "4a33c1f2eb48788ab1fb521175924078a1eb5257328f3d95ba2a6a2b856d9000",
            36900: "af6798501ed2b97b53490f66e63331a73f94b5ccf59bb46b5f33d1bd5460410a",
            42900: "ef47622c253cd1f2c30dc614699663fc21282097aa02fff393c3ee859a89258f",
            45200: "0e073f2330fa0a4be2362756c25aa983b0c86982a6d5249a159ba438c1fd0c1d",
            49152: "bad3a743945e9f18fd5d27474ebb98ff1c75c3b42ef6abe1b245885a17bae831",
        }
        for target, count in ((20000, 4), (24743, 9), (36900, 9), (42900, 9),
                              (45200, 9), (49152, 9)):
            refs = [_image_block() for _ in range(count)]
            text_len = target - 15399 - count*1032
            layout = refpack.build_packed_layout(text_len, 37, 30, 54, 207, refs)
            self.assertEqual(layout.seq_len, target)
            self.assertEqual(hashlib.sha256(layout.position_ids.numpy().tobytes()).hexdigest(),
                             expected[target])


class RefNodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        path = RELEASE/"comfyui/paiton_h3/__init__.py"
        spec = importlib.util.spec_from_file_location("paiton_h3_node", path)
        cls.node = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.node)

    def test_node_registers_nine_image_slots(self):
        self.assertIn("PaitonH3Ref2VA", self.node.NODE_CLASS_MAPPINGS)
        schema = self.node.PaitonH3Ref2VA.INPUT_TYPES()
        optional = schema["optional"]
        for index in range(9):
            self.assertEqual(optional[f"ref_image_{index}"], ("IMAGE",))
        for index in range(3):
            self.assertEqual(optional[f"ref_video_{index}"], ("IMAGE",))
            self.assertEqual(optional[f"ref_video_audio_{index}"], ("AUDIO",))
            self.assertEqual(optional[f"ref_audio_{index}"], ("AUDIO",))

    def test_node_forwards_reference_slots_to_the_core_node(self):
        calls = []
        class Core:
            @classmethod
            def execute(cls, **kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace(result=("COND", "LATENT"))
        core_module = types.ModuleType("comfy_extras.nodes_minimax_h3")
        core_module.MiniMaxH3ReferenceToVideo = Core
        sys.modules["comfy_extras"] = types.ModuleType("comfy_extras")
        sys.modules["comfy_extras.nodes_minimax_h3"] = core_module
        image = torch.zeros(1, 4, 4, 3)
        video = torch.zeros(5, 4, 4, 3)
        audio = {"waveform": torch.zeros(1, 2, 400), "sample_rate": 32000}
        result = self.node.PaitonH3Ref2VA().apply(
            "clip", "vae", "audio_vae", "prompt", 864, 480, 124, "match",
            ref_image_8=image, ref_video_1=video, ref_video_audio_1=audio,
            ref_audio_2=audio)
        self.assertEqual(result, ("COND", "LATENT"))
        self.assertEqual(calls[0]["ref_images"], {"ref_image_8": image})
        self.assertEqual(calls[0]["ref_videos"], {"ref_video_1": video})
        self.assertEqual(calls[0]["ref_video_audios"], {"ref_video_audio_1": audio})
        self.assertEqual(calls[0]["ref_audios"], {"ref_audio_2": audio})
        self.assertEqual(calls[0]["width"], 864)

    def _apply_with_layout(self, text_len, latent_t, launch=True, latent_h=30, latent_w=54):
        from paiton_video import projection
        refs = [{"kind": "image", "latent_h": 48, "latent_w": 86} for _ in range(9)]
        class Samples:
            tensors = (types.SimpleNamespace(shape=(1, 24, latent_t, latent_h, latent_w)),
                       types.SimpleNamespace(shape=(1, 32, 2, 207)))
        class Core:
            @classmethod
            def execute(cls, **kwargs):
                cond = [[types.SimpleNamespace(shape=(1, text_len, 5120)),
                         {"minimax_refs": refs}]]
                return types.SimpleNamespace(result=(cond, {"samples": Samples()}))
        core_module = types.ModuleType("comfy_extras.nodes_minimax_h3")
        core_module.MiniMaxH3ReferenceToVideo = Core
        sys.modules["comfy_extras"] = types.ModuleType("comfy_extras")
        sys.modules["comfy_extras.nodes_minimax_h3"] = core_module
        saved = projection._launch, projection.max_tokens
        projection._launch = object() if launch else None
        projection.max_tokens = 49152 if launch else 32768
        try:
            return self.node.PaitonH3Ref2VA().apply(
                "clip", "vae", "audio_vae", "prompt", 864, 480, 124, "match")
        finally:
            projection._launch, projection.max_tokens = saved

    def test_node_accepts_the_nine_reference_budget(self):
        result = self._apply_with_layout(56, 37)
        self.assertEqual(result[0][0][1]["minimax_refs"][0]["latent_h"], 48)

    def test_node_rejects_an_over_cap_reference_layout(self):
        with self.assertRaisesRegex(ValueError, "exceeds the 49152-token cap"):
            self._apply_with_layout(30000, 100)

    def test_preflight_rejects_over_cap_without_engine_node(self):
        # The R2V node can execute before PaitonH3Optimize; the cap must not
        # depend on projection._launch having been installed.
        with self.assertRaisesRegex(ValueError, "exceeds the 49152-token cap"):
            self._apply_with_layout(30000, 100, launch=False)

    def test_preflight_rejects_53k_before_engine_node(self):
        # Exact 1344x768 nine-ref match request, R2V before PaitonH3Optimize.
        with self.assertRaisesRegex(ValueError,
                                    r"packed sequence 53065 exceeds the 49152-token cap"):
            self._apply_with_layout(6067, 37, launch=False, latent_h=48, latent_w=84)

    def test_free_memory_node_passes_conditioning_through(self):
        calls = []
        module = types.ModuleType("comfy.model_management")
        module.unload_all_models = lambda: calls.append("unload")
        module.soft_empty_cache = lambda force=False: calls.append(("empty", force))
        sys.modules["comfy.model_management"] = module
        conditioning = [[torch.zeros(1, 2), {}]]
        result = self.node.PaitonH3FreeMemory().apply(conditioning)
        self.assertIs(result[0], conditioning)
        self.assertEqual(calls, ["unload", ("empty", True)])
        self.assertIn("PaitonH3FreeMemory", self.node.NODE_CLASS_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
