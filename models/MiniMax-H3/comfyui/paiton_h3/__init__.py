"""H3 artifact selection and explicit, separate video/audio VAE precisions."""
from pathlib import Path
import torch
import folder_paths
import comfy.sd
import comfy.utils
from paiton_video import adaln, fusions, projection, refpack

adaln.install_checkpoint_loader()


class PaitonH3Optimize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"model":("MODEL",),"engine":(["paiton","stock"],)}}
    RETURN_TYPES=("MODEL",)
    FUNCTION="apply"
    CATEGORY="Paiton/video"

    def apply(self,model,engine):
        from comfy.ldm.minimax.model import MiniMaxH3Model
        if not isinstance(model.model.diffusion_model,MiniMaxH3Model):
            raise ValueError("Paiton H3 requires a MiniMax H3 model")
        if engine=="stock":
            return (model,)
        if projection._launch is None:
            artifact=Path(projection.__file__).resolve().parents[1]/"artifacts/minimax_h3_projections_gfx1201.so"
            projection.install_projections(artifact)
        result=model.clone()
        previous=result.model_options.get("model_function_wrapper")
        def wrapper(model_function,arguments):
            payload=(arguments.get("c") or {}).get("minimax_payload") or {}
            if not isinstance(payload,dict):
                payload={}
            layout=payload.get("layout")
            before=fusions.calls["attention"]
            fusions.note_model_step(bool(payload.get("refs")))
            with projection.projection_scope():
                if previous is not None:
                    out=previous(model_function,arguments)
                else:
                    out=model_function(arguments["input"],arguments["timestep"],**arguments["c"])
            if layout is not None and layout.seq_len >= 1024:
                calls=fusions.calls["attention"]-before
                if calls != fusions.H3_BLOCKS:
                    raise AssertionError(
                        f"H3 attention calls {calls} != {fusions.H3_BLOCKS} for {layout.seq_len} tokens")
            return out
        result.set_model_unet_function_wrapper(wrapper)
        return (result,)


class PaitonH3Ref2VA:
    """Nine-slot reference conditioning, forwarded to ComfyUI's H3 R2V node."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for index in range(9):
            optional[f"ref_image_{index}"] = ("IMAGE",)
        for index in range(3):
            optional[f"ref_video_{index}"] = ("IMAGE",)
            optional[f"ref_video_audio_{index}"] = ("AUDIO",)
            optional[f"ref_audio_{index}"] = ("AUDIO",)
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    FUNCTION = "apply"
    CATEGORY = "Paiton/video"

    def apply(self, clip, vae, audio_vae, prompt, width, height, length,
              ref_image_size="match", **refs):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

        def collect(prefix, count):
            return {f"{prefix}{index}": refs[f"{prefix}{index}"]
                    for index in range(count) if refs.get(f"{prefix}{index}") is not None}

        output = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, vae=vae, audio_vae=audio_vae, prompt=prompt,
            width=width, height=height, length=length, ref_image_size=ref_image_size,
            ref_images=collect("ref_image_", 9),
            ref_videos=collect("ref_video_", 3),
            ref_video_audios=collect("ref_video_audio_", 3),
            ref_audios=collect("ref_audio_", 3))
        cond, latent = output.result
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if samples is not None and hasattr(samples, "tensors"):
            video, audio = samples.tensors
            # Fail before sampling if the request cannot fit the qualified kernel.
            # This runs even when the engine node is scheduled later.
            refpack.build_packed_layout(
                cond[0][0].shape[1], video.shape[2], video.shape[3], video.shape[4],
                audio.shape[-1], cond[0][1].get("minimax_refs", []),
                video_timestep=0.0, audio_timestep=0.0,
                max_tokens=projection.advertised_max_tokens())
        return cond, latent


class PaitonH3AdaLN:
    """Replace per-block AdaLN projections with the pruned cached tables."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "tables": ("STRING", {"default": ""}),
            "preset": (["base50", "turbo4", "turbo8"], {"default": "base50"}),
        }, "optional": {
            "mode": (["pruned_8dim", "cached_tables"], {"default": "pruned_8dim"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Paiton/video"

    def apply(self, model, tables, preset, mode="pruned_8dim"):
        from comfy.ldm.minimax.model import MiniMaxH3Model
        diffusion = model.model.diffusion_model
        if not isinstance(diffusion, MiniMaxH3Model):
            raise ValueError("Paiton AdaLN requires a MiniMax H3 model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if mode == "pruned_8dim":
            adaln.install_pruned(diffusion, adaln.load_pruned_adaln(tables, device=device))
        elif mode == "cached_tables":
            adaln.install(diffusion, adaln.load_tables(tables, preset, device=device))
        else:
            raise ValueError(f"Unknown H3 AdaLN mode: {mode}")
        return (model,)


class PaitonH3FreeMemory:
    """Drop the text encoder/VAE residency after conditioning, before sampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",)}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "Paiton/video"

    def apply(self, conditioning):
        import comfy.model_management as model_management
        model_management.unload_all_models()
        model_management.soft_empty_cache(force=True)
        return (conditioning,)


class PaitonH3VAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"component":(["video","audio"],)}}
    RETURN_TYPES=("VAE",)
    FUNCTION="load"
    CATEGORY="Paiton/video"

    def load(self,component):
        name=("minimax_h3_video_vae_int8_convrot.safetensors" if component=="video"
              else "minimax_h3_audio_vae_fp32.safetensors")
        sd,metadata=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae",name),return_metadata=True)
        return (comfy.sd.VAE(sd=sd,metadata=metadata,
                            dtype=torch.bfloat16 if component=="video" else torch.float32),)


NODE_CLASS_MAPPINGS={"PaitonH3Optimize":PaitonH3Optimize,"PaitonH3Ref2VA":PaitonH3Ref2VA,
                     "PaitonH3AdaLN":PaitonH3AdaLN,"PaitonH3FreeMemory":PaitonH3FreeMemory,
                     "PaitonH3VAELoader":PaitonH3VAELoader}
NODE_DISPLAY_NAME_MAPPINGS={"PaitonH3Optimize":"MiniMax H3 Engine",
                            "PaitonH3Ref2VA":"MiniMax H3 Reference to Video (Paiton)",
                            "PaitonH3AdaLN":"MiniMax H3 Cached AdaLN",
                            "PaitonH3FreeMemory":"MiniMax H3 Free Memory",
                            "PaitonH3VAELoader":"MiniMax H3 VAE"}
