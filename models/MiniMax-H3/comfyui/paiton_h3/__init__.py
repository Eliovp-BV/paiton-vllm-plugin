"""H3 artifact selection and explicit, separate video/audio VAE precisions."""
from pathlib import Path
import torch
import folder_paths
import comfy.sd
import comfy.utils
import nodes
from paiton_video import projection
from .settings import video_settings


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
            with projection.projection_scope():
                if previous is not None:
                    return previous(model_function,arguments)
                return model_function(arguments["input"],arguments["timestep"],**arguments["c"])
        result.set_model_unet_function_wrapper(wrapper)
        return (result,)


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


class PaitonH3VideoSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "duration_seconds": ("INT", {"default": 15, "min": 5, "max": 15, "step": 1, "display": "slider",
                "tooltip": "Requested seconds at 24 fps. H3 rounds up to its supported frame grid; 15 seconds produces 362 frames (15.08 seconds)."}),
            "resolution": ("INT", {"default": 3, "min": 1, "max": 3, "step": 1, "display": "slider",
                "tooltip": "Landscape: 1 = 576×320, 2 = 704×384, 3 = 864×480. Portrait swaps the axes; Square uses the shorter edge."}),
            "aspect": (["Landscape", "Portrait", "Square"],),
        }}
    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "frames")
    FUNCTION = "settings"
    CATEGORY = "Paiton/video"

    def settings(self, duration_seconds, resolution, aspect):
        return video_settings(duration_seconds, resolution, aspect)


class PaitonH3InputImage:
    """Use ComfyUI's normal image upload/decoder, with an explicit empty option."""
    @classmethod
    def INPUT_TYPES(cls):
        files = nodes.LoadImage.INPUT_TYPES()["required"]["image"][0]
        return {"required": {"image": (["(none)"] + list(files), {"image_upload": True})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "Paiton/video"

    def load(self, image):
        if image == "(none)":
            return (None,)
        return (nodes.LoadImage().load_image(image)[0],)

    @classmethod
    def IS_CHANGED(cls, image):
        return "none" if image == "(none)" else nodes.LoadImage.IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        return True if image == "(none)" else nodes.LoadImage.VALIDATE_INPUTS(image)


NODE_CLASS_MAPPINGS={"PaitonH3Optimize":PaitonH3Optimize,"PaitonH3VAELoader":PaitonH3VAELoader,
                    "PaitonH3VideoSettings":PaitonH3VideoSettings,"PaitonH3InputImage":PaitonH3InputImage}
NODE_DISPLAY_NAME_MAPPINGS={"PaitonH3Optimize":"MiniMax H3 Engine","PaitonH3VAELoader":"MiniMax H3 VAE",
                           "PaitonH3VideoSettings":"Video length and resolution","PaitonH3InputImage":"Optional input image"}

WEB_DIRECTORY="./web"
