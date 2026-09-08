"""H3 artifact selection and explicit, separate video/audio VAE precisions."""
from pathlib import Path
import torch
import folder_paths
import comfy.sd
import comfy.utils
from paiton_video import projection


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


NODE_CLASS_MAPPINGS={"PaitonH3Optimize":PaitonH3Optimize,"PaitonH3VAELoader":PaitonH3VAELoader}
NODE_DISPLAY_NAME_MAPPINGS={"PaitonH3Optimize":"MiniMax H3 Engine","PaitonH3VAELoader":"MiniMax H3 VAE"}

WEB_DIRECTORY="./web"
