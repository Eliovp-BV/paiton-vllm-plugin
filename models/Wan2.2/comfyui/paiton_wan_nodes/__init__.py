"""Separate Wan model selection, explicit presets and optional image input."""
from pathlib import Path
import nodes
from paiton_wan.dmd import sample

class PaitonWanLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'preset':(['fast','base'],),'engine':(['stock','paiton'],)}}
    RETURN_TYPES=('MODEL','CLIP','VAE');FUNCTION='load';CATEGORY='Paiton/Wan'
    def load(self,preset,engine):
        from paiton_wan.loading import load
        return load(preset,engine,compile_blocks=True)

class PaitonWanBaseLoader(PaitonWanLoader):
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'engine':(['stock','paiton'],)}}
    def load(self,engine):return super().load('base',engine)

class PaitonWanFastLoader(PaitonWanLoader):
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'engine':(['stock','paiton'],)}}
    def load(self,engine):return super().load('fast',engine)

class PaitonWanTextEncode(nodes.CLIPTextEncode):
    CATEGORY='Paiton/Wan'
    @classmethod
    def IS_CHANGED(cls,**kwargs):
        # Keep loaded models cached, but compute conditioning for every request.
        return float('nan')

class PaitonWanSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'duration':(['2 seconds (49 frames)','5 seconds (121 frames)'],),
            'resolution':(['480 class (832 x 480)','720 class (1280 x 704)'],),
            'orientation':(['Landscape','Portrait'],)}}
    RETURN_TYPES=('INT','INT','INT');RETURN_NAMES=('width','height','frames')
    FUNCTION='settings';CATEGORY='Paiton/Wan'
    def settings(self,duration,resolution,orientation):
        w,h=(832,480) if resolution.startswith('480') else (1280,704)
        if orientation=='Portrait':w,h=h,w
        return w,h,49 if duration.startswith('2 ') else 121

class PaitonWanBaseSettings(PaitonWanSettings):
    @classmethod
    def INPUT_TYPES(cls):
        values=super().INPUT_TYPES()
        values['required']['resolution']=(['480 class (832 x 480)'],)
        return values

class PaitonWanFastSettings(PaitonWanSettings):
    @classmethod
    def INPUT_TYPES(cls):
        values=super().INPUT_TYPES()
        values['required']['orientation']=(['Landscape'],)
        values['required']['resolution']=(['720 class (1280 x 704)','480 class (832 x 480)'],)
        return values

class PaitonWanInputImage:
    @classmethod
    def INPUT_TYPES(cls):
        images=nodes.LoadImage.INPUT_TYPES()['required']['image'][0]
        return {'required':{'image':(['(none)']+list(images),{'image_upload':True})}}
    RETURN_TYPES=('IMAGE',);FUNCTION='load';CATEGORY='Paiton/Wan'
    def load(self,image):return (None,) if image=='(none)' else (nodes.LoadImage().load_image(image)[0],)
    @classmethod
    def IS_CHANGED(cls,image):return 'none' if image=='(none)' else nodes.LoadImage.IS_CHANGED(image)
    @classmethod
    def VALIDATE_INPUTS(cls,image):return True if image=='(none)' else nodes.LoadImage.VALIDATE_INPUTS(image)

class PaitonFastWanSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'model':('MODEL',),'positive':('CONDITIONING',),
                            'seed':('INT',{'default':1201,'min':0,'max':0xffffffffffffffff}),
                            'width':('INT',{'default':832,'min':480,'max':1280,'step':32}),
                            'height':('INT',{'default':480,'min':480,'max':1280,'step':32}),
                            'frames':('INT',{'default':49,'min':49,'max':121,'step':4})}}
    RETURN_TYPES=('LATENT',);FUNCTION='generate';CATEGORY='Paiton/Wan'
    def generate(self,model,positive,seed,width,height,frames):
        from comfy.utils import ProgressBar
        progress=ProgressBar(3)
        return (sample(model,positive,seed,width,height,frames,lambda i,t:progress.update_absolute(i,3)),)

NODE_CLASS_MAPPINGS={c.__name__:c for c in (PaitonWanTextEncode,PaitonWanBaseLoader,PaitonWanFastLoader,PaitonWanBaseSettings,PaitonWanFastSettings,PaitonWanInputImage,PaitonFastWanSampler)}
NODE_DISPLAY_NAME_MAPPINGS={'PaitonWanBaseLoader':'Wan2.2 base TI2V engine','PaitonWanFastLoader':'FastWan text engine','PaitonWanTextEncode':'Prompt','PaitonWanBaseSettings':'Image/video duration (480 class)','PaitonWanFastSettings':'Video duration and resolution',
                          'PaitonWanInputImage':'Optional image (base TI2V only)','PaitonFastWanSampler':'FastWan three-step DMD (text only)'}

WEB_DIRECTORY="./web"
