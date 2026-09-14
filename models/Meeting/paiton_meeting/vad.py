"""CPU VAD skips only chunks with no detected speech; ASR keeps original clocks."""
import torch

class SileroVAD:
    def __init__(self,path):
        self.model=torch.jit.load(str(path),map_location='cpu').eval()
    def has_speech(self,samples):
        self.model.reset_states()
        with torch.inference_mode():
            for offset in range(0,len(samples),512):
                piece=torch.from_numpy(samples[offset:offset+512])
                if piece.numel()<512:piece=torch.nn.functional.pad(piece,(0,512-piece.numel()))
                if float(self.model(piece,16000))>=.5:return True
        return False
