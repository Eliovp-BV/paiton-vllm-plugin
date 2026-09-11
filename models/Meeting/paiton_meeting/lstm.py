"""Torch integration for a Torch-independent Paiton prediction-cell artifact."""
import ctypes
import hashlib
import json
from pathlib import Path
import torch

class PaitonLSTM(torch.nn.Module):
    def __init__(self,stock,artifact):
        super().__init__()
        self.stock=stock
        if stock.input_size!=640 or stock.hidden_size!=640 or stock.num_layers!=2 or stock.bidirectional or not stock.batch_first:
            raise ValueError('Unsupported recurrent network geometry.')
        artifact=Path(artifact)
        manifest=json.loads(artifact.with_suffix('.json').read_text())
        if hashlib.sha256(artifact.read_bytes()).hexdigest()!=manifest['sha256']:
            raise ValueError('Prediction-cell artifact hash mismatch.')
        self.dtype=stock.weight_ih_l0.dtype
        expected='float16' if self.dtype==torch.float16 else 'float32'
        if manifest['dtype']!=expected or manifest['gpu_arch']!='gfx1201':
            raise ValueError('Prediction-cell artifact dtype/target mismatch.')
        device=torch.cuda.get_device_properties(stock.weight_ih_l0.device)
        if device.gcnArchName.split(':')[0]!='gfx1201':raise ValueError('Prediction-cell artifact requires gfx1201.')
        self.lib=ctypes.CDLL(str(artifact))
        self.fn=getattr(self.lib,manifest['abi']);self.fn.argtypes=[ctypes.c_void_p]*11;self.fn.restype=ctypes.c_int

    def forward(self,x,hx=None):
        if self.training or x.shape!=(1,1,640) or x.dtype!=self.dtype or not x.is_cuda or not x.is_contiguous():
            return self.stock(x,hx)
        if hx is None:
            h=torch.zeros(2,1,640,device=x.device,dtype=x.dtype);c=torch.zeros_like(h)
        else:h,c=hx
        if not h.is_contiguous() or not c.is_contiguous():return self.stock(x,hx)
        hout=torch.empty_like(h);cout=torch.empty_like(c)
        gates=torch.empty(2560,device=x.device,dtype=torch.float32)
        stream=torch.cuda.current_stream(x.device).cuda_stream
        value=x
        for layer in range(2):
            tensors=[value,h[layer],c[layer],getattr(self.stock,f'weight_ih_l{layer}'),getattr(self.stock,f'weight_hh_l{layer}'),getattr(self.stock,f'bias_ih_l{layer}'),getattr(self.stock,f'bias_hh_l{layer}'),hout[layer],cout[layer],gates]
            status=self.fn(*[t.data_ptr() for t in tensors],stream)
            if status:raise RuntimeError('Paiton prediction-cell execution failed: '+str(status))
            value=hout[layer]
        return value.reshape(1,1,640),(hout,cout)
