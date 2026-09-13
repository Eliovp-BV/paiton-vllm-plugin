"""vLLM image processing with native Paiton GGUF vision and language execution."""
import ctypes
import hashlib
import json
import os
import weakref
from pathlib import Path

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_5 import Qwen3_5ProcessingInfo
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tokenizers.registry import cached_tokenizer_from_config

from paiton_vllm_plugin.gguf_source import GGUFTensorSource
from .paiton_qwen38_gguf import PaitonQwen38GGUFForCausalLM, _copy_tensor_bytes
from .paiton_qwen38_multimodal import PaitonQwen38ForConditionalGeneration

logger=init_logger('vllm.paiton.gguf.vision')


class _MultimodalGGUFBackbone(PaitonQwen38GGUFForCausalLM):
    _gguf_multimodal = True


class NativeGGUFVision(nn.Module):
    """Storage and C ABI only; all encoder math executes in the native artifact."""
    spatial_merge_size=2
    dtype=torch.float32

    def __init__(self, model_path):
        super().__init__()
        self.contract=json.loads((model_path/'vision-runtime.json').read_text())
        c=self.contract
        if (c.get('abi_version'),c.get('gpu_arch'),c.get('max_patches'),c.get('gelu')) != (1,'gfx1201',4096,'tanh'):
            raise ValueError('Unsupported native GGUF vision contract')
        binary=model_path/'libpaiton_qwen38_vision.so'
        if hashlib.sha256(binary.read_bytes()).hexdigest()!=c['sha256']:
            raise ValueError('Native vision artifact SHA256 mismatch')
        self.native=ctypes.CDLL(str(binary))
        p,i,z=ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t
        if self.native.paiton_vision_abi_version()!=1:
            raise ValueError('Native vision exported ABI mismatch')
        self.native.paiton_vision_create.argtypes=[ctypes.POINTER(p)]
        self.native.paiton_vision_destroy.argtypes=[p]
        self.native.paiton_vision_workspace_bytes.argtypes=[i]
        self.native.paiton_vision_workspace_bytes.restype=z
        self.native.paiton_vision_run.argtypes=[p,ctypes.POINTER(p),p,p,i,i,p,z,p,i]
        self.handle=p()
        if self.native.paiton_vision_create(ctypes.byref(self.handle)):
            raise RuntimeError('Native vision rocBLAS handle creation failed')
        self._finalizer=weakref.finalize(self,self.native.paiton_vision_destroy,self.handle)
        self._weights=[]
        self._logged=False

    @property
    def device(self):
        return self._weights[0].device

    def load_weights(self, checkpoint):
        c=self.contract['checkpoint']
        with GGUFTensorSource(checkpoint,sha256=c['sha256'],size_bytes=c['size_bytes']) as source:
            if len(source.tensors)!=len(self.contract['bindings']):
                raise ValueError('Projector tensor count mismatch')
            for record in self.contract['bindings']:
                actual=source.tensors[record['name']]
                if any(actual[k]!=record[k] for k in ('dimensions','type_id','size_bytes')):
                    raise ValueError('Projector tensor layout mismatch')
                tensor=torch.empty(record['size_bytes'],dtype=torch.uint8,device='cuda')
                _copy_tensor_bytes(source,record['name'],tensor)
                self._weights.append(tensor)
        self._pointers=(ctypes.c_void_p*len(self._weights))(*(x.data_ptr() for x in self._weights))
        logger.info('Loaded verified mixed BF16/FP32 GGUF projector into native Paiton: %d tensors',len(self._weights))

    def forward(self,pixel_values,grid_thw):
        grids=grid_thw.tolist() if isinstance(grid_thw,torch.Tensor) else grid_thw
        if len(grids)!=1 or len(grids[0])!=3 or grids[0][0]!=1:
            raise ValueError('This native vision profile supports one still image per request')
        _,h,w=grids[0];n=h*w
        if h<2 or w<2 or h%2 or w%2 or n>self.contract['max_patches']:
            raise ValueError('Image grid exceeds the native 4096-patch profile')
        if pixel_values.shape!=(n,1536) or pixel_values.dtype!=torch.float32 or not pixel_values.is_cuda:
            raise ValueError('Native vision requires CUDA FP32 [patches,1536] processor output')
        pixels=pixel_values.contiguous()
        output=torch.empty((n//4,5120),dtype=torch.float32,device=pixels.device)
        scratch=torch.empty(self.native.paiton_vision_workspace_bytes(n),dtype=torch.uint8,device=pixels.device)
        status=self.native.paiton_vision_run(self.handle,self._pointers,pixels.data_ptr(),output.data_ptr(),
            h,w,scratch.data_ptr(),scratch.numel(),torch.cuda.current_stream().cuda_stream,27)
        if status:
            raise RuntimeError(f'Native GGUF vision execution failed: {status}')
        if not self._logged:
            logger.info('Executing image encoder and projector through native Paiton HIP/rocBLAS artifact')
            self._logged=True
        return output


@MULTIMODAL_REGISTRY.register_processor(Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,dummy_inputs=Qwen3VLDummyInputsBuilder)
class PaitonQwen38GGUFForConditionalGeneration(PaitonQwen38ForConditionalGeneration):
    supports_multimodal_pruning=False

    def __init__(self,*,vllm_config,prefix='model'):
        nn.Module.__init__(self)
        self.config=vllm_config.model_config.hf_config
        self.model_config=vllm_config.model_config
        self.multimodal_config=self.model_config.multimodal_config
        if self.multimodal_config is None or vllm_config.quant_config is not None:
            raise ValueError('Native GGUF image model requires an unquantized vLLM multimodal shell')
        if self.multimodal_config.is_multimodal_pruning_enabled():
            raise ValueError('Native GGUF image profile does not support visual token pruning')
        self.use_data_parallel=False
        self.is_multimodal_pruning_enabled=False
        self.video_pruning_rate=0
        self._tokenizer=cached_tokenizer_from_config(self.model_config)
        self.use_deepstack=False
        self.deepstack_num_level=0
        self.visual_dim=5120
        self.multiscale_dim=0
        with self._mark_tower_model(vllm_config,{'image'}):
            self.visual=NativeGGUFVision(Path(self.model_config.model))
        with self._mark_language_model(vllm_config):
            self.language_model=_MultimodalGGUFBackbone(vllm_config=vllm_config,prefix='')
        self.make_empty_intermediate_tensors=self.language_model.make_empty_intermediate_tensors

    def _process_video_input(self, video_input):
        raise ValueError('Video is outside this native GGUF release profile')

    def load_weights(self,weights):
        del weights
        loaded=self.language_model.load_weights(())
        checkpoint=os.getenv('PAITON_GGUF_PROJECTOR')
        if not checkpoint:
            raise ValueError('PAITON_GGUF_PROJECTOR must identify the pinned projector')
        self.visual.load_weights(Path(checkpoint))
        return loaded|{'visual.'+r['name'] for r in self.visual.contract['bindings']}
