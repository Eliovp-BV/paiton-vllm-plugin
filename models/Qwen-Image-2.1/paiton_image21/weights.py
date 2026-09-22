"""Isolated external reference consumer for Quark MXFP4 weights.

This consumer reconstructs a BF16 matrix for each operation. It does not claim
native MXFP4 GEMM, compiler integration, or an inference speedup.
"""
import hashlib
import ctypes
import json
import math
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open

E2M1 = [0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]
_NATIVE_UNPACK = None
_NATIVE_LIBRARY = None

def enable_native_unpack(path):
    """Raw-pointer C ABI adapter; native library has no framework dependency."""
    global _NATIVE_UNPACK,_NATIVE_LIBRARY
    _NATIVE_LIBRARY=ctypes.CDLL(str(Path(path).resolve()))
    _NATIVE_UNPACK=_NATIVE_LIBRARY.paiton_mxfp4_unpack
    _NATIVE_UNPACK.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
    _NATIVE_UNPACK.restype=ctypes.c_int

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda:f.read(8*1024*1024),b""):
            h.update(data)
    return h.hexdigest()

def decode_matrix(weight, scale, columns, lookup):
    rows = weight.shape[0]
    padded = weight.shape[1]*2
    if _NATIVE_UNPACK is not None:
        assert weight.is_cuda and weight.is_contiguous() and scale.is_contiguous()
        out=torch.empty((rows,padded),device=weight.device,dtype=torch.bfloat16)
        rc=_NATIVE_UNPACK(weight.data_ptr(),scale.data_ptr(),out.data_ptr(),out.numel(),
                          torch.cuda.current_stream(weight.device).cuda_stream)
        if rc:
            raise RuntimeError(f"Native unpack failed with HIP status {rc}")
        # Match the reference BF16 conversion's compact layout after padding
        # removal. A strided convolution weight can select different arithmetic.
        return out[:,:columns].contiguous()
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).reshape(rows,padded)
    decoded = lookup[codes.long()].reshape(rows,padded//32,32)
    decoded = decoded * torch.exp2(scale.float()-127).unsqueeze(-1)
    return decoded.reshape(rows,padded)[:,:columns].to(torch.bfloat16)

class PackedWeightMixin:
    @property
    def weight(self):
        cached=self.__dict__.get("_materialized_weight")
        if cached is not None:
            return cached
        shape = self._original_weight_shape
        return decode_matrix(self._packed_weight,self._packed_scale,
                             math.prod(shape[1:]),self._mxfp4_lookup).reshape(shape)

class PackedEmbeddingMixin(PackedWeightMixin):
    def forward(self, indices):
        assert self.max_norm is None, "MXFP4 embedding max_norm is unsupported"
        flat = indices.reshape(-1)
        weight = self._packed_weight.index_select(0,flat)
        scale = self._packed_scale.index_select(0,flat)
        value = decode_matrix(weight,scale,self.embedding_dim,self._mxfp4_lookup)
        return value.reshape(*indices.shape,self.embedding_dim)

_TYPES = {}

def install_packed(module, row, weight, scale, device):
    shape = tuple(row["shape"])
    padded = tuple(row["padded_shape"])
    assert tuple(module.weight.shape) == shape
    assert weight.dtype == scale.dtype == torch.uint8
    assert tuple(weight.shape) == (shape[0],padded[1]//2)
    assert tuple(scale.shape) == (shape[0],padded[1]//32)
    assert row["kind"] in ("linear","embedding","conv2d","conv3d")
    old_type = type(module)
    mixin = PackedEmbeddingMixin if row["kind"] == "embedding" else PackedWeightMixin
    pair = (mixin,old_type)
    if pair not in _TYPES:
        _TYPES[pair] = type("MXFP4Reference"+old_type.__name__,(mixin,old_type),{})
    del module._parameters["weight"]
    module.__class__ = _TYPES[pair]
    module._original_weight_shape = shape
    module.register_buffer("_packed_weight",weight.to(device))
    module.register_buffer("_packed_scale",scale.to(device))
    module.register_buffer("_mxfp4_lookup",torch.tensor(E2M1,dtype=torch.float32,device=device),persistent=False)

def load_component(snapshot, bundle, name, device="cuda", quantized=True, verify_hashes=False, event=None):
    from diffusers import QwenImage21Transformer2DModel, AutoencoderKLQwenImage21
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    snapshot, bundle = Path(snapshot), Path(bundle)
    manifest = json.loads((bundle/"result.json").read_text())
    assert manifest["status"] == "completed_conversion_only"
    assert manifest["format"] == "paiton-research-mxfp4-weight-only-v2"
    with init_empty_weights(include_buffers=False):
        if name == "text_encoder":
            config = Qwen3VLConfig.from_pretrained(str(snapshot/name),local_files_only=True)
            config.dtype = torch.bfloat16
            model = Qwen3VLForConditionalGeneration(config)
        else:
            cls = QwenImage21Transformer2DModel if name=="transformer" else AutoencoderKLQwenImage21
            model = cls.from_config(cls.load_config(str(snapshot/name)))
    expected = {k:tuple(p.shape) for k,p in model.named_parameters()}
    loaded = set()
    meta = manifest["components"][name]
    if quantized:
        assert sha256(bundle/name/"config.json") == meta["config_sha256"]
        assert sha256(snapshot/name/"config.json") == meta["config_sha256"]
        targets = {x["name"]+".weight":x for x in meta["converted_layers"]}
        paths = [bundle/name/x["file"] for x in meta["files"]]
        if verify_hashes:
            for path,row in zip(paths,meta["files"]):
                assert path.stat().st_size == row["bytes"] and sha256(path)==row["sha256"],path
    else:
        targets = {}
        paths = sorted((snapshot/name).glob("*.safetensors"))
    for path in paths:
        with safe_open(path,framework="pt",device="cpu") as f:
            for key in sorted(f.keys()):
                if key.endswith(".weight_scale"):
                    continue
                assert key in expected and key not in loaded,key
                value = f.get_tensor(key)
                if key in targets:
                    row = targets[key]
                    install_packed(model.get_submodule(row["name"]),row,value,
                                   f.get_tensor(row["name"]+".weight_scale"),device)
                else:
                    assert tuple(value.shape) == expected[key],key
                    parent,leaf = key.rsplit(".",1)
                    setattr(model.get_submodule(parent),leaf,torch.nn.Parameter(
                        value.to(device=device,dtype=torch.bfloat16),requires_grad=False))
                loaded.add(key)
                del value
        if event:
            event("loaded_shard",component=name,file=path.name,tensors=len(loaded))
    assert loaded == set(expected),sorted(set(expected)-loaded)
    for module in model.modules():
        for key,value in module._buffers.items():
            if value is not None:
                assert not value.is_meta,key
                module._buffers[key] = value.to(device)
    assert not any(p.is_meta for p in model.parameters())
    model.eval()
    payload = sum(t.numel()*t.element_size() for t in model.state_dict().values())
    assert payload == (meta["tensor_payload_bytes"] if quantized else meta["source_bf16_bytes"]),(name,payload)
    assert model.dtype == torch.bfloat16,(name,model.dtype)
    if event:
        event("component_loaded",component=name,quantized=quantized,tensor_payload_bytes=payload)
    return model

def remove_single_frame_temporal_caches(vae):
    def decode_hook(module,args,kwargs):
        assert args[0].ndim == 5 and args[0].shape[2] == 1
        assert kwargs["first_chunk"] is True
        return args,dict(kwargs,feat_cache=None)
    def encode_hook(module,args,kwargs):
        assert args[0].ndim == 5 and args[0].shape[2] == 1
        return args,dict(kwargs,feat_cache=None)
    vae.decoder.register_forward_pre_hook(decode_hook,with_kwargs=True)
    vae.encoder.register_forward_pre_hook(encode_hook,with_kwargs=True)

def materialize_denoiser_weights(transformer):
    """Reference baseline: expand once for all steps; release before VAE."""
    count=0
    for module in transformer.modules():
        if isinstance(module,PackedWeightMixin):
            if module.__dict__.get("_materialized_weight") is None:
                module._materialized_weight=module.weight
                count+=module._materialized_weight.numel()*2
    return count

def release_denoiser_weights(transformer):
    for module in transformer.modules():
        module.__dict__.pop("_materialized_weight",None)

def load_pipeline(snapshot,bundle,quantized_components=("text_encoder","transformer","vae"),
                  device="cuda",verify_hashes=False,event=None):
    from diffusers import QwenImage21Pipeline,FlowMatchEulerDiscreteScheduler
    from transformers import Qwen3VLProcessor
    snapshot = Path(snapshot)
    components = {name:load_component(snapshot,bundle,name,device,name in quantized_components,verify_hashes,event)
                  for name in ("text_encoder","transformer","vae")}
    pipe = QwenImage21Pipeline(
        **components,
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(snapshot/"scheduler"),local_files_only=True),
        processor=Qwen3VLProcessor.from_pretrained(str(snapshot/"processor"),local_files_only=True))
    remove_single_frame_temporal_caches(pipe.vae)
    assert not pipe.vae.use_tiling
    return pipe
