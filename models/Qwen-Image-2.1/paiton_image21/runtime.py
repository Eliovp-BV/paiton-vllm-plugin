"""Serialized, bounded image requests using a pinned packed checkpoint."""
import os
for _key,_value in (("MIOPEN_FIND_MODE","FAST"),("MIOPEN_DEBUG_CONV_GEMM","0")):
    os.environ.setdefault(_key,_value)   # VAE convolution solver selection; an explicit environment value wins (experiments)
os.environ.update(MIOPEN_ENABLE_LOGGING="0",MIOPEN_LOG_LEVEL="2",
                  HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TORCH_COMPILE_DISABLE="1",
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
import fcntl
import json
from pathlib import Path
import threading
import time
# the minors prompt guard is required: a missing or broken guard module stops the engine from importing / starting
from . import guard as prompt_guard
from .png import encode_png
import numpy as np

PNG_THREADS=max(1,min(8,os.cpu_count() or 1))

def to_uint8_pixels(image):
    """[1][C][H][W] denormalized bf16 pipeline output ("pt") -> uint8 [H][W][C] host array, computed on the GPU with the
    framework's PIL conversion arithmetic (bf16 -> fp32, x * 255 in fp32, round half to even, uint8): identical pixels,
    no fp32 image on the host."""
    import torch
    return (image[0].float()*255).round().to(torch.uint8).permute(1,2,0).contiguous().to("cpu")

def validate_request(prompt, width=2048, height=2048, steps=40, guidance=1.0,
                     seed=42, mode="text-to-image", image=None, sigmas=None):
    if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>512:
        raise ValueError("prompt must contain 1 to 512 characters")
    allowed,reason=prompt_guard.check_prompt(prompt)
    if not allowed:
        raise ValueError(prompt_guard.REFUSAL)
    if type(width) is not int or type(height) is not int or width!=height or width not in (1024,2048):
        raise ValueError("Qualified sizes are 1024x1024 and 2048x2048, batch one")
    required=len(sigmas) if sigmas else 40   # a checkpoint with a fixed sampling schedule (Turbo) defines its step count
    if type(steps) is not int or steps!=required or type(guidance) not in (int,float) or guidance!=1.0:
        raise ValueError(f"This model requires {required} steps and guidance 1.0")
    if type(seed) is not int or not 0<=seed<2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if mode not in ("text-to-image","rgba","edit"):
        raise ValueError("mode must be text-to-image, rgba or edit")
    if mode=="edit":
        if image is None or width!=1024:
            raise ValueError("Editing requires one input image and 1024x1024 output")
        if image.width*image.height>2048*2048:
            raise ValueError("Edit input is limited to 4,194,304 pixels")
    elif image is not None:
        raise ValueError("Input images are accepted only for edit requests")
    settings=dict(width=width,height=height,output_resolution=width,num_inference_steps=steps,
                  true_cfg_scale=guidance,use_kv_cache=True)
    if sigmas:
        settings["sigmas"]=list(sigmas)
    return settings

class ImageEngine:
    PRECISION_PROFILES={
        # exact: every step bit-exact against the pinned BF16 framework arithmetic; the wave64 exact attention kernel
        # (bit-exact on the reference suite and end to end) serves the dense and cached-decode attention
        "exact":{"PAITON_IMAGE21_ATTENTION_W64":"1"},
        # exact with the v1.0.1 wave32 attention kernel (fallback / comparison)
        "exact-w32":{},
        # release candidate: cached text prefix + first 7 denoising steps exact; steps 8-40 with int8 activations
        # (per 256-block scales, exact-weight int8 reconstruction) through wave64 int8 GEMMs fed by the fused LayerNorm
        # quantizer, int8-QK + fp8-P/V attention on those steps, wave64 exact attention on the exact steps
        "schedule-int8":{"PAITON_IMAGE21_LP":"int8-256","PAITON_IMAGE21_LP_W64":"1","PAITON_IMAGE21_LP_FROM_STEP":"8",
                         "PAITON_IMAGE21_ATTENTION_QK8":"1","PAITON_IMAGE21_ATTENTION_FULL8":"1","PAITON_IMAGE21_ATTENTION_W64":"1"},
        # the same with ten exact denoising steps and int8-QK-only attention (more headroom, measured 112.7 s)
        "schedule-int8-11":{"PAITON_IMAGE21_LP":"int8-256","PAITON_IMAGE21_LP_W64":"1","PAITON_IMAGE21_LP_FROM_STEP":"11",
                            "PAITON_IMAGE21_ATTENTION_QK8":"1","PAITON_IMAGE21_ATTENTION_W64":"1"},
        # bit-exact fp8 weights with fp8 per-row activations on the eleven-forward schedule (slower GEMMs)
        "schedule-fp8":{"PAITON_IMAGE21_LP":"fp8-row","PAITON_IMAGE21_LP_FROM_STEP":"11","PAITON_IMAGE21_ATTENTION_W64":"1"},
        # Turbo FP8 checkpoint (8 steps): steps 1-2 with bf16 activations on the 8-bit weights, steps 3-8 fp8 activations
        # with power-of-two per-token scales through the FP8 GEMM fed directly by the stored codes and row scales;
        # exact wave64 attention on every step (the arithmetic contract of the checkpoint's paiton_quant.json)
        "turbo-fp8":{"PAITON_IMAGE21_LP":"fp8-row","PAITON_IMAGE21_LP_FROM_STEP":"2","PAITON_IMAGE21_LP_POW2":"1",
                     "PAITON_IMAGE21_ATTENTION_W64":"1"},
        # the same with the wave64 fp8 GEMM (speed candidate)
        "turbo-fp8-w64":{"PAITON_IMAGE21_LP":"fp8-row","PAITON_IMAGE21_LP_W64":"1","PAITON_IMAGE21_LP_FROM_STEP":"2",
                         "PAITON_IMAGE21_LP_POW2":"1","PAITON_IMAGE21_ATTENTION_W64":"1"},
    }
    def __init__(self,model_dir,backend="native",artifact_dir=None,verify=True,native_fusions=False,precision_profile=None):
        self.prompt_guard=prompt_guard.self_test()   # fail closed before any GPU work
        if native_fusions and backend != "native":
            raise ValueError("native_fusions requires the native backend")
        precision_profile=precision_profile or os.environ.get("PAITON_IMAGE21_PROFILE") or "exact"
        if precision_profile not in self.PRECISION_PROFILES:
            raise ValueError(f"unknown precision profile {precision_profile!r}")
        if precision_profile != "exact" and not native_fusions:
            raise ValueError("precision profiles other than exact require native fusions")
        for key,value in self.PRECISION_PROFILES[precision_profile].items():
            os.environ.setdefault(key,value)   # explicit environment switches still win (experiments)
        self.precision_profile=precision_profile
        import torch
        from .weights import load_pipeline,enable_native_unpack,sha256
        if backend not in ("native","reference"):
            raise ValueError("backend must be native or reference")
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("This package requires the qualified ROCm environment")
        props=torch.cuda.get_device_properties(0)
        if not props.gcnArchName.startswith("gfx1201"):
            raise RuntimeError("Only Radeon AI PRO R9700 / gfx1201 is qualified")
        self._gpu_lock=open(os.environ.get("PAITON_GPU_LOCK", "/tmp/paiton-studio-gpu-1000.lock"),"a")
        fcntl.flock(self._gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        free,total=torch.cuda.mem_get_info()
        if free<30*2**30:
            raise RuntimeError("The R9700 needs at least 30 GiB free before loading")
        # Leave measured headroom for driver/library allocations outside the allocator.
        torch.cuda.set_per_process_memory_fraction(26*2**30/total)
        torch.set_num_threads(4)
        self.busy=threading.Lock()
        self.model_dir=Path(model_dir).resolve()
        manifest=self.model_dir/"result.json"
        self.checkpoint_sha256=sha256(manifest)
        checkpoint_metadata=json.loads(manifest.read_text())
        self.model_id=checkpoint_metadata.get("model_id","paiton-image-2.1")
        self.model_variant=checkpoint_metadata.get("model_variant","original")
        # fixed sampling schedule of the checkpoint (Turbo: model_index.json sample_sigmas); None = 40-step 2.1 schedule
        index=self.model_dir/"model_index.json"
        self.sample_sigmas=json.loads(index.read_text()).get("sample_sigmas") if index.is_file() else None
        self.steps=len(self.sample_sigmas) if self.sample_sigmas else 40
        self.backend=backend
        self.allocator_budget_gib=26
        self.allocator_config="expandable_segments:True"
        self.native_regions=None
        self.native_fusion_status="disabled"
        if backend=="native":
            artifacts=Path(artifact_dir) if artifact_dir else Path(__file__).resolve().parent.parent/"artifacts"
            contract=json.loads((artifacts/"manifest.json").read_text())
            library=artifacts/contract["file"]
            if contract["architecture"]!="gfx1201" or sha256(library)!=contract["sha256"]:
                raise RuntimeError("Native artifact identity mismatch")
            enable_native_unpack(library)
        begin=time.perf_counter()
        self.pipeline=load_pipeline(self.model_dir,self.model_dir,verify_hashes=verify)
        self._slim_text_encoder()
        if native_fusions:
            from .native_regions import install
            companions={name:artifacts/name for name in ("attention","attention-w64","attention-qk8","attention-full8","normfuse","gemm-fp8","gemm-int8","gemm-w64","quantize","unpack-fp8") if (artifacts/name/"manifest.json").is_file()}
            if os.environ.get("PAITON_IMAGE21_ATTENTION_W64")=="1" and "attention-w64" in companions:   # wave64 exact attention (bit-exact; the exact profile)
                companions["attention"]=companions["attention-w64"]
            fp8_directories=None
            low_precision=os.environ.get("PAITON_IMAGE21_LP") or ("fp8-row" if os.environ.get("PAITON_IMAGE21_FP8")=="1" else None)
            if low_precision:   # low-precision GEMM path of the precision schedule (selected by the profile switches)
                fp8_directories=dict(gemm_directory=companions.get("gemm-fp8"),quantize_directory=companions["quantize"],
                                     unpack_directory=companions["unpack-fp8"],gemm_int8_directory=companions.get("gemm-int8"),
                                     gemm_w64_directory=companions.get("gemm-w64"),format=low_precision,
                                     wave64=os.environ.get("PAITON_IMAGE21_LP_W64")=="1",
                                     from_step=int(os.environ.get("PAITON_IMAGE21_LP_FROM_STEP","0")),
                                     exact_prefix=os.environ.get("PAITON_IMAGE21_LP_QUANTIZE_PREFIX")!="1",
                                     pow2=os.environ.get("PAITON_IMAGE21_LP_POW2")=="1")
            qk8_directory=None
            if low_precision and os.environ.get("PAITON_IMAGE21_ATTENTION_QK8")=="1":
                qk8_directory=companions.get("attention-full8" if os.environ.get("PAITON_IMAGE21_ATTENTION_FULL8")=="1" else "attention-qk8")
            self.native_regions=install(self.pipeline.transformer,artifacts/"bf16-regions",
                                        companions.get("attention"),companions.get("normfuse"),fp8_directories,qk8_directory)
            self.native_fusion_status="active" if self.native_regions is not None else "unsupported configuration; original regions retained"
            if self.native_regions is not None:
                active=["bf16-regions"]+[name for name,library in (("attention",self.native_regions.attention_library),
                                                                ("normfuse",self.native_regions.normfuse_library)) if library is not None]
                if self.native_regions.fp8 is not None:
                    active.append(f"low-precision {self.native_regions.fp8.format}{' wave64' if self.native_regions.fp8.wave64 else ''}")
                if getattr(self.native_regions,"qk8_library",None) is not None:
                    active.append(("int8-QK + fp8-PV" if os.environ.get("PAITON_IMAGE21_ATTENTION_FULL8")=="1" else "int8-QK")+" attention on low-precision forwards")
                self.native_fusion_status="active ("+", ".join(active)+")"
        self.vae_regions=None
        if native_fusions and os.environ.get("PAITON_IMAGE21_VAE_FAST","1")=="1":
            from .vae_fast import install as install_vae
            norm_directory=artifacts/"vae-norm" if (artifacts/"vae-norm"/"manifest.json").is_file() else None
            self.vae_regions=install_vae(self.pipeline.vae,norm_directory)
        # text encoder placement: "1" = host-resident (language model uploaded per request), "lite" = language model
        # resident on the GPU with the token embedding and the vision tower on the host, else fully resident
        te_mode=os.environ.get("PAITON_IMAGE21_TE_HOST","0")
        self.text_encoder_host="lite" if te_mode=="lite" else te_mode=="1"
        if self.text_encoder_host:
            self._host_text_encoder(language_resident=te_mode=="lite")
        self._guarded_encode=prompt_guard.install(self.pipeline)   # outermost: every prompt encoding passes the guard
        torch.cuda.synchronize()
        self.load_seconds=time.perf_counter()-begin
        self.stream=torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        self.request_count=0
        self._prefix_caches=[]
        def remember_prefix(module,args,kwargs):
            cache=kwargs.get("kv_cache")
            if cache is not None and all(cache is not old for old in self._prefix_caches):
                self._prefix_caches.append(cache)
            fp8=getattr(self.native_regions,"fp8",None) if self.native_regions is not None else None
            if fp8 is not None:
                fp8.begin_transformer_forward()   # precision schedule counts transformer forwards, not block forwards
        self.pipeline.transformer.register_forward_pre_hook(remember_prefix,with_kwargs=True)
        # optional start-up warm-up (PAITON_IMAGE21_WARMUP="1024" or "1024,2048"): one discarded request per size so the
        # first served request runs warm (kernel loading, solver selection, allocator growth); costs start-up time
        self.warmup_seconds={}
        for size in [int(value) for value in os.environ.get("PAITON_IMAGE21_WARMUP","").split(",") if value.strip()]:
            begin_warmup=time.perf_counter()
            self.generate("A plain grey ceramic cup on a wooden table, soft daylight",width=size,height=size,seed=0)
            self.warmup_seconds[size]=time.perf_counter()-begin_warmup
        self.request_count=0

    def _slim_text_encoder(self):
        """The pipeline reads the text encoder's hidden states only: its 151,936 x 4,096 output projection (1.2 GB) is
        replaced by an empty-logits stub (identical hidden states; set PAITON_IMAGE21_TE_LOGITS=1 to keep it)."""
        import torch
        te=self.pipeline.text_encoder
        self.text_encoder_logits=os.environ.get("PAITON_IMAGE21_TE_LOGITS")=="1" or not isinstance(getattr(te,"lm_head",None),torch.nn.Linear)
        if not self.text_encoder_logits:
            class NoLogits(torch.nn.Module):   # stand-in output projection: empty logits, hidden states unchanged
                def forward(self,hidden_states):
                    return hidden_states[...,:0]
            te.lm_head=NoLogits()
            torch.cuda.empty_cache()

    def _host_text_encoder(self,language_resident=False):
        """Keep the text encoder's tensors in pinned host memory and upload the language-model tensors for the prompt
        encoding of each request (frees their VRAM for the denoiser and the VAE; costs one host-to-device copy per
        request). The token embedding is looked up on the host (only the prompt's rows are copied) and the vision
        tower is uploaded only when a request runs it (edits)."""
        import torch
        te=self.pipeline.text_encoder
        language=getattr(te.model,"language_model",te.model)
        embed=language.embed_tokens
        visual=getattr(te.model,"visual",None)
        visual_modules=set(visual.modules()) if visual is not None else set()
        self._te_slots=[]
        self._te_visual_slots=[]
        for module in te.modules():
            for kind,store in (("p",module._parameters),("b",module._buffers)):
                for key,value in store.items():
                    if value is not None and value.is_cuda:
                        if language_resident and module is not embed and module not in visual_modules:
                            continue   # "lite": the language model stays on the GPU
                        host=torch.empty(value.shape,dtype=value.dtype,pin_memory=True)
                        host.copy_(value)
                        store[key]=torch.nn.Parameter(host,requires_grad=False) if kind=="p" else host
                        if module is embed:
                            continue   # never uploaded: host lookup below
                        (self._te_visual_slots if module in visual_modules else self._te_slots).append((store,key,kind,host))
        torch.cuda.empty_cache()
        def embed_forward(input_ids,_embed=embed):
            rows=torch.nn.functional.embedding(input_ids.reshape(-1).cpu(),_embed.weight,_embed.padding_idx)
            return rows.view(*input_ids.shape,rows.shape[-1]).to(input_ids.device)
        if type(embed) is torch.nn.Embedding and embed.max_norm is None:
            embed.forward=embed_forward
        else:   # unexpected embedding module: upload it with the language model
            self._te_slots.extend((embed._parameters,key,"p",value.data) for key,value in embed._parameters.items() if value is not None)
        if visual is not None:
            visual.register_forward_pre_hook(lambda module,args:self._upload_slots(self._te_visual_slots))
        pipeline=self.pipeline
        # the pipeline derives its execution device from the components: keep it on the GPU with a host-resident encoder
        cls=type(pipeline)
        pipeline.__class__=type(cls.__name__,(cls,),{"_execution_device":property(lambda _:torch.device("cuda")),
                                                      "device":property(lambda _:torch.device("cuda"))})
        original=pipeline.encode_prompt
        def encode_prompt(*args,**kwargs):
            self._text_encoder_to("cuda")
            try:
                return original(*args,**kwargs)
            finally:
                self._text_encoder_to(None)
        pipeline.encode_prompt=encode_prompt

    def _upload_slots(self,slots):
        import torch
        for store,key,kind,host in slots:
            value=host.to("cuda",non_blocking=True)
            store[key]=torch.nn.Parameter(value,requires_grad=False) if kind=="p" else value

    def _text_encoder_to(self,device):
        import torch
        if device:
            self._upload_slots(self._te_slots)
            return
        for store,key,kind,host in self._te_slots+self._te_visual_slots:
            store[key]=torch.nn.Parameter(host,requires_grad=False) if kind=="p" else host

    def _release_prefix(self):
        for cache in self._prefix_caches:
            for layer in cache.layer_caches:
                layer.k=layer.v=None
        self._prefix_caches.clear()

    def generate(self,prompt,width=2048,height=2048,steps=None,guidance=1.0,seed=42,
                 mode="text-to-image",image=None):
        import torch
        settings=validate_request(prompt,width,height,self.steps if steps is None else steps,guidance,seed,mode,image,
                                  self.sample_sigmas)
        if getattr(self.pipeline,"encode_prompt",None) is not self._guarded_encode:
            raise RuntimeError("prompt guard is not installed on the pipeline; refusing the request")
        if not self.busy.acquire(blocking=False):
            raise RuntimeError("The single-request GPU worker is busy")
        started=time.perf_counter()
        if self.native_regions is not None and getattr(self.native_regions,"fp8",None) is not None:
            self.native_regions.fp8.forwards=0   # precision schedule counts transformer forwards per request
        stop=threading.Event()
        samples=[]
        def sample():
            while not stop.is_set():
                free,total=torch.cuda.mem_get_info()
                samples.append(total-free)
                stop.wait(.02)
        monitor=threading.Thread(target=sample,daemon=True)
        actual_prompt=prompt
        if mode=="rgba":
            actual_prompt="This is an RGBA image with transparency. "+prompt+" The image has alpha channel and the background is transparent."
        try:
            self._release_prefix()
            torch.cuda.reset_peak_memory_stats()
            monitor.start()
            def callback(pipe,step,timestep,values):
                if step==self.steps-1:
                    self._release_prefix()
                return values
            kwargs={"image":image.convert("RGBA")} if image is not None else {}
            with torch.cuda.stream(self.stream),torch.inference_mode():
                image=self.pipeline(prompt=actual_prompt,**settings,**kwargs,output_type="pt",
                    generator=torch.Generator("cuda").manual_seed(seed),callback_on_step_end=callback).images
                pixels=to_uint8_pixels(image)
            self.stream.synchronize()
            pixels=pixels.numpy()
            from PIL import Image
            result=Image.fromarray(pixels)
            seconds=time.perf_counter()-started
            assert result.size==(width,height) and result.mode=="RGBA"
            png_bytes=encode_png(pixels,level=6,threads=PNG_THREADS)   # same pixels, parallel deflate
            row=dict(mode=mode,seed=seed,settings=settings,backend=self.backend,
                     checkpoint_sha256=self.checkpoint_sha256,request_index=self.request_count,
                     complete_request_to_pil_seconds=seconds,complete_request_to_png_seconds=time.perf_counter()-started,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                     sampled_peak_device_bytes=max(samples,default=0),sampling_interval_seconds=.02,
                     stream="dedicated non-default HIP stream",alpha_extrema=list(result.getextrema()[3]))
            row["native_fusions"]=self.native_fusion_status
            row["precision_profile"]=self.precision_profile
            row["model"]=self.model_id
            row["model_variant"]=self.model_variant
            row["prompt_guard"]=self.prompt_guard
            row["steps"]=self.steps
            row["text_encoder_host"]=self.text_encoder_host
            row["text_encoder_logits"]=self.text_encoder_logits
            if self.vae_regions is not None:
                row["vae_regions"]=dict(self.vae_regions)
            row["allocator_budget_gib"]=self.allocator_budget_gib
            row["allocator_config"]=self.allocator_config
            if self.native_regions is not None:
                row["native_fusions_sha256"]=self.native_regions.manifest["sha256"]
                for name,manifest in (("attention",self.native_regions.attention_manifest),("normfuse",self.native_regions.normfuse_manifest)):
                    if manifest is not None:
                        row[f"native_{name}_sha256"]=manifest["sha256"]
                row["native_counts"]=dict(self.native_regions.counts)
                fp8=self.native_regions.fp8
                if fp8 is not None:
                    inexact=fp8.check_exact()
                    row["fp8_gemm"]=dict(inexact_weight_values=inexact,**fp8.receipt())
                    if inexact and not fp8.int8:
                        raise RuntimeError(f"fp8 weight reconstruction was not exact: {inexact} values")
            self.request_count+=1
            return result,png_bytes,row
        finally:
            stop.set()
            if monitor.is_alive():
                monitor.join(timeout=1)
            self._release_prefix()
            self.busy.release()
