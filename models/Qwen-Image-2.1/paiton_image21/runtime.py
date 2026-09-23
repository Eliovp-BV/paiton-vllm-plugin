"""Serialized, bounded image requests using a pinned packed checkpoint."""
import os
os.environ.update(MIOPEN_FIND_MODE="FAST",MIOPEN_DEBUG_CONV_GEMM="0",
                  MIOPEN_ENABLE_LOGGING="0",MIOPEN_LOG_LEVEL="2",
                  HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TORCH_COMPILE_DISABLE="1",
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
import fcntl
import io
import json
from pathlib import Path
import threading
import time

def validate_request(prompt, width=2048, height=2048, steps=40, guidance=1.0,
                     seed=42, mode="text-to-image", image=None):
    if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>512:
        raise ValueError("prompt must contain 1 to 512 characters")
    if type(width) is not int or type(height) is not int or width!=height or width not in (1024,2048):
        raise ValueError("Qualified sizes are 1024x1024 and 2048x2048, batch one")
    if type(steps) is not int or steps!=40 or type(guidance) not in (int,float) or guidance!=1.0:
        raise ValueError("This profile requires 40 steps and guidance 1.0")
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
    return dict(width=width,height=height,output_resolution=width,num_inference_steps=steps,
                true_cfg_scale=guidance,use_kv_cache=True)

class ImageEngine:
    def __init__(self,model_dir,backend="native",artifact_dir=None,verify=True,native_fusions=False):
        if native_fusions and backend != "native":
            raise ValueError("native_fusions requires the native backend")
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
        if native_fusions:
            from .native_regions import install
            companions={name:artifacts/name for name in ("attention","normfuse") if (artifacts/name/"manifest.json").is_file()}
            self.native_regions=install(self.pipeline.transformer,artifacts/"bf16-regions",
                                        companions.get("attention"),companions.get("normfuse"))
            self.native_fusion_status="active" if self.native_regions is not None else "unsupported configuration; original regions retained"
            if self.native_regions is not None:
                active=["bf16-regions"]+[name for name,library in (("attention",self.native_regions.attention_library),
                                                                ("normfuse",self.native_regions.normfuse_library)) if library is not None]
                self.native_fusion_status="active ("+", ".join(active)+")"
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
        self.pipeline.transformer.register_forward_pre_hook(remember_prefix,with_kwargs=True)

    def _release_prefix(self):
        for cache in self._prefix_caches:
            for layer in cache.layer_caches:
                layer.k=layer.v=None
        self._prefix_caches.clear()

    def generate(self,prompt,width=2048,height=2048,steps=40,guidance=1.0,seed=42,
                 mode="text-to-image",image=None):
        import torch
        settings=validate_request(prompt,width,height,steps,guidance,seed,mode,image)
        if not self.busy.acquire(blocking=False):
            raise RuntimeError("The single-request GPU worker is busy")
        started=time.perf_counter()
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
                if step==39:
                    self._release_prefix()
                return values
            kwargs={"image":image.convert("RGBA")} if image is not None else {}
            with torch.cuda.stream(self.stream),torch.inference_mode():
                result=self.pipeline(prompt=actual_prompt,**settings,**kwargs,
                    generator=torch.Generator("cuda").manual_seed(seed),callback_on_step_end=callback).images[0]
            self.stream.synchronize()
            seconds=time.perf_counter()-started
            assert result.size==(width,height) and result.mode=="RGBA"
            buffer=io.BytesIO()
            result.save(buffer,format="PNG")
            row=dict(mode=mode,seed=seed,settings=settings,backend=self.backend,
                     checkpoint_sha256=self.checkpoint_sha256,request_index=self.request_count,
                     complete_request_to_pil_seconds=seconds,complete_request_to_png_seconds=time.perf_counter()-started,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                     sampled_peak_device_bytes=max(samples,default=0),sampling_interval_seconds=.02,
                     stream="dedicated non-default HIP stream",alpha_extrema=list(result.getextrema()[3]))
            row["native_fusions"]=self.native_fusion_status
            row["allocator_budget_gib"]=self.allocator_budget_gib
            row["allocator_config"]=self.allocator_config
            if self.native_regions is not None:
                row["native_fusions_sha256"]=self.native_regions.manifest["sha256"]
                for name,manifest in (("attention",self.native_regions.attention_manifest),("normfuse",self.native_regions.normfuse_manifest)):
                    if manifest is not None:
                        row[f"native_{name}_sha256"]=manifest["sha256"]
                row["native_counts"]=dict(self.native_regions.counts)
            self.request_count+=1
            return result,buffer.getvalue(),row
        finally:
            stop.set()
            if monitor.is_alive():
                monitor.join(timeout=1)
            self._release_prefix()
            self.busy.release()
