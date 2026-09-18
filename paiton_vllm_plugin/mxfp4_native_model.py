"""Experimental model-level error checking for graphable native projections.

Tensor ownership stays in this external vLLM adapter. A sticky native error word
survives all prefill chunks and graph replays and is checked before logits are
returned. No error is cleared between forwards, so a prefill without logits
cannot hide an earlier invalid activation.
"""
import os
import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration


class NativeForwardErrorScope:
    def __init__(self, runtime, device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Bind native error storage before graph capture")
        self.runtime = runtime
        self.error = torch.empty(1, dtype=torch.int32, device=device)
        # vLLM skips attention during memory profiling. Its downstream tensors
        # are not model outputs and may contain uninitialized values. Keep that
        # separate from the sticky flag protecting actual inference/replays.
        self.profiling_error = torch.empty(1, dtype=torch.int32, device=device)
        runtime.clear_error(self.error, torch.cuda.current_stream(device).cuda_stream)
        runtime.clear_error(self.profiling_error, torch.cuda.current_stream(device).cuda_stream)
        self.profiling = False
        self.padding = None
        self.prefill_scratch = None
        self.profiling_forwards = 0
        self.capture_rows = set()
        self.error_checks = 0
        self.logits_checks = 0
        self.serving = True
        self.warmup_error = None

    def check(self):
        if not self.serving:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Native error check must run outside the captured forward, before logits return")
        self.runtime.check_error(self.error, torch.cuda.current_stream(self.error.device).cuda_stream)
        self.error_checks += 1


class PaitonQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix=""):
        if (vllm_config.parallel_config.tensor_parallel_size != 1
            or vllm_config.parallel_config.pipeline_parallel_size != 1
            or int(vllm_config.compilation_config.mode) != 0):
            raise ValueError("Native forward error scope currently requires TP1/PP1 and compilation mode NONE")
        self.paiton_gdn_stock_prefill = os.getenv("PAITON_EXPERIMENTAL_GDN_STOCK_PREFILL") == "1"
        if self.paiton_gdn_stock_prefill:
            from .gdn_native_stock_prefill import validate_config
            validate_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.paiton_forward_error_scope = None
        self.paiton_defer_warmup_error = os.getenv("PAITON_NATIVE_DEFER_WARMUP_ERROR") == "1"
        if self.paiton_defer_warmup_error and vllm_config.parallel_config.worker_cls != "paiton_vllm_plugin.mxfp4_native_worker.PaitonMxFp4Worker":
            raise ValueError("Deferred warmup checking requires the Paiton worker that arms checks before serving")

    def _bind_native_error_scope(self):
        if self.paiton_forward_error_scope is not None:
            return
        layers = [layer for layer in self.modules() if hasattr(layer, "paiton_mxfp4_shape")]
        if len(layers) != 304:
            raise ValueError("Expected all304 target projections before binding native error scope")
        scope = NativeForwardErrorScope(layers[0].paiton_mxfp4_runtime, layers[0].weight.device)
        if scope.runtime.prefill is not None:
            needed = max(scope.runtime.prefill_workspace_size(*layer.paiton_mxfp4_shape) for layer in layers)
            if not needed:
                raise ValueError('No supported native tiled prefill shapes')
            # One target forward owns the current stream. Reuse this workspace
            # sequentially across layers/chunks rather than allocating per GEMM.
            scope.prefill_scratch = torch.empty(needed,dtype=torch.uint8,device=scope.error.device)
        scope.serving = not self.paiton_defer_warmup_error
        for layer in layers:
            if layer.weight.device != scope.error.device:
                raise ValueError("Native forward error scope requires one device")
            layer.paiton_forward_error_scope = scope
            layer.paiton_quant_error = scope.error
        attention_layers = 0
        for name, layer in self.named_modules():
            if hasattr(layer, "paiton_mxfp4_shape"):
                layer.paiton_layer_name = name
            if hasattr(layer, 'paiton_replay'):
                layer.bind_native_scope(scope)
            impl = getattr(layer, 'impl', None)
            if getattr(impl, 'is_paiton_paged_attention', False):
                impl.bind_paiton_scope(scope)
                attention_layers += 1
        if os.environ.get('PAITON_EXPERIMENTAL_ATTENTION') == '1' and attention_layers != 16:
            raise ValueError(f'Expected16 target native attention layers, found{attention_layers}')
        if os.environ.get('PAITON_EXPERIMENTAL_SILU_FP8') == '1':
            from .silu_native_fp8 import bind
            bind(self,scope)
        if os.environ.get('PAITON_EXPERIMENTAL_NORM_FP8') == '1':
            from .norm_native_fp8 import bind
            bind(self,scope)
        if self.paiton_gdn_stock_prefill:
            from .gdn_native_stock_prefill import bind
            self.paiton_gdn_stock_prefill_layers = bind(self, scope)
        self.paiton_forward_error_scope = scope

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        self._bind_native_error_scope()
        scope = self.paiton_forward_error_scope
        context = get_forward_context()
        scope.profiling = context.attn_metadata is None
        # The V2 runner updates this persistent GPU mask before every graph
        # replay. Native quantization excludes only padding; active NaNs still
        # latch the serving error. No host token count is frozen into a graph.
        scope.padding = context.is_padding
        if scope.profiling:
            scope.profiling_forwards += 1
        try:
            return super().forward(input_ids=input_ids, positions=positions,
                intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds, **kwargs)
        finally:
            scope.profiling = False
            scope.padding = None

    def prepare_native_target_head(self, shared_head):
        from .target_head_native import NativeTargetHead
        self.paiton_native_target_head = NativeTargetHead(self, shared_head)
        self.paiton_target_head_armed = False
        self.paiton_target_head_policy_fallbacks = 0

    def compute_logits(self, hidden_states):
        scope = self.paiton_forward_error_scope
        if scope is None:
            raise RuntimeError("Native target forward did not bind its error scope")
        head = getattr(self, 'paiton_native_target_head', None)
        if head is not None and self.paiton_target_head_armed:
            logits = head(hidden_states)
            if logits is not None:
                # Check the complete native forward and sparse-head export at
                # one boundary, before the sampler can consume these logits.
                scope.check()
                scope.logits_checks += 1
                return logits
        elif head is not None:
            self.paiton_target_head_policy_fallbacks += 1
        scope.check()
        scope.logits_checks += 1
        return super().compute_logits(hidden_states)
