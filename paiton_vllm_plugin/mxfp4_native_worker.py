"""Public vLLM worker extension for the warmup-to-serving native boundary."""
import torch
from vllm.v1.worker.gpu_worker import Worker


class PaitonMxFp4Worker(Worker):
    def compile_or_warm_up_model(self):
        import os
        if os.environ.get('PAITON_EXPERIMENTAL_DFLASH_HEAD') == '1':
            model = self.get_model()
            model._bind_native_error_scope()
            draft = self.get_draft_model()
            if draft is None or not hasattr(draft, 'prepare_native_head'):
                raise RuntimeError('Native draft-head worker requires the registered DFlash2 model')
            draft.prepare_native_head(model.paiton_forward_error_scope)
            if os.environ.get('PAITON_EXPERIMENTAL_TARGET_HEAD') == '1':
                if (not self.use_v2_model_runner
                    or self.vllm_config.scheduler_config.async_scheduling):
                    raise ValueError('Native target-head policy currently requires synchronous V2 scheduling')
                if self.vllm_config.model_config.logits_processors:
                    raise ValueError('Native target-head policy does not support global logits processors')
                from .target_head_policy import GreedyBatchPolicy
                self.paiton_target_head_policy = GreedyBatchPolicy()
                model.prepare_native_target_head(draft.paiton_native_head)
        result = super().compile_or_warm_up_model()
        scope = getattr(self.get_model(), "paiton_forward_error_scope", None)
        if scope is None or scope.serving:
            raise RuntimeError("Native worker expected an unarmed model after synthetic warmup")
        status = scope.runtime.hip.hipDeviceSynchronize()
        if status:
            raise RuntimeError(f"Native warmup completion failed with HIP status {status}")
        stream = torch.cuda.current_stream(scope.error.device).cuda_stream
        # vLLM's synthetic forwards/captures own no user request history. Read
        # and retain their diagnostic before admitting the first real request.
        try:
            scope.runtime.check_error(scope.error, stream)
        except ValueError as error:
            scope.warmup_error = str(error)
        if os.environ.get('PAITON_EXPERIMENTAL_GDN_REPLAY') == '1':
            from .gdn_native_replay import runtime
            runtime().prepare_serving(scope)
        scope.runtime.clear_error(scope.error, stream)
        scope.runtime.clear_error(scope.profiling_error, stream)
        scope.serving = True
        scope.check()
        return result

    def execute_model(self, scheduler_output):
        policy = getattr(self, 'paiton_target_head_policy', None)
        if policy is not None:
            self.get_model().paiton_target_head_armed = False
            policy.update(scheduler_output)
        return super().execute_model(scheduler_output)

    def sample_tokens(self, grammar_output):
        policy = getattr(self, 'paiton_target_head_policy', None)
        if policy is None:
            return super().sample_tokens(grammar_output)
        model = self.get_model()
        model.paiton_target_head_armed = policy.allows_sampling(grammar_output)
        try:
            return super().sample_tokens(grammar_output)
        finally:
            model.paiton_target_head_armed = False
