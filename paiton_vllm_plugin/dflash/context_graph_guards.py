"""Pinned external-runtime scope checks, with fail-closed metadata defaults."""
def configuration_allowed(config):
    parallel=config.parallel_config
    if any(getattr(parallel,key,None)!=1 for key in ('tensor_parallel_size','pipeline_parallel_size',
        'data_parallel_size','decode_context_parallel_size','prefill_context_parallel_size')):return False
    if getattr(config.model_config,'enable_sleep_mode',True):return False
    if getattr(config,'lora_config','unknown') is not None:return False
    if getattr(config,'kv_transfer_config','unknown') is not None:return False
    if getattr(config.cache_config,'kv_offloading_size','unknown') not in (None,0):return False
    offload=getattr(config,'offload_config',None)
    if offload is None:return False
    if getattr(getattr(offload,'uva',None),'cpu_offload_gb',None)!=0:return False
    if getattr(getattr(offload,'prefetch',None),'offload_group_size',None)!=0:return False
    spec=getattr(config,'speculative_config',None)
    if spec is None or getattr(spec,'num_speculative_tokens',None)!=7:return False
    return getattr(spec,'method',None)=='dflash'

def capture_allowed(slots,monitor):
    if slots is not None or not monitor.cudagraph_capturing_enabled:return False
    monitor.validate_cudagraph_capturing_enabled()
    return True
