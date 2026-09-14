"""Scoped workaround for tokenizers DecodeStream's incomplete UTF-8 prefix bug."""
import os


def install_gguf_detokenizer_compat():
    if os.getenv('PAITON_GGUF_SAFE_DETOKENIZER') != '1':
        return
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer, SlowIncrementalDetokenizer
    if getattr(IncrementalDetokenizer, '_paiton_gguf_utf8_compat', False):
        return
    original = IncrementalDetokenizer.from_new_request

    def from_new_request(cls, tokenizer, request):
        ids = request.prompt_token_ids
        # DecodeStream can emit the entire prompt after an unfinished UTF-8
        # byte token. Existing vLLM slow detokenization tracks that suffix.
        if tokenizer is not None and ids and tokenizer.decode(
            ids, skip_special_tokens=request.sampling_params.skip_special_tokens
        ).endswith('\ufffd'):
            return SlowIncrementalDetokenizer(tokenizer, request)
        return original(tokenizer, request)

    IncrementalDetokenizer.from_new_request = classmethod(from_new_request)
    IncrementalDetokenizer._paiton_gguf_utf8_compat = True
