"""vLLM integration for the compiled Paiton mixed-GGUF backbone.

Torch provides vLLM-owned storage. The backbone, packed embedding lookup,
and BF16 output projection execute native Paiton artifacts.
"""
import ctypes
import hashlib
import json
import os
import time
import weakref
from pathlib import Path

import torch
from torch import nn
from vllm.logger import init_logger

from paiton_vllm_plugin.gguf_source import GGUFTensorSource
from .paiton_qwen38 import PaitonQwen38ForCausalLM, _w4_lm_head_enabled
from paiton_vllm_plugin.runtime.core.utils.qwen38_loader import configure_qwen38_cache_contract

logger = init_logger("vllm.paiton.gguf")


class _PackedEmbedding(nn.Module):
    def __init__(self, model_path, record, activation_dtype):
        super().__init__()
        metadata = json.loads((model_path / 'gguf-runtime.json').read_text())
        library_path = model_path / 'libpaiton_gguf_runtime.so'
        if metadata.get('gpu_arch') != 'gfx1201' or metadata.get('abi_version') != 1:
            raise ValueError('GGUF helper architecture/ABI mismatch')
        if hashlib.sha256(library_path.read_bytes()).hexdigest() != metadata.get('sha256'):
            raise ValueError('GGUF helper SHA256 mismatch')
        self.native = ctypes.CDLL(str(library_path))
        if self.native.paiton_gguf_abi_version() != 1:
            raise ValueError('GGUF helper exported ABI mismatch')
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        self.activation_dtype = activation_dtype
        suffix = "_f32" if activation_dtype == torch.float32 else ""
        self.lookup = getattr(self.native, "paiton_gguf_embedding"+suffix)
        self.lookup.argtypes = [integer,pointer,pointer,pointer,integer,integer,integer,pointer]
        self.lookup.restype = integer
        self.lookup_i32 = getattr(self.native, "paiton_gguf_embedding_i32"+suffix)
        self.lookup_i32.argtypes = self.lookup.argtypes
        self.lookup_i32.restype = integer
        self.native.paiton_gguf_blas_create.argtypes = [ctypes.POINTER(pointer)]
        self.native.paiton_gguf_blas_destroy.argtypes = [pointer]
        self.head = self.native.paiton_gguf_bf16_head
        self.head.argtypes = [pointer,pointer,pointer,pointer,integer,integer,integer,pointer]
        self.linear = self.native.paiton_gguf_linear
        self.linear.argtypes = [integer,pointer,pointer,pointer,integer,integer,integer,integer,pointer]
        self.handle = pointer()
        if self.native.paiton_gguf_blas_create(ctypes.byref(self.handle)):
            raise RuntimeError('GGUF logits handle initialization failed')
        self._finalizer = weakref.finalize(self, self.native.paiton_gguf_blas_destroy, self.handle)
        self.type_id = record['type_id']
        self.width, self.vocab = record['dimensions']
        self.register_buffer('packed_weight', torch.empty(record['size_bytes'], dtype=torch.uint8), persistent=False)

    def forward(self, input_ids):
        if input_ids.dtype not in (torch.int32, torch.int64) or not input_ids.is_contiguous() or input_ids.ndim != 1:
            raise ValueError('GGUF embedding requires contiguous int32/int64 token IDs')
        output = torch.empty((input_ids.numel(), self.width), dtype=self.activation_dtype, device=input_ids.device)
        lookup = self.lookup_i32 if input_ids.dtype == torch.int32 else self.lookup
        status = lookup(self.type_id, self.packed_weight.data_ptr(), input_ids.data_ptr(),
            output.data_ptr(), input_ids.numel(), self.width, self.vocab,
            torch.cuda.current_stream(input_ids.device).cuda_stream)
        if status:
            raise RuntimeError(f'Native GGUF embedding failed: HIP status {status}')
        return output


def _copy_tensor_bytes(source, name, destination):
    raw_destination = destination.view(torch.uint8).reshape(-1)
    expected = source.tensors[name]['size_bytes']
    if raw_destination.numel() != expected:
        raise ValueError(f'GGUF destination size mismatch: {name}')
    for offset in range(0, expected, source.max_read_bytes):
        count = min(source.max_read_bytes, expected-offset)
        staging = torch.frombuffer(bytearray(source.read_bytes(name,offset,count)), dtype=torch.uint8)
        raw_destination[offset:offset+count].copy_(staging, non_blocking=False)


class PaitonQwen38GGUFForCausalLM(PaitonQwen38ForCausalLM):
    _gguf_multimodal = False

    def _configure_cache_contract(self, vllm_config):
        contract = vllm_config.model_config.hf_config.paiton_qwen38_contract
        configure_qwen38_cache_contract(vllm_config.cache_config, resolve_auto=True,
            conv_dtype=contract['gdn_conv_state_dtype'])

    def _use_native_decode_graph(self, num_tokens):
        enabled = (
            num_tokens == 1
            and self._native_decode_graph_enabled
            and self.contract.get('activation_dtype') == 'float32'
            and self.contract.get('attention_query_arithmetic') == 'bf16_pair'
        )
        if enabled and not self._native_graph_logged:
            logger.info('Using native HIP decode graph with live KV/GDN metadata and context bound %d',
                        self.contract['max_context_length'])
            self._native_graph_logged = True
        return enabled

    def _allocate_compiled_outputs(self, num_tokens, device):
        return {'hidden_states':torch.empty((num_tokens,self.config.hidden_size),
                                            dtype=self.activation_dtype,device=device)}

    def compute_logits(self, hidden_states):
        if hidden_states.dtype != self.activation_dtype or not hidden_states.is_contiguous() or not self.lm_head.weight.is_contiguous():
            raise ValueError('Native GGUF logits require contiguous tensors matching the precision contract')
        logits = torch.empty((hidden_states.shape[0],self.config.vocab_size),
                             dtype=torch.float32,device=hidden_states.device)
        if hidden_states.shape[0] == 0:
            return logits
        stream = torch.cuda.current_stream(hidden_states.device).cuda_stream
        if self.activation_dtype == torch.float32:
            status = self.embed_tokens.linear(30,self.lm_head.weight.data_ptr(),
                hidden_states.data_ptr(),logits.data_ptr(),hidden_states.shape[0],
                self.config.vocab_size,self.config.hidden_size,1,stream)
        else:
            status = self.embed_tokens.head(self.embed_tokens.handle,self.lm_head.weight.data_ptr(),
                hidden_states.data_ptr(),logits.data_ptr(),hidden_states.shape[0],
                self.config.vocab_size,self.config.hidden_size,stream)
        if status:
            raise RuntimeError(f'Native BF16 output head failed: {status}')
        trace_dir = os.getenv('PAITON_GGUF_LOGITS_TRACE_DIR')
        trace_count = getattr(self, '_logits_trace_count', 0)
        if trace_dir and self._native_forward_calls and trace_count < 8:
            # Explicit diagnostic copies are outside the compiled execution path.
            destination = Path(trace_dir)
            destination.mkdir(parents=True, exist_ok=True)
            logits.detach().cpu().numpy().tofile(destination/f'logits-{trace_count}.f32')
            (destination/f'logits-{trace_count}.json').write_text(json.dumps(
                dict(shape=list(logits.shape), dtype='float32', native_forward_calls=self._native_forward_calls)))
            self._logits_trace_count = trace_count + 1
        return logits

    def _prepare_weight_contract(self):
        contract = self.manifest.get('paiton_qwen38_contract')
        configured = getattr(self.vllm_config.model_config.hf_config, 'paiton_qwen38_contract', None)
        if contract != configured or not isinstance(contract, dict):
            raise ValueError('GGUF artifact/config contract mismatch')
        required = dict(version=5, weight_format='gguf-mixed',
            norm_binding='fp32_multiplicative', gdn_decay_binding='negative_exp_already_stored',
            gdn_value_head_layout='tiled', gdn_backend='native', full_attention_backend='native',
            mtp_speculative=False, multimodal=self._gguf_multimodal, tp_size=1, max_batch_size=1)
        if any(contract.get(key) != value for key,value in required.items()):
            raise ValueError('Unsupported native GGUF precision/state contract')
        activation = contract.get('activation_dtype')
        if activation not in ('bfloat16', 'float32') or contract.get('gdn_conv_state_dtype') != activation:
            raise ValueError('GGUF activation/convolution precision contract mismatch')
        self.activation_dtype = {'bfloat16':torch.bfloat16, 'float32':torch.float32}[activation]
        if _w4_lm_head_enabled():
            raise ValueError('GGUF output head must preserve its BF16 weights')
        if self.vllm_config.scheduler_config.max_num_seqs != 1:
            raise ValueError('Initial GGUF model supports one active sequence')
        if self.vllm_config.cache_config.enable_prefix_caching:
            raise ValueError('This GGUF profile requires prefix caching to be disabled')
        if contract.get('diagnostic'):
            raise ValueError('GGUF internal-tap artifacts are diagnostic only')
        if contract['num_hidden_layers'] != 64 and os.getenv('PAITON_GGUF_ALLOW_PARTIAL_MODEL') != '1':
            raise ValueError('Partial GGUF artifacts are diagnostic only')
        bindings = self.manifest.get('gguf_bindings', [])
        digest = hashlib.sha256(json.dumps(bindings,sort_keys=True).encode()).hexdigest()
        if digest != contract['bindings_sha256']:
            raise ValueError('GGUF artifact binding contract hash mismatch')
        if len({b['target'] for b in bindings}) != len(bindings):
            raise ValueError('Duplicate GGUF constant binding')
        self.contract = contract
        self.memory_estimate = None
        self._native_forward_calls = 0
        graph_setting = os.getenv('PAITON_GGUF_NATIVE_DECODE_GRAPHS', '1')
        if graph_setting not in ('0', '1'):
            raise ValueError('PAITON_GGUF_NATIVE_DECODE_GRAPHS must be 0 or 1')
        self._native_decode_graph_enabled = graph_setting == '1'
        self._native_graph_logged = False

    def _make_embedding(self):
        return _PackedEmbedding(self.model_path, self.contract['embedding'], self.activation_dtype)

    def load_weights(self, weights):
        del weights
        started = time.monotonic()
        selected = self.contract['checkpoint']['selected']
        name = selected['file']
        if Path(name).name != name:
            raise ValueError('GGUF checkpoint filename must be a basename')
        checkpoint = Path(os.getenv('PAITON_GGUF_CHECKPOINT', str(self.model_path/name)))
        loaded = set()
        logger.info("Verifying pinned GGUF checkpoint size and SHA256: %s", checkpoint)
        with GGUFTensorSource(checkpoint, sha256=selected['sha256'], size_bytes=selected['size_bytes']) as source:
            verified_at = time.monotonic()
            logger.info('GGUF checksum and inventory verification completed in %.3f seconds', verified_at-started)
            if source.inventory['metadata'].get('general.architecture') != 'qwen35':
                raise ValueError('GGUF architecture mismatch')
            _copy_tensor_bytes(source, 'token_embd.weight', self.embed_tokens.packed_weight)
            if source.tensors['output.weight']['type_id'] != 30:
                raise ValueError('GGUF output head must be BF16')
            _copy_tensor_bytes(source, 'output.weight', self.lm_head.weight)
            for binding in self.manifest['gguf_bindings']:
                record = source.tensors[binding['source']]
                dtype = {'float32':torch.float32, 'uint8':torch.uint8}[binding['dtype']]
                if dtype == torch.float32 and record['type_id'] != 0:
                    raise ValueError('GGUF small constants must remain FP32')
                tensor = torch.empty(binding['shape'],dtype=dtype,device='cuda')
                _copy_tensor_bytes(source,binding['source'],tensor)
                self.compiled_model.set_constant_with_tensor(binding['target'],tensor)
                loaded.add(binding['target'])
        logger.info('Loaded pinned GGUF into native Paiton: %d constants, %.3f seconds; artifact=%s',
                    len(loaded),time.monotonic()-started,self.model_so_path)
        return loaded | {'embed_tokens.packed_weight','lm_head.weight'}

    def forward(self, *args, **kwargs):
        from vllm.forward_context import get_forward_context
        scheduled = get_forward_context().attn_metadata is not None
        result = super().forward(*args, **kwargs)
        if scheduled:
            self._native_forward_calls += 1
            if self._native_forward_calls == 1:
                logger.info('Executing scheduled vLLM request through native Paiton GGUF artifact: %s', self.model_so_path)
        return result
