"""Native local vLLM Granite helper; audio models use their own runtimes."""
import json
import time
from pathlib import Path
from .compact_summary import CompactSummarizer, final_content
from .summary import SUMMARY_SCHEMA


class MeetingMetricsWorkerExtension:
    """Named local RPC avoids serializing executable callback functions."""
    def meeting_memory(self):
        import torch
        return dict(summary_peak_allocated=torch.cuda.max_memory_allocated(),
                    summary_peak_reserved=torch.cuda.max_memory_reserved())


class VLLMSummarizer(CompactSummarizer):
    backend = 'vllm'

    def __init__(self, directory):
        from transformers import AutoTokenizer
        from vllm import LLM
        directory = Path(directory).resolve(strict=True)
        if json.loads((directory/'config.json').read_text()).get('model_type') != 'granite':
            raise ValueError('This summary adapter requires the pinned Granite checkpoint.')
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
                                                       trust_remote_code=False)
        self.model = LLM(model=str(directory), dtype='bfloat16', max_model_len=16384,
            max_num_seqs=1, max_num_batched_tokens=4096, gpu_memory_utilization=0.45,
            enforce_eager=False, enable_prefix_caching=False, trust_remote_code=False,
            load_format='safetensors', disable_log_stats=True, seed=1201,
            attention_backend='TRITON_ATTN',
            worker_extension_cls='paiton_meeting.vllm_summary.MeetingMetricsWorkerExtension')
        self.loading_seconds = time.perf_counter()-started
        self.timings = []

    def generate(self, system, payload, schema=SUMMARY_SCHEMA):
        from vllm import SamplingParams
        messages = [dict(role='system', content=system+'\nReturn JSON matching this schema: '+json.dumps(schema)),
                    dict(role='user', content=payload)]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                    enable_thinking=True, low_effort=True, tokenize=False)
        count = len(self.tokenizer.encode(prompt))
        if count + 4096 > 16384:
            raise ValueError('Summary request exceeds the configured context budget.')
        started = time.perf_counter()
        response = self.model.generate([prompt], SamplingParams(max_tokens=4096,
            temperature=1.0, top_p=0.95, seed=1201+len(self.timings)), use_tqdm=False)[0].outputs[0]
        content = final_content(response.text, response.finish_reason == 'stop')
        self.timings.append(dict(seconds=time.perf_counter()-started,
                                 input_tokens=count, output_tokens=len(response.token_ids)))
        return content

    def memory_metrics(self):
        return self.model.collective_rpc('meeting_memory', timeout=30)[0]
