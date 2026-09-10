"""Small local Granite text helper for the dedicated speech pipeline."""
import json
import time
from pathlib import Path

from .summary import SUMMARY_SCHEMA, VERIFY_SCHEMA, summarize, verify_summary


def final_content(text, finished):
    if not finished:
        raise ValueError('Summary generation did not finish; no partial summary was accepted.')
    if '</think>' not in text:
        raise ValueError('Summary reasoning did not finish.')
    content = text.split('</think>', 1)[1].strip()
    if not content:
        raise ValueError('Summarizer returned no structured content.')
    return content


class CompactSummarizer:
    """BF16 SDPA, bounded context, seeded sampling, no remote inference or code."""
    def __init__(self, directory):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        directory = Path(directory)
        config = json.loads((directory / 'config.json').read_text())
        if config.get('model_type') != 'granite':
            raise ValueError('This summary adapter requires the pinned Granite checkpoint.')
        self.torch = torch
        torch.manual_seed(1201)
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
                                                       trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.bfloat16,
            attn_implementation='sdpa').to('cuda').eval()
        torch.cuda.synchronize()
        self.loading_seconds = time.perf_counter() - started
        self.timings = []

    def generate(self, system, payload, schema=SUMMARY_SCHEMA):
        torch = self.torch
        messages = [dict(role='system', content=system + '\nReturn JSON matching this schema: ' + json.dumps(schema)),
                    dict(role='user', content=payload)]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=True, low_effort=True,
            tokenize=True, return_dict=True, return_tensors='pt').to('cuda')
        prompt_tokens = inputs['input_ids'].shape[1]
        if prompt_tokens + 4096 > 16384:
            raise ValueError('Summary request exceeds the configured context budget.')
        started = time.perf_counter()
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=4096, do_sample=True,
                                      temperature=1.0, top_p=0.95)
        torch.cuda.synchronize()
        generated = ids[0, prompt_tokens:]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        finished = len(generated) > 0 and int(generated[-1]) in eos
        content = final_content(self.tokenizer.decode(generated, skip_special_tokens=True), finished)
        self.timings.append(dict(seconds=time.perf_counter()-started,
                                 input_tokens=prompt_tokens, output_tokens=len(generated)))
        return content

    def summarize(self, segments):
        started = time.perf_counter()
        self.timings = []
        draft = summarize(segments, self.generate, lambda s: len(self.tokenizer.encode(s)))
        final, audit = verify_summary(draft, segments,
            lambda system, payload: self.generate(system, payload, VERIFY_SCHEMA))
        return dict(summary=final, summary_audit=audit, summary_calls=self.timings,
                    summary_loading_seconds=self.loading_seconds,
                    summary_seconds=time.perf_counter()-started,
                    summary_status='generated-draft',
                    summary_peak_allocated=self.torch.cuda.max_memory_allocated(),
                    summary_peak_reserved=self.torch.cuda.max_memory_reserved())
