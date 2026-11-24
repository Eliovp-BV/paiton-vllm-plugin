"""Check the Inkling 2 port against reference fixtures (bring-up tool, eager mode, one sequence, greedy).

Usage: PAITON_INK2_TAP_DIR=<empty dir> python3 port_check.py <model dir> <fixture dir> <out dir>
For every `<fixture dir>/<name>.npz` (keys input_ids, greedy) it feeds the fixture's token ids to vLLM, generates as
many greedy tokens as the fixture holds, and writes `<out dir>/<name>.npz` in the fixture layout (input_ids,
last_logits, argmax, greedy, hidden) from the prefill forward's tap file, so the reference's compare script reads both.
"""
import glob
import os
import sys

import numpy as np
import torch
from vllm import LLM, SamplingParams


def main() -> None:
    model_dir, fixture_dir, out_dir = sys.argv[1:4]
    tap_dir = os.environ["PAITON_INK2_TAP_DIR"]
    llm = LLM(model=model_dir, quantization="inkling2", enforce_eager=True, max_model_len=8192, max_num_seqs=1,
              limit_mm_per_prompt={"image": 0}, enable_prefix_caching=False, seed=0)
    for path in sorted(glob.glob(f"{fixture_dir}/*.npz")):
        name, fixture = os.path.basename(path)[:-4], np.load(path)
        ids = fixture["input_ids"].tolist()
        seen = set(os.listdir(tap_dir))
        params = SamplingParams(temperature=0.0, max_tokens=len(fixture["greedy"]), skip_special_tokens=False)
        out = llm.generate([{"prompt_token_ids": ids}], params)[0].outputs[0]
        prefill = None
        for tap in sorted(set(os.listdir(tap_dir)) - seen):
            data = torch.load(f"{tap_dir}/{tap}")
            if data["positions"].tolist() == list(range(len(ids))):
                prefill = data
                break
        assert prefill is not None, f"{name}: no tap of a {len(ids)}-token forward (prompt split over forwards?)"
        np.savez_compressed(f"{out_dir}/{name}.npz", input_ids=np.array(ids), greedy=np.array(out.token_ids),
                            last_logits=prefill["last_logits"].numpy(), argmax=prefill["argmax"].numpy(),
                            hidden=prefill["hidden"].numpy(), final=prefill["final"].numpy())
        same = np.array(out.token_ids)[: len(fixture["greedy"])] == fixture["greedy"][: len(out.token_ids)]
        print(name, "tokens", len(ids), "generated", len(out.token_ids), "finish", out.finish_reason,
              "greedy prefix equal to fixture", len(same) if same.all() else int(same.argmin()), repr(out.text)[:300],
              flush=True)
    print("PORT CHECK DONE")


if __name__ == "__main__":  # vLLM spawns its engine process
    main()
