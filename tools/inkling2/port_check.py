"""Check the Inkling 2 port against reference fixtures (bring-up tool, eager mode, greedy).

Usage: python3 port_check.py <model dir> <fixture dir> <out dir> [teacher | gen | image] [key=value ...]
For every `<fixture dir>/<name>.npz` (keys input_ids, greedy):
  (no mode)  feeds the fixture's token ids, generates as many greedy tokens as the fixture holds and writes
             `<out dir>/<name>.npz` in the fixture layout (input_ids, last_logits, argmax, greedy, hidden) from the
             prefill forward's tap file, so the reference's compare script reads both. Needs PAITON_INK2_TAP_DIR
             (an empty directory) and one sequence at a time.
  teacher    feeds the prompt and the fixture's greedy tokens in one forward and prints how often the port's top-1
             token equals the fixture's at the generated positions (nothing is written; needs the tap directory).
             `<fixture dir>` may be several directories joined by commas.
  gen        generates all prompts in one call (no tap) and writes input_ids and greedy only: the decode path
             under the serving options below, to be checked afterwards with `teacher`.
  image      `<fixture dir>` is the vision fixture file (`fixtures/vision/vision_fixture.npz`): its image prompt
             runs with the image through the vision encoder (images are switched on for this mode only). Prints
             the last prompt position's logits against the fixture's (top-1, KL) and the greedy reply. Needs the
             tap directory; with PAITON_INK2_OPS=native also PAITON_INK2_VISION_SO.
Options: seqs=N (max_num_seqs), batched=N (max_num_batched_tokens: prompts longer than N are prefilled in chunks),
prefix=1 (prefix caching on; `gen` then generates everything a second time and prints whether the tokens repeat),
maxlen=N (max_model_len), kvbytes=N (kv_cache_memory_bytes), tokens=N (`gen`: exactly N tokens per prompt, the end
token ignored, whatever the fixture's `greedy` holds).
"""
import glob
import os
import sys

import numpy as np
import torch
from vllm import LLM, SamplingParams


def main() -> None:
    model_dir, fixture_dirs, out_dir = sys.argv[1:4]
    mode = next((a for a in sys.argv[4:] if "=" not in a), "")
    opt = {k: int(v) for k, v in (a.split("=") for a in sys.argv[4:] if "=" in a)}
    assert mode in ("", "teacher", "gen", "image") and not set(opt) - {"seqs", "batched", "prefix", "maxlen", "kvbytes", "tokens"}
    tap_dir = os.environ.get("PAITON_INK2_TAP_DIR")
    assert (tap_dir is None) == (mode == "gen"), "the tap is for one sequence per forward: unset it for gen only"
    extra = {"max_num_batched_tokens": opt["batched"]} if "batched" in opt else {}
    if "kvbytes" in opt:
        extra["kv_cache_memory_bytes"] = opt["kvbytes"]
    llm = LLM(model=model_dir, quantization="inkling2", enforce_eager=True, max_model_len=opt.get("maxlen", 8192),
              max_num_seqs=opt.get("seqs", 1), limit_mm_per_prompt={"image": int(mode == "image")},
              enable_prefix_caching=bool(opt.get("prefix")), seed=0, **extra)
    if mode == "image":
        from PIL import Image

        fixture = np.load(fixture_dirs)
        ids, greedy = fixture["prompt_input_ids"].tolist(), fixture["prompt_greedy"]
        first, pads = ids.index(200054), ids.count(200054)
        # The processor expands the one pad after the image marker to the image's pads itself.
        prompt = {"prompt_token_ids": ids[: first + 1] + ids[first + pads :],
                  "multi_modal_data": {"image": Image.fromarray(fixture["img1_image_rgb_uint8"])}}
        params = SamplingParams(temperature=0.0, max_tokens=len(greedy), skip_special_tokens=False)
        out = llm.generate([prompt], params)[0].outputs[0]
        prefill = None
        for tap in sorted(os.listdir(tap_dir)):
            data = torch.load(f"{tap_dir}/{tap}")
            if prefill is None and data["positions"].tolist() == list(range(len(ids))):
                prefill = data
            os.remove(f"{tap_dir}/{tap}")
        assert prefill is not None, f"no tap of a {len(ids)}-token forward (image slots differ from the fixture?)"
        port, ref = prefill["last_logits"].double(), torch.from_numpy(fixture["prompt_last_logits"]).double()
        size = min(len(port), len(ref))
        port, ref = port[:size], ref[:size]
        kl = float((ref.log_softmax(-1).exp() * (ref.log_softmax(-1) - port.log_softmax(-1))).sum())
        same = np.array(out.token_ids)[: len(greedy)] == greedy[: len(out.token_ids)]
        print("image prompt tokens", len(ids), "image slots", pads, "last logits: top-1", int(port.argmax()), "fixture",
              int(ref.argmax()), "top-5 shared", len(set(port.topk(5).indices.tolist()) & set(ref.topk(5).indices.tolist())),
              f"KL(fixture || port) {kl:.3e} max |difference| {float((port - ref).abs().max()):.4f}", flush=True)
        print("generated", len(out.token_ids), "finish", out.finish_reason, "greedy prefix equal to fixture",
              len(same) if same.all() else int(same.argmin()), "of", len(greedy), repr(out.text), flush=True)
        np.savez_compressed(f"{out_dir}/image_prompt.npz", greedy=np.array(out.token_ids), last_logits=port.numpy())
        print("PORT CHECK DONE")
        return
    paths = [p for d in fixture_dirs.split(",") for p in sorted(glob.glob(f"{d}/*.npz"))]
    if mode == "gen":
        fixtures = [np.load(p) for p in paths]
        prompts = [{"prompt_token_ids": f["input_ids"].tolist()} for f in fixtures]
        params = [SamplingParams(temperature=0.0, max_tokens=opt.get("tokens", len(f["greedy"])),
                                 ignore_eos="tokens" in opt, skip_special_tokens=False) for f in fixtures]
        first = [o.outputs[0].token_ids for o in llm.generate(prompts, params)]
        for path, fixture, tokens in zip(paths, fixtures, first):
            np.savez(f"{out_dir}/{os.path.basename(path)}", input_ids=fixture["input_ids"], greedy=np.array(tokens))
            print(os.path.basename(path)[:-4], "tokens", len(fixture["input_ids"]), "generated", len(tokens), flush=True)
        if opt.get("prefix"):
            second = [o.outputs[0].token_ids for o in llm.generate(prompts, params)]
            for path, a, b in zip(paths, first, second):
                same = [x == y for x, y in zip(a, b)]
                print(os.path.basename(path)[:-4], "second pass (prefix cache hit) equal to the first",
                      f"{sum(same)}/{len(a)}", "first miss at", same.index(False) if False in same else None, flush=True)
        print("PORT CHECK DONE")
        return
    for path in paths:
        name, fixture = path if "," in fixture_dirs else os.path.basename(path)[:-4], np.load(path)
        ids = fixture["input_ids"].tolist()
        prompt_len, greedy = len(ids), fixture["greedy"]
        if mode == "teacher":
            ids += greedy[:-1].tolist()
        seen = set(os.listdir(tap_dir))
        params = SamplingParams(temperature=0.0, max_tokens=1 if mode else len(greedy), skip_special_tokens=False)
        out = llm.generate([{"prompt_token_ids": ids}], params)[0].outputs[0]
        prefill = None
        for tap in sorted(set(os.listdir(tap_dir)) - seen):
            data = torch.load(f"{tap_dir}/{tap}")
            if prefill is None and data["positions"].tolist() == list(range(len(ids))):
                prefill = data
            os.remove(f"{tap_dir}/{tap}")  # a 2,400-token forward's tap is 1 GB
        assert prefill is not None, f"{name}: no tap of a {len(ids)}-token forward (prompt split over forwards?)"
        if mode == "teacher":
            same = prefill["argmax"].numpy()[prompt_len - 1 :] == greedy
            print(name, "teacher-forced top-1 equal to fixture", f"{same.sum()}/{len(same)}", round(same.mean(), 4),
                  "first miss at", int(same.argmin()) if not same.all() else None, flush=True)
            continue
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
