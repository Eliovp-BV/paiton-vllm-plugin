# Model weights and existing downloads

Start with your [model's weight setup](../README.md#model-weights-and-existing-downloads).
The launcher, quantization, checkpoint revision and required companion models
must match that guide. A similarly named BF16, AWQ, MXFP4 or GGUF checkpoint is
not interchangeable with the selected release.

## Find your Hugging Face cache

`hf download OWNER/MODEL` without `--local-dir` uses the Hub cache. By default
this is `$HOME/.cache/huggingface/hub`. To select the cache already configured in
your current shell, including the legacy cache variable:

```bash
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
printf '%s\n' "$HF_HUB_CACHE"
```

If you downloaded with `--cache-dir` or used another disk, set that path instead:

```bash
export HF_HUB_CACHE="/absolute/path/to/your/hub-cache"
```

This variable points to the directory containing `models--OWNER--MODEL`, not to
one snapshot. `HF_HOME` is its usual parent and can also contain login state.
Use the same user account and cache location used for the original download.
See Hugging Face's [cache layout](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
and [environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

## Local folder or cached snapshot?

These are three distinct locations:

```text
/absolute/path/to/my-model/                   # --local-dir download
  config.json
  tokenizer.json
  model*.safetensors

/absolute/path/to/hub-cache/                  # HF_HUB_CACHE
  models--OWNER--MODEL/
    blobs/
    snapshots/COMMIT/                        # one pinned checkpoint
      config.json
      model*.safetensors

/absolute/path/to/runtime-cache/              # compiled/runtime caches
```

GGUF and image/video packages have different required files; use their model
guides. Preserve the checkpoint's configuration, tokenizer, index and all
referenced weight shards. An empty directory, a Git LFS pointer, or one shard
from a larger checkpoint is not a complete model.

The release uses a specific revision. A previous download of `main` might be a
different revision or might include only some files. Read the revision in the
model guide before selecting a snapshot. A cache lookup is not, on its own,
proof that all required files are present.

## Native CLI

In the model's supported Python/vLLM environment, select a complete folder with
`paiton --model-dir /absolute/path/to/model serve PRESET`. This also works with
a complete cached snapshot while its blob files remain available on the host.
The native CLI verifies the checkpoint and remembers an explicitly selected
folder after successful preparation.

Without an explicit folder or a remembered selection, the named preset checks
the configured Hugging Face cache and downloads missing pinned files. Use
`--prepare-only` before `serve` to prepare ahead of time. `--offline` also needs
the matching native bundle already prepared; cached weights alone are not a
fully prepared offline installation. See [native execution](NATIVE_EXECUTION.md).

## Containers and snapshot links

Container paths such as `/models/target` are inside Docker. Set the host paths
shown in the model guide; its mounts make those files visible inside Docker.
An exported host variable only affects a container if its launcher handles it.
The examples use absolute paths. Replace placeholder paths before running a
command. Docker's `--mount type=bind` rejects a missing source directory instead
of creating an empty model folder. The source weights are mounted read-only,
and a separate writable directory or volume holds runtime caches.

Hub snapshots often link to `../../blobs/...`. Mounting just a snapshot at an
unrelated container path can break those links. Follow the model-specific cache
instructions, which preserve the complete repository cache or use a standalone
copy. Do not move a snapshot away from its blobs or create an external symlink
that the container cannot follow.

The MiniCPM5, GPT-OSS and Qwen3-Coder local-folder examples mount a standalone
checkpoint into the container's expected pinned snapshot location. This exposes
existing files to the released loader without copying them or creating a fake
cache on the host. Use exactly the qualified source files; the destination
directory name alone does not establish checkpoint identity.

Runtime caches and prepared model directories are separate from the Hub cache.
Some packages create additional prepared tensors or shards even when source
weights are already downloaded. Those requirements are listed in each guide.
