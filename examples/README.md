# Data examples

`manifest.example.jsonl` is a small, non-public-data example. Every line is
one training sample. The paths are intentionally placeholders and must be
replaced with files that the user is allowed to use and redistribute or keep
private on the training machine.

## Required fields

- `id`: unique sample identifier.
- `caption`: the instruction/caption used as text conditioning.
- `target_video`: the ground-truth video to reconstruct during training.
- `reference_images`: zero or more image paths, in the order referenced by
  `<Picture N>` tags in `caption` (MiniMax's native prompt format, 1-based —
  the same tags the official inference pipeline uses for reference images).
- `reference_videos`: zero or more video paths. Ordering is positional; the
  official H3 integration defines no in-caption tag for video references.
- `reference_audios`: zero or more audio paths. Ordering is positional, as
  with videos.

`target_audio` is optional in the manifest schema. It is included because H3
is an audio-video model, but the current `prepare_cache.py` debug path does
not encode it yet. Do not claim full audio-supervised Ref2VA training from
that script alone.

## Constructing a sample

1. Put the target video and all reference assets under a local dataset root.
2. Write one JSON object per line; do not embed binary data in the JSONL.
3. Use paths accessible from every worker when using multi-GPU training.
4. Keep the asset order stable; `<Picture N>` indices are 1-based.
5. Run the validator from the repository root:

```bash
python validate_manifest.py \
  --manifest path/to/train.jsonl \
  --check-files
```

For a public release, publish this schema and a few placeholder examples,
not private media, signed URLs, internal project metadata, or absolute paths
from the original dataset.

## Downloaded public example

`public_reference_examples.jsonl` demonstrates the same construction using
three Wikimedia Commons public-domain images in `public_assets/`. The examples
cover a two-image animal interaction, a single-person prompt, and a
multi-reference person-and-animal prompt. `target_video` is intentionally
`null`: replace it with a real training video before running the cache builder.
Source and license information is recorded in `public_assets/SOURCES.md`.
