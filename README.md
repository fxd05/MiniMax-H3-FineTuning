# MiniMax-H3-FineTuning

English | [中文](README_zh.md)

A minimal, working fine-tuning path for [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) —
the open-weight omni-modal model that jointly generates video **and** synchronized stereo audio.

MiniMax released the H3 weights "to support further development, including fine-tuning", but ships **no
trainer**. The Hugging Face Diffusers integration is inference-only. This repository fills the gap with a
supervised rectified-flow trainer (~150 lines) built directly on the official Diffusers integration, plus a
latent-caching preprocessor — and documents the four H3-specific numeric conventions that will silently
corrupt your weights if you get them wrong (see [FIXES.md](FIXES.md); we learned them the hard way).

## How it works

**Two-stage design** (so the 33B transformer is the only thing in GPU memory during training):

1. **`prepare_cache.py`** — encodes each training sample offline: target video → H3-VisualVAE latents
   (patchified rows), target audio → H3-AudioVAE latents, caption → Qwen3-VL layer-50 hidden states.
   One `.pt` file per sample.
2. **`train.py`** — loads only the transformer (`transformer` = FL2VA variant or `transformer_ref` = Ref2VA
   variant), rebuilds the packed video/audio/text sequence layout with the official
   `build_packed_sequence` / `build_row_timesteps` utilities, and optimizes a rectified-flow MSE loss.

**Training objective.** H3's released checkpoints are guidance-distilled rectified-flow models with two
peculiarities relative to the common SD3/Wan-style convention:

- the transformer's time input is `t = 1 − σ` (t=1 clean, t=0 pure noise), and
- the transformer predicts a **data-ward velocity** `v = x₀ − ε` (the scheduler reconstructs
  `x₀ = x_t + (1−t)·v` — note the `+`).

The loss therefore noises `x_t = (1−σ)·x₀ + σ·ε` and regresses `pred → (x₀ − ε)`, with **two different σ
per step**: video and audio use separate shifted schedules (`σ = shift·u / (1+(shift−1)·u)`, shift 12.0 for
video and 3.0 for audio) sampled at the same `u`, mirroring how the two schedulers advance in lockstep at
inference.

**Trainable-parameter modes:**

| `--trainable` | What trains | Use case |
|---|---|---|
| `heads` (default) | `proj_out` + `audio_proj_out` (~1M params) | smoke test the whole pipeline on one GPU |
| `lora` | PEFT LoRA on `to_qkv`, `to_out.0`, `linear_1`, `linear_2` | practical single-node fine-tuning |
| `all` | full 33B | requires `--strategy deepspeed` (ZeRO-3 config included) |

Checkpoints store only trainable tensors for `heads`/`lora` (a few MB instead of ~66 GB — full-state
serialization every 100 steps stalls rank 0 long enough to trip the NCCL watchdog; see FIXES.md #5).

## Quickstart

```bash
# 1. Environment: Python 3.11, torch >= 2.8. The H3 classes are NOT in a released
#    diffusers wheel yet — install_env.sh installs the pinned integration revision.
bash install_env.sh

# 2. Write a manifest (schema + placeholder samples in examples/)
python validate_manifest.py --manifest path/to/train.jsonl --check-files

# 3. Cache latents (debug scale: 256x256, 22 frames; raise for real runs)
python prepare_cache.py \
  --metadata path/to/train.jsonl \
  --output cache/train \
  --model /path/to/MiniMax-H3 \
  --height 256 --width 256 --frames 22 --encode-text

# 4. Train (single GPU smoke)
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model /path/to/MiniMax-H3 --variant ref2va \
  --cache cache/train --output runs/smoke \
  --max-steps 10 --trainable heads

# 4'. Multi-GPU (8x, DDP)
python -m torch.distributed.run --nproc_per_node 8 train.py \
  --model /path/to/MiniMax-H3 --variant ref2va \
  --cache cache/train --output runs/heads_1000 \
  --max-steps 1000 --trainable heads
```

Constraints inherited from the model: `num_frames % 17 == 5`, height/width divisible by 32, 24 fps video,
32 kHz stereo audio, `audio_latent_frames = round(num_frames / 24 * 40)`.

## Does it work?

With the conventions wrong (timestep direction + velocity sign inverted), a heads-only run shows loss
*rising* (7.2 → 9.5 over 10 steps). With the fixes in this repo, the same setup trains at loss ≈ 0.3–1.0,
stable across 1000 steps on 8×A800, and the trained heads produce coherent generations when patched back
into the inference pipeline. Details in [FIXES.md](FIXES.md).

## Current limitations

- `prepare_cache.py` is the minimal cache path: it encodes the **target video only** — audio latents are
  zero placeholders (the trainer detects this and zeroes the audio loss) and reference media are not yet
  encoded into the conditioning sequence. Extending the cache to real audio + reference encoding is the
  main open TODO for faithful Ref2VA training.
- No sample shuffling between epochs; batch size is fixed at 1 sequence per step.
- Sparse attention (used in H3's final training stage) is not released; training runs full attention, so
  keep clips short / resolution moderate.

## Legal

Model weights are governed by the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
(territory and usage restrictions apply; fine-tuned weights are Model Derivatives). This repository contains
code only — no model weights and no training media.
