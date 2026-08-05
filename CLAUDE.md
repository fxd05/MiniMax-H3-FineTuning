# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A minimal supervised fine-tuning path for [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), the open-weight omni-modal model that generates video + synchronized stereo audio. MiniMax ships no trainer and the released Diffusers integration is inference-only; this repo provides a rectified-flow trainer plus a latent-caching preprocessor. Model weights are governed by the MiniMax H3 Community License; this repo is code only.

## Environment

- Python 3.11 venv at `.venv` (created with `conda create -p .venv python=3.11 -y`), torch >= 2.8. `install_env.sh` verifies the env and installs the pinned revisions — the H3 classes (`MiniMaxH3Transformer3DModel`, `AutoencoderKLMiniMaxH3`, `diffusers.modular_pipelines.minimax_h3.packing`) are NOT in a released diffusers wheel, so diffusers is pinned to the git commit `abc5e9bf…`. That commit imports `huggingface_hub.get_cached_repo_tree`, which only exists in hub >= 1.0 (and its metadata requires `>=1.23`), hence the explicit `huggingface-hub>=1.23,<2` pin — the diffusers install is `--no-deps` so hub is never pulled in otherwise. transformers is pinned to v5.0.0: 4.57.x hard-requires hub<1.0 at import time (empty intersection with the hub pin), and Qwen3-VL — needed by the H3 text encoder — survives in 5.0.0. peft must be >= 0.20 (0.17 imports `HybridCache`, removed in transformers 5). `requirements.txt` mirrors the same pins; keep the two in sync when bumping.
- Point at a downloaded model checkout with `--model /path/to/MiniMax-H3` (subfolders `transformer` / `transformer_ref` / `vae` / `text_encoder` / `tokenizer` / `processor`).

## Standard workflow

```bash
bash install_env.sh                      # verify/repair env (run from repo root)

# 1. Manifest (JSONL, one sample per line; schema in examples/README.md)
python validate_manifest.py --manifest path/to/train.jsonl --check-files

# 2. Encode latents offline, one .pt per sample (debug scale: 256x256, 22 frames)
python prepare_cache.py --metadata path/to/train.jsonl --output cache/train \
  --model /path/to/MiniMax-H3 --height 256 --width 256 --frames 22 --encode-text

# 3. Train — single GPU smoke test, then multi-GPU
CUDA_VISIBLE_DEVICES=0 python train.py --model /path/to/MiniMax-H3 --variant ref2va \
  --cache cache/train --output runs/smoke --max-steps 10 --trainable heads
python -m torch.distributed.run --nproc_per_node 8 train.py --model /path/to/MiniMax-H3 \
  --variant ref2va --cache cache/train --output runs/heads_1000 --max-steps 1000 --trainable heads

# 4. Inference — official modular pipeline (this repo has no denoiser; generation
#    is the diffusers integration pinned in install_env.sh). Optionally patch a
#    trained heads checkpoint into the transformer.
CUDA_VISIBLE_DEVICES=0 python infer.py --model /path/to/MiniMax-H3 --task t2va \
  --prompt "A tiger walks through a snowy forest." --output runs/tiger.mp4
python infer.py --model /path/to/MiniMax-H3 --task ref2va \
  --reference image=examples/public_assets/amur_tiger.jpg \
  --reference image=examples/public_assets/flock_of_sheep.jpg \
  --prompt "<Picture 1>中的老虎扑向<Picture 2>中的羊" \
  --checkpoint runs/heads_1000/checkpoint-1000.pt --output runs/tiger_ft.mp4
```

Helpers: `select_1000.py` builds a deterministic, stratified 1000-sample manifest (seeded RNG, keeps all video-reference rows); `plot_loss.py --log runs/*/train.log --output loss.png` plots the `step=… loss=…` lines from `train.log`. There are no unit tests — the smoke test is `--max-steps 10 --trainable heads` on one GPU, and the primary correctness signal is loss behavior (stable ≈ 0.3–1.0; a *rising* loss means a convention got inverted, see below).

Inference facts (verified against the pinned diffusers commit): use `MiniMaxH3ModularPipeline.from_pretrained` for `t2va`/`fl2va`/`fl2va_last_frame`, `MiniMaxH3Ref2VABlocks().init_pipeline(...)` for `ref2va`; no `guidance_scale`/`negative_prompt` (guidance-distilled); `num_frames` snaps to 17n+5 and must land within 5–15 s; output is a state object with `get("videos")[0]`, `get("audio")[0]` (shape (1,2,num_samples)), `get("sampling_rate")`; mux with `diffusers.utils.encode_video(fps=24, audio=..., audio_sample_rate=...)`. The transformer is `pipe.transformer` (t2va/fl2va) or `pipe.transformer_ref` (ref2va) — patch train.py `heads` checkpoints with `load_state_dict(..., strict=False)` there; LoRA checkpoints need PEFT `load_adapter` instead. Memory (measured): the plain path loads every component in bf16 — 33B transformer ≈ 63 GB, Qwen3-VL-32B text encoder ≈ 63 GB, VAEs ≈ 10 GB — ≈ 135 GB of weights alone, so even a 90 GB card OOMs. infer.py therefore defaults to the official single-card recipe: a `ComponentsManager` registers the components and `enable_auto_cpu_offload` swaps them on/off the GPU, so the weights sit in host RAM (~135 GB) and one 80–96 GB card runs generation (`--reserve-margin` tunes the free-VRAM margin, `--no-offload` restores the plain path). Attention is still full self-attention: without the Hopper-only `--attention-backend _flash_3_hub`, keep height/width small (256–384) or the per-head score matrix OOMs. Two 80 GB cards can skip all of that with `device_map` (docs recipe).

## Architecture

**Two-stage design** so the 33B transformer is the only thing in GPU memory during training:

1. `prepare_cache.py` — encodes each sample offline into one `.pt` (dict with keys `video`, `audio`, `prompt`, `caption`, `height`, `width`, `latent_frames`, `audio_frames`): target video → patchified H3-VisualVAE latents, caption → Qwen3-VL **layer-50 hidden states**, audio → **zero placeholder** (not yet encoded; the trainer detects this and zeroes the audio loss). Note the VAE must run in float32 (H3's released VAE keeps convolutions in fp32); the transformer cache is stored float32/bf16.
2. `train.py` — loads only the transformer, rebuilds the packed sequence layout with the official `build_packed_sequence` / `build_row_timesteps` (from `diffusers.modular_pipelines.minimax_h3.packing`), and optimizes a rectified-flow MSE loss. `H3Cache` shards `.pt` files by rank and iterates without shuffling; batch size is fixed at 1 sequence per step.

Trainable modes: `heads` (`proj_out` + `audio_proj_out`, ~1M params), `lora` (PEFT on `to_qkv`, `to_out.0`, `linear_1`, `linear_2`), `all` (full 33B, requires `--strategy deepspeed` with `deepspeed_zero3.json`). `--variant` selects `transformer_ref` (Ref2VA) vs `transformer` (FL2VA) subfolder. Logging goes to stdout and `runs/<name>/train.log` (UTC timestamps).

## The five conventions (read FIXES.md before touching training math)

The first version of this trainer produced a rising loss; every fix below was derived from the official Diffusers integration source, and any regression reintroduces silent weight corruption:

1. **Timestep direction**: the transformer's time input is `t = 1 − σ` (t=1 clean, t=0 pure noise). Noise with `x_t = (1−σ)x₀ + σ·ε` but pass `1.0 − σ` to `build_row_timesteps`; conditioning rows stay pinned at 0.999.
2. **Velocity sign**: the model predicts a data-ward velocity `v = x₀ − ε` (scheduler denoises `x₀ = x_t + (1−t)·v`, note the `+`). Regress `pred → (clean − noise)`, never the SD3/Wan-style `(ε − x₀)`.
3. **Zero-placeholder audio**: mask the audio loss term by multiplying by `0.0` when cached audio is all-zero — never drop the term entirely, or DDP crashes (`audio_proj_out` gets no gradient).
4. **Dual shifted schedules**: sample one `u ~ U(0.02, 0.98)` and map through both shift curves `σ = shift·u / (1 + (shift−1)·u)` — video shift 12.0, audio shift 3.0 — mirroring the two schedulers advancing in lockstep at inference.
5. **Checkpointing**: for `heads`/`lora`, save only tensors with `requires_grad=True` (a few MB). Full-state serialization of the 33B model (~66 GB) every 100 steps stalls rank 0 and trips the NCCL watchdog (SIGABRT).

## Model constraints

`num_frames % 17 == 5` (`align_num_frames`), height/width divisible by 32, 24 fps video, 32 kHz stereo audio, `audio_frames = audio_latent_num_frames(num_frames)` (≈ round(frames/24·40)). Sparse attention is not released — training runs full attention, so keep clips short and resolution moderate.
