# The five fixes

The first version of this trainer produced a *rising* loss (7.2 → 9.5 in 10 steps, heads mode). Every one
of the issues below is invisible at the "code runs fine" level; all evidence comes from the official
Diffusers MiniMax-H3 integration source (scheduler/packing docstrings), not guesswork.

## 1. Timestep direction is inverted relative to σ

`MiniMaxH3Scheduler.step` docstring:

> The current timestep, one of `self.timesteps` (so **`timestep == 1 - sigma`**).

The transformer's time input is `1 − σ` (1 = clean, 0 = pure noise). If you noise with
`x_t = (1−t)·x₀ + t·ε` — where your `t` *is* σ — and then feed that same `t` to `build_row_timesteps`,
the conditioning is exactly inverted: an almost-clean sample is labeled "almost pure noise". A telltale
inconsistency: the conditioning rows use `0.999` (≈ clean in the `1−σ` convention) while the generated
rows got raw σ.

**Fix:** pass `1.0 − σ_video` and `1.0 − σ_audio`.

## 2. The velocity target sign is opposite to the usual flow-matching convention

Same docstring:

> The model output is a **data-ward velocity**, so the denoised estimate is `x0 = x_t + (1 - t) * v` —
> **note the `+`, the opposite of the usual flow-match convention.**

So the model predicts `v = x₀ − ε`. The habitual SD3/Wan-style target `(ε − x₀)` is the *negative* of what
this model natively outputs — training on it actively pushes the weights the wrong way (hence the rising
loss).

**Fix:** regress `pred → (clean − noise)`.

## 3. Zero-placeholder audio must not train the audio head

The minimal cache path stores zero tensors for audio. Training `audio_proj_out` against
`(0 − ε)` teaches the audio head to predict pure noise. Simply *skipping* the audio loss term breaks DDP
(`audio_proj_out` receives no gradient and DDP raises). 

**Fix:** multiply the audio loss by `0.0` when the cached audio is all-zero — gradients still flow
(as zeros) so DDP stays happy, and the audio head is untouched.

## 4. Video and audio use different shifted noise schedules

At inference the two schedulers advance in lockstep with different shifts
(`σ = shift·u / (1 + (shift−1)·u)`, video shift 12.0, audio shift 3.0), so at any step σ_video ≠ σ_audio.
Training both modalities at one shared uniform `t` never visits the (σ_v, σ_a) pairings that inference
actually traverses.

**Fix:** sample one `u ~ U(0.02, 0.98)` and map it through both shift curves.

## 5. Full-state checkpointing trips the NCCL watchdog

Saving `model.state_dict()` for a 33B bf16 model writes ~66 GB per checkpoint. In heads/LoRA modes that is
all frozen weight. Worse than the disk cost: rank 0 blocks inside `torch.save` for minutes while the other
ranks wait at the next collective — after the timeout the NCCL watchdog kills the job with SIGABRT.

**Fix:** for `heads`/`lora`, save only tensors with `requires_grad=True` (a few MB), keep full state only
for `--trainable all` (where DeepSpeed handles sharded saving anyway).

## Verification

Same data, same hyperparameters, heads mode, 8 GPUs:

| | loss @ step 1→10 | 1000-step behavior |
|---|---|---|
| before fixes | 7.25 → 9.54 (rising) | — |
| after fixes | ≈ 0.3–1.0 (stable, σ-dependent variance) | completes; checkpoints are ~8 MB |

A/B inference with the trained heads patched into the official pipeline (same seed, same prompt, same
references) produces coherent generations; the video head moves ~4–7% in relative norm after 1000 steps at
lr 1e-5 while the audio head stays put (weight-decay drift only), as intended with placeholder audio.
