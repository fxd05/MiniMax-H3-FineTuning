"""MiniMax-H3 supervised rectified-flow fine-tuning over cached latents.

Trains the released H3 transformer (FL2VA or Ref2VA variant) with a
rectified-flow objective that matches the conventions of the official
Diffusers MiniMax-H3 integration. See FIXES.md for the four numeric
conventions that are easy to get wrong (timestep direction, velocity sign,
placeholder-audio handling, dual shifted noise schedules).
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from diffusers import MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.packing import build_packed_sequence, build_row_timesteps


class H3Cache(Dataset):
    def __init__(self, root: Path, rank: int = 0, world_size: int = 1):
        files = sorted(Path(p) for p in glob.glob(str(root / "*.pt")))
        self.files = files[rank::world_size]
        if not self.files:
            raise FileNotFoundError(f"No .pt cache files under {root}; run prepare_cache.py first")
    def __len__(self): return len(self.files)
    def __getitem__(self, i): return torch.load(self.files[i], map_location="cpu", weights_only=False)


def layout_for(item, device):
    tags = torch.ones(item["prompt"].shape[0], dtype=torch.long)
    layout = build_packed_sequence(tags, item["latent_frames"], item["height"] // 16, item["width"] // 16, item["audio_frames"], (1, 2, 2))
    return {k: getattr(layout, k).to(device) for k in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True); ap.add_argument("--cache", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--variant", choices=("fl2va", "ref2va"), default="ref2va")
    ap.add_argument("--strategy", choices=("ddp", "deepspeed"), default="ddp")
    ap.add_argument("--deepspeed-config", type=Path, default=Path(__file__).resolve().parent / "deepspeed_zero3.json")
    ap.add_argument("--max-steps", type=int, default=1000); ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--trainable", choices=("heads", "lora", "all"), default="heads"); ap.add_argument("--resume", type=Path)
    ap.add_argument("--video-shift", type=float, default=12.0); ap.add_argument("--audio-shift", type=float, default=3.0)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed and args.strategy == "ddp":
        torch.distributed.init_process_group("nccl")
    elif distributed and args.strategy == "deepspeed":
        import deepspeed
        deepspeed.init_distributed("nccl")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    log_handle = None
    if rank == 0:
        log_handle = (args.output / "train.log").open("a", buffering=1)
        log_handle.write(f"{datetime.now(timezone.utc).isoformat()} start world_size={world_size} variant={args.variant} trainable={args.trainable}\n")
    model = MiniMaxH3Transformer3DModel.from_pretrained(args.model, subfolder="transformer_ref" if args.variant == "ref2va" else "transformer", torch_dtype=torch.bfloat16).to(device)
    model.enable_gradient_checkpointing()
    for p in model.parameters(): p.requires_grad_(args.trainable == "all")
    if args.trainable == "heads":
        for name, p in model.named_parameters():
            p.requires_grad_("proj_out" in name or "audio_proj_out" in name)
    elif args.trainable == "lora":
        from peft import LoraConfig
        model.add_adapter(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["to_qkv", "to_out.0", "linear_1", "linear_2"], init_lora_weights="gaussian"))
    if distributed and args.strategy == "ddp":
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        print(f"trainable parameters: {sum(p.numel() for p in params):,} world_size={world_size}")
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    engine = None
    if args.strategy == "deepspeed":
        import deepspeed
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=params,
            optimizer=optimizer,
            config=str(args.deepspeed_config),
        )
        engine = model
    data = H3Cache(args.cache, rank, world_size); step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False); model.load_state_dict(state["model"], strict=False); optimizer.load_state_dict(state["optimizer"]); step = state["step"]
    model.train()
    while step < args.max_steps:
        for item in data:
            if step >= args.max_steps: break
            layout = layout_for(item, device)
            # [FIX4] Sample noise levels from BOTH shifted schedules: sigma = shift*u/(1+(shift-1)*u).
            # At inference the video/audio schedulers advance in lockstep with different shifts
            # (video 12.0, audio 3.0), so training must cover the same (sigma_v, sigma_a) pairings.
            u = torch.rand((), device=device).clamp(0.02, 0.98)
            sigma_v = args.video_shift * u / (1.0 + (args.video_shift - 1.0) * u)
            sigma_a = args.audio_shift * u / (1.0 + (args.audio_shift - 1.0) * u)
            video = item["video"].to(device, torch.bfloat16); audio = item["audio"].to(device, torch.bfloat16); clean_v, clean_a = video.clone(), audio.clone()
            noise_v, noise_a = torch.randn_like(video), torch.randn_like(audio); video = (1 - sigma_v) * clean_v + sigma_v * noise_v; audio = (1 - sigma_a) * clean_a + sigma_a * noise_a
            # [FIX1] Model contract: timestep == 1 - sigma (t=1 clean, t=0 pure noise; see the
            # MiniMaxH3Scheduler.step docstring). Passing sigma directly inverts the conditioning.
            # Conditioning rows stay pinned at 0.999, which is already in the 1-sigma convention.
            timesteps, timestep_indices = build_row_timesteps(type("L", (), {"sequence_length": int(layout["token_tags"].numel()), "video_indices": layout["video_indices"], "audio_indices": layout["audio_indices"], "num_condition_video_rows": 0, "num_condition_audio_rows": 0})(), float(1.0 - sigma_v), float(1.0 - sigma_a), 0.999, 0.999)
            pred_v, pred_a = model(hidden_states=video[None], audio_hidden_states=audio[None], encoder_hidden_states=item["prompt"].to(device, torch.bfloat16)[None], timestep=timesteps.to(device), timestep_indices=timestep_indices.to(device), attention_kwargs=None, return_dict=False, **layout)
            # [FIX2] The model outputs a data-ward velocity v = x0 - noise (the scheduler denoises
            # with x0 = x_t + (1-t)*v, "note the +" — opposite of the usual flow-matching target).
            # Regressing (noise - x0) flips the sign and actively corrupts the weights.
            # [FIX3] When the cache stores zero-placeholder audio, zero the audio loss weight instead
            # of dropping the term, so DDP still sees gradients flow through audio_proj_out.
            audio_weight = 1.0 if bool(clean_a.abs().sum() > 0) else 0.0
            loss = F.mse_loss(pred_v.float(), (clean_v - noise_v)[None].float()) + audio_weight * F.mse_loss(pred_a.float(), (clean_a - noise_a)[None].float())
            if engine is not None:
                engine.backward(loss)
                engine.step()
            else:
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()
            step += 1
            if rank == 0 and (step == 1 or step % 10 == 0):
                message = f"step={step} loss={loss.item():.6f} lr={args.lr:.8g} sigma_v={float(sigma_v):.3f} audio_w={audio_weight}"
                print(message, flush=True)
                log_handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
            if step % 100 == 0 or step == args.max_steps:
                if engine is not None:
                    engine.save_checkpoint(str(args.output), tag=f"checkpoint-{step}")
                elif rank == 0:
                    # [FIX5] Save only trainable tensors for heads/lora modes. Serializing the full
                    # 33B state dict (~66 GB) every 100 steps stalls rank 0 for minutes while other
                    # ranks wait at the next collective, tripping the NCCL watchdog (SIGABRT).
                    core = model.module if distributed else model
                    saved_model = {k: v for k, v in core.state_dict().items() if k in {n for n, p in core.named_parameters() if p.requires_grad}} if args.trainable != "all" else core.state_dict()
                    torch.save({"model": saved_model, "optimizer": optimizer.state_dict() if args.trainable != "all" else None, "step": step}, args.output / f"checkpoint-{step}.pt")
    if distributed:
        torch.distributed.destroy_process_group()
    if log_handle is not None:
        log_handle.write(f"{datetime.now(timezone.utc).isoformat()} finished step={step}\n")
        log_handle.close()


if __name__ == "__main__": main()
