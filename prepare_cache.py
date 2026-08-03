"""Encode the minimal public H3 training cache.

Text conditioning is cached at Qwen3-VL layer 50. This debug path encodes the
target video only. Audio latents are zero placeholders, and reference media
lists are not yet packed into H3 reference rows. See examples/README.md before
using this cache for a complete multimodal Ref2VA experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKLMiniMaxH3
from diffusers.modular_pipelines.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
    align_num_frames,
    audio_latent_num_frames,
    patchify_video_latents,
    video_latent_num_frames,
)
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor


def read_video(path: str, frames: int, height: int, width: int) -> torch.Tensor:
    container = av.open(path)
    stream = container.streams.video[0]
    decoded = []
    for frame in container.decode(stream):
        image = frame.to_image().convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        decoded.append(np.asarray(image, dtype=np.float32) / 255.0)
        if len(decoded) >= frames:
            break
    container.close()
    if not decoded:
        raise RuntimeError(f"No video frames: {path}")
    while len(decoded) < frames:
        decoded.append(decoded[-1])
    return torch.from_numpy(np.stack(decoded)).permute(3, 0, 1, 2).contiguous()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--frames", type=int, default=22)
    ap.add_argument("--encode-text", action="store_true")
    args = ap.parse_args()
    if args.height % 32 or args.width % 32 or args.frames != align_num_frames(args.frames):
        raise ValueError("H3 cache dimensions require height/width divisible by 32 and frames of form 17*n+5")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # H3's released VAE keeps its convolution path in float32. The transformer
    # cache is later stored in float32/BF16, but encoding must receive float32.
    vae = AutoencoderKLMiniMaxH3.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32).to(device)
    text_encoder = tokenizer = processor = None
    if args.encode_text:
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(args.model / "text_encoder", torch_dtype=dtype).to(device)
        tokenizer = Qwen2TokenizerFast.from_pretrained(args.model / "tokenizer")
        processor = Qwen3VLProcessor.from_pretrained(args.model / "processor")
    rows = [json.loads(line) for line in args.metadata.open() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    for index, row in enumerate(rows):
        video = read_video(row.get("target_video", row["video"]), args.frames, args.height, args.width)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None, None]
        video_latent = vae.encode(((video.to(device) - mean) / std)[None].to(torch.float32)).latent_dist.mode()
        latent_mean = torch.as_tensor(vae.config.latents_mean, device=device, dtype=video_latent.dtype).view(1, -1, 1, 1, 1)
        latent_std = torch.as_tensor(vae.config.latents_std, device=device, dtype=video_latent.dtype).view(1, -1, 1, 1, 1)
        video_latent = (video_latent - latent_mean) / latent_std
        video_rows = patchify_video_latents(video_latent.float(), (1, 2, 2))
        n_audio = audio_latent_num_frames(args.frames)
        audio_rows = torch.zeros(n_audio * 2, 32)
        prompt = row.get("caption", row.get("prompt", ""))
        if text_encoder is not None:
            ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            mm = processor.create_mm_token_type_ids(ids.tolist())
            out = text_encoder.model(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=torch.tensor(mm, device=device), use_cache=False, output_hidden_states=True)
            prompt_embed = out.hidden_states[50][0].to(torch.bfloat16).cpu()
        else:
            prompt_embed = torch.zeros((min(max(len(prompt), 8), 256), 5120), dtype=torch.bfloat16)
        item = {"video": video_rows.cpu().to(torch.float32), "audio": audio_rows, "prompt": prompt_embed, "caption": prompt, "height": args.height, "width": args.width, "latent_frames": video_latent.shape[2], "audio_frames": n_audio}
        torch.save(item, args.output / f"{index:08d}.pt")
        print(f"[{index + 1}/{len(rows)}] {args.output / f'{index:08d}.pt'} video={tuple(video_rows.shape)} text={tuple(prompt_embed.shape)}", flush=True)


if __name__ == "__main__":
    main()
