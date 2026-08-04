"""MiniMax-H3 inference through the official diffusers modular pipeline.

This repo trains; generation is the official modular-pipeline integration,
pinned by install_env.sh / requirements.txt to the diffusers commit
abc5e9bf... (the H3 classes are not in a released wheel). API facts below
are verified against that commit's source/docs:

- MiniMaxH3ModularPipeline.from_pretrained serves tasks t2va / fl2va /
  fl2va_last_frame; task ref2va is loaded via
  MiniMaxH3Ref2VABlocks().init_pipeline(...).
- No guidance_scale / negative_prompt: the checkpoint is guidance-distilled.
- The two schedulers advance in lockstep (video shift=12.0, audio shift=3.0)
  — the same dual-shift convention train.py optimizes for.
- Output is a state object: state.get("videos")[0] (video frames),
  state.get("audio")[0] (shape (1, 2, num_samples)), state.get("sampling_rate").

Memory: the 33B transformer is ~66 GB in bf16 — one 80 GB GPU for the
plain path here; see the diffusers docs for int8 / offload / device_map
recipes on smaller cards.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from diffusers import MiniMaxH3ModularPipeline, MiniMaxH3Ref2VABlocks
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
from diffusers.modular_pipelines.minimax_h3.packing import align_num_frames
from diffusers.utils import encode_video


def build_pipe(model: Path, task: str):
    if task == "ref2va":
        # Ref2VA registers the transformer as `transformer_ref`.
        pipe = MiniMaxH3Ref2VABlocks().init_pipeline(str(model))
    else:
        pipe = MiniMaxH3ModularPipeline.from_pretrained(str(model))
    pipe.load_components(dtype=torch.bfloat16)
    pipe.to("cuda")
    return pipe


def apply_checkpoint(pipe, checkpoint: Path, task: str) -> None:
    """Patch a train.py checkpoint (heads mode: proj_out + audio_proj_out) in.

    train.py saves only requires_grad tensors, so strict=False is required.
    LoRA checkpoints from train.py are not handled here — load those with
    PEFT (pipe.transformer.load_adapter(...)) instead.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    transformer = getattr(pipe, "transformer_ref" if task == "ref2va" else "transformer")
    missing, unexpected = transformer.load_state_dict(state["model"], strict=False)
    print(f"patched {checkpoint}: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing:", missing[:10])
    if unexpected:
        print("  unexpected:", unexpected[:10])


def main() -> None:
    ap = argparse.ArgumentParser(description="MiniMax-H3 video+audio generation")
    ap.add_argument("--model", type=Path, required=True, help="MiniMax-H3 repo checkout (same layout train.py expects)")
    ap.add_argument("--task", choices=("t2va", "fl2va", "fl2va_last_frame", "ref2va"), default="t2va")
    ap.add_argument("--prompt")
    ap.add_argument("--image", type=Path, help="fl2va: first-frame keyframe")
    ap.add_argument("--last-image", type=Path, help="fl2va_last_frame: last-frame keyframe")
    ap.add_argument("--reference", action="append", metavar="TYPE=PATH",
                    help="ref2va: repeatable, e.g. image=a.jpg video=b.mp4 audio=c.wav (max 9i/3v/3a)")
    ap.add_argument("--num-frames", type=int, default=124, help="snapped up to 17*n+5; duration must stay 5-15 s")
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-type", choices=("pil", "pt", "np", "latent"), default="pil")
    ap.add_argument("--output", type=Path, default=Path("runs/infer.mp4"))
    ap.add_argument("--checkpoint", type=Path, help="optional train.py heads checkpoint to patch in")
    args = ap.parse_args()

    if args.height % 32 or args.width % 32:
        raise SystemExit("height/width must be multiples of 32")
    num_frames = align_num_frames(args.num_frames)
    if not 5 * 24 <= num_frames <= 15 * 24:
        print(f"warning: {num_frames} frames at 24 fps is outside 5-15 s", file=sys.stderr)

    pipe = build_pipe(args.model, args.task)
    if args.checkpoint:
        apply_checkpoint(pipe, args.checkpoint, args.task)

    kwargs = {}
    if args.prompt:
        kwargs["prompt"] = args.prompt
    if args.image:
        kwargs["image"] = Image.open(args.image).convert("RGB")
    if args.last_image:
        kwargs["last_image"] = Image.open(args.last_image).convert("RGB")
    if args.reference:
        refs = []
        for spec in args.reference:
            kind, _, path = spec.partition("=")
            if kind not in ("image", "video", "audio") or not path:
                raise SystemExit(f"--reference must be TYPE=PATH with TYPE in image|video|audio, got {spec!r}")
            refs.append(MiniMaxH3Reference(**{kind: path}))
        kwargs["references"] = refs
        if args.task != "ref2va":
            print("note: references passed with a non-ref2va task; ignoring", file=sys.stderr)

    kwargs.update(num_frames=num_frames, height=args.height, width=args.width,
                  num_inference_steps=args.steps,
                  generator=torch.Generator(device="cuda").manual_seed(args.seed),
                  output_type=args.output_type)
    state = pipe(**kwargs)

    video = state.get("videos")[0]
    audio = state.get("audio")[0]
    sampling_rate = state.get("sampling_rate")
    print(f"video={getattr(video, 'shape', None)} audio={tuple(audio.shape)} sampling_rate={sampling_rate}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encode_video(video, fps=24, output_path=str(args.output), audio=audio, audio_sample_rate=sampling_rate)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
