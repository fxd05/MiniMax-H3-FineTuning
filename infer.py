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

Memory: the plain path here loads every component in bf16 — the 33B
transformer (~63 GB), the Qwen3-VL-32B text encoder (~63 GB) and the VAEs
(~10 GB): ~135 GB of weights alone, so even a 90 GB card OOMs (verified).
`--offload` (default) is the official single-card recipe for this commit:
components are registered with a ComponentsManager that moves them on and
off the accelerator on demand, so the weights live in host RAM (~135 GB)
and one GPU only ever holds the active component.

Canvas: MiniMax-H3 was *released for a 768 pixel short edge only* (soft area
cap 768x1344, see diffusers.modular_pipelines.minimax_h3.packing), so
leaving `--height`/`--width` unset resolves the pipeline's own native
canvas — 1344x768 for t2va's 16:9, the aspect of the keyframe for fl2va.
Smaller canvases are a speed lever, not the design point: 256x320 and even
the old 640x480 default are far below training resolution, and the model
visibly degrades there (blurry, unstable motion).

Attention: full self-attention over the packed sequence. `--attention-backend`
defaults to "auto": on Hopper (sm_90, e.g. H100) it tries the flash-3 hub
kernels (`_flash_3_hub`, ~3x faster, no score matrix) and falls back to
torch SDPA *and* a 640x480 canvas if the kernel fetch fails (e.g. no hub
access); without flash the per-head score matrix itself OOMs at the native
canvas, so pass an explicit backend/canvas combination then.

`--multi-gpu` is the docs two-card recipe, defined for t2va only: the
workflow is split into a conditioner (the text-encoder step, ~63 GB) and the
rest (setup / VAEs / layout / denoise / decode, ~73 GB), each half pinned to
its own card via its own ComponentsManager + enable_auto_cpu_offload — that
call is what attaches the input-dispatch hooks, so the halves must be run as
two pipeline calls exchanging a state object (conditioner first). fl2va and
ref2va don't compose this way (their text-encoder steps consume
keyframes/prepared_references that the setup step in the other half
produces) — run those with --offload.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from diffusers import ComponentsManager, MiniMaxH3ModularPipeline, MiniMaxH3Ref2VABlocks
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
from diffusers.modular_pipelines.minimax_h3.packing import align_num_frames, resolve_canvas_size
from diffusers.utils import encode_video


def _resolve_model_root(model: Path) -> Path:
    """Return a model root whose modular_model_index.json points at local paths.

    The modular index shipped with the checkpoint declares every component with
    pretrained_model_name_or_path="MiniMaxAI/MiniMax-H3" (the hub id), so
    from_pretrained(local_dir) still tries to download from the hub. That works
    on machines with a warm HF cache, but on this cluster the checkout lives in
    read-only /lpai with no cache, and hub access goes through a mirror that
    ices Xet downloads (404 on xet-read-token). Rewrite the index to absolute
    local paths in a writable mirror dir (runs/, gitignored) and use that as
    the root; weight shards are still read from the original subfolders.
    """
    idx = model / "modular_model_index.json"
    if not idx.exists():
        return model
    data = json.loads(idx.read_text())
    changed = False
    for entry in data.values():
        # component entries are [source, class, {spec, ...}]
        if not (isinstance(entry, list) and len(entry) == 3 and isinstance(entry[2], dict)):
            continue
        spec = entry[2]
        if spec.get("pretrained_model_name_or_path") not in (None, str(model)):
            spec["pretrained_model_name_or_path"] = str(model)
            changed = True
    if not changed:
        return model
    local_root = Path(__file__).resolve().parent / "runs" / "_local_model_index"
    local_root.mkdir(parents=True, exist_ok=True)
    (local_root / "modular_model_index.json").write_text(json.dumps(data, indent=2))
    print(f"localized modular index: {local_root} -> {model}")
    return local_root


def _ensure_processor_compat(pipe) -> None:
    """Restore Qwen3VLProcessor.create_mm_token_type_ids, dropped in transformers 5.0.0.

    The pinned diffusers encoders call it to tag text/image/video runs for the
    Qwen3-VL 3D-rotary positions. In 5.0.0 the model re-derives that layout from
    the vision pad token ids in input_ids instead (the tensor lands in **kwargs
    and is ignored), so any consistent tensor would do — keep the old 0/1/2
    semantics anyway.
    """
    proc = getattr(pipe, "processor", None)
    if proc is None or hasattr(proc, "create_mm_token_type_ids"):
        return
    image_pad = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_pad = proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    def create_mm_token_type_ids(sequences):
        return [[1 if t == image_pad else 2 if t == video_pad else 0 for t in seq] for seq in sequences]

    proc.create_mm_token_type_ids = create_mm_token_type_ids
    print("processor compat: Qwen3VLProcessor.create_mm_token_type_ids restored "
          "(transformers 5.0.0 dropped it)")


def build_pipe(model: Path, task: str, offload: bool = True,
               reserve_margin: str = "12GB"):
    model = _resolve_model_root(model)
    manager = ComponentsManager() if offload else None
    kwargs = {"components_manager": manager} if offload else {}
    if task == "ref2va":
        # Ref2VA registers the transformer as `transformer_ref`.
        pipe = MiniMaxH3Ref2VABlocks().init_pipeline(str(model), **kwargs)
    else:
        pipe = MiniMaxH3ModularPipeline.from_pretrained(str(model), **kwargs)
    pipe.load_components(dtype=torch.bfloat16)
    _ensure_processor_compat(pipe)
    if offload:
        manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin=reserve_margin)
    else:
        pipe.to("cuda")
    return pipe


def load_prompt(path: Path) -> str:
    """Extract a prompt from a file.

    JSON/JSONL files (by extension, or a first non-empty line that parses as
    JSON) yield the `prompt` field, falling back to `caption` — the manifest
    schema documented in examples/README.md. Any other file is read as plain
    text. Exactly one prompt must result.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"{path}: no such file") from None
    is_json = path.suffix.lower() in (".json", ".jsonl") or text.lstrip().startswith("{")
    if not is_json:
        prompt = text.strip()
        if not prompt:
            raise SystemExit(f"{path}: empty prompt file")
        return prompt
    prompts = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{i}: not valid JSON: {exc}") from exc
        prompt = obj.get("prompt", obj.get("caption"))
        if prompt is None:
            raise SystemExit(f"{path}:{i}: no `prompt` or `caption` field: {sorted(obj)}")
        prompts.append(prompt)
    if len(prompts) == 1:
        return prompts[0]
    if len(prompts) > 1:
        raise SystemExit(f"{path}: {len(prompts)} prompts found; keep exactly one line, "
                         "or use --prompt instead")
    raise SystemExit(f"{path}: no prompts found")


def _transformer(pipe):
    return getattr(pipe, "transformer_ref", None) or getattr(pipe, "transformer", None)


def pick_attention_backend(pipe, requested: str | None) -> str:
    """Set and return the effective attention backend.

    `requested` in (None, "auto") tries the Hopper flash-3 hub kernels
    (`_flash_3_hub`, the official single-card recipe) and falls back to torch
    SDPA if the GPU is not Hopper or the kernel fetch fails (no hub access).
    An explicit name is applied verbatim.
    """
    transformer = _transformer(pipe)
    if requested and requested != "auto":
        transformer.set_attention_backend(requested)
        print(f"attention backend: {requested}")
        return requested
    if torch.cuda.get_device_capability() >= (9, 0):
        try:
            transformer.set_attention_backend("_flash_3_hub")
            print("attention backend: _flash_3_hub (Hopper flash-3 kernels from the Hub)")
            return "_flash_3_hub"
        except Exception as exc:
            print(f"warning: _flash_3_hub unavailable ({exc}); falling back to torch SDPA",
                  file=sys.stderr)
    print("attention backend: torch SDPA")
    return "torch_sdpa"


def _build_pipe_multi_gpu(model: Path, task: str, main_device: str = "cuda:0",
                          text_encoder_device: str = "cuda:1"):
    """Docs two-card recipe (t2va): split the workflow, pin each half to its
    own card with its own ComponentsManager.

    A ComponentsManager attaches its dispatch hooks only when
    enable_auto_cpu_offload is called, so each half needs its own
    manager + call: the conditioner (the text-encoder step, ~63 GB) runs on
    `text_encoder_device`, the rest (setup / VAEs / layout / denoise / decode,
    ~73 GB) on `main_device`. The two halves exchange a state object:
    `state = conditioner(prompt=prompt)` then `pipe(state=state, ...)`.

    Only t2va composes this way — the fl2va/ref2va text-encoder steps consume
    keyframes/prepared_references produced by the setup step in the other half.
    """
    # _resolve_model_root rewrites the modular index to a local mirror dir; both
    # halves load their weights through that index (which points back at the
    # original checkout's subfolders).
    root = _resolve_model_root(model)
    workflow = MiniMaxH3ModularPipeline.from_pretrained(str(root)).blocks.get_workflow(task)

    text_manager = ComponentsManager()
    text_manager.enable_auto_cpu_offload(device=text_encoder_device)
    conditioner = workflow.sub_blocks.pop("text_encoder").init_pipeline(
        str(root), components_manager=text_manager)
    conditioner.load_components(dtype=torch.bfloat16)
    _ensure_processor_compat(conditioner)

    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device=main_device)
    pipe = workflow.init_pipeline(str(root), components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    _ensure_processor_compat(pipe)
    return conditioner, pipe


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
    ap.add_argument("--prompt", help="text prompt; mutually exclusive with --prompt-file")
    ap.add_argument("--prompt-file", type=Path, metavar="FILE",
                    help="read the prompt from a file: plain text, or JSON/JSONL with a `prompt`/"
                         "`caption` field (the manifest schema); mutually exclusive with --prompt")
    ap.add_argument("--image", type=Path, help="fl2va: first-frame keyframe")
    ap.add_argument("--last-image", type=Path, help="fl2va_last_frame: last-frame keyframe")
    ap.add_argument("--reference", action="append", metavar="TYPE=PATH",
                    help="ref2va: repeatable, e.g. image=a.jpg video=b.mp4 audio=c.wav (max 9i/3v/3a)")
    ap.add_argument("--num-frames", type=int, default=124, help="snapped up to 17*n+5; duration must stay 5-15 s")
    ap.add_argument("--height", type=int, default=480,
                    help="canvas height, a multiple of 32; default: MiniMax-H3's native 768-short-edge canvas "
                         "(1344x768 for t2va 16:9, the keyframe's aspect for fl2va)")
    ap.add_argument("--width", type=int, default=640,
                    help="canvas width, a multiple of 32; must be passed together with --height")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--offload", dest="offload", action="store_true", default=True,
                    help="auto CPU offload via ComponentsManager (default; ~135 GB of weights live in host RAM)")
    ap.add_argument("--no-offload", dest="offload", action="store_false",
                    help="load everything onto CUDA (needs ~135 GB free VRAM)")
    ap.add_argument("--multi-gpu", action="store_true",
                    help="docs 2-card recipe (t2va only): split the workflow, text-encoder "
                         "half on its own GPU, the rest on the main one; needs 2 x >=80 GB")
    ap.add_argument("--text-encoder-device", default="cuda:1",
                    help="GPU for the text-encoder half with --multi-gpu (default cuda:1)")
    ap.add_argument("--reserve-margin", default="12GB",
                    help="VRAM the offload manager keeps free (default 12GB)")
    ap.add_argument("--attention-backend", default="flash",
                    help="'auto' (default): try the Hopper flash-3 hub kernels (_flash_3_hub, ~3x faster, no "
                         "score matrix) and fall back to torch SDPA on a small canvas; or an explicit backend "
                         "name (e.g. _flash_3_hub, torch_sdpa)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-type", choices=("pil", "pt", "np", "latent"), default="pil")
    ap.add_argument("--output", type=Path, default=Path("runs/infer.mp4"))
    ap.add_argument("--checkpoint", type=Path, help="optional train.py heads checkpoint to patch in")
    args = ap.parse_args()

    if args.prompt and args.prompt_file:
        raise SystemExit("--prompt and --prompt-file are mutually exclusive")
    if args.prompt_file:
        args.prompt = load_prompt(args.prompt_file)
        print(f"prompt from {args.prompt_file}: {args.prompt!r}")

    if (args.height is None) != (args.width is None):
        raise SystemExit("--height and --width must be passed together, or neither (native canvas)")
    if args.height is not None and (args.height % 32 or args.width % 32):
        raise SystemExit("height/width must be multiples of 32")
    num_frames = align_num_frames(args.num_frames)
    if not 5 * 24 <= num_frames <= 15 * 24:
        print(f"warning: {num_frames} frames at 24 fps is outside 5-15 s", file=sys.stderr)

    conditioner = None
    if args.multi_gpu:
        if args.task != "t2va":
            raise SystemExit("--multi-gpu implements the docs two-card recipe, which is t2va only "
                             "(fl2va/ref2va text-encoding needs setup outputs from the other half); "
                             "run those tasks with --offload (the default)")
        if args.image or args.last_image or args.reference:
            raise SystemExit("--multi-gpu is t2va only: --image/--last-image/--reference are not supported")
        if not args.prompt:
            raise SystemExit("--multi-gpu t2va requires --prompt")
        conditioner, pipe = _build_pipe_multi_gpu(args.model, args.task,
                                                  text_encoder_device=args.text_encoder_device)
    else:
        pipe = build_pipe(args.model, args.task, offload=args.offload,
                          reserve_margin=args.reserve_margin)
    attention_backend = pick_attention_backend(pipe, args.attention_backend)
    if args.checkpoint:
        apply_checkpoint(pipe, args.checkpoint, args.task)

    # Canvas: leaving height/width unset resolves MiniMax-H3's own 768-short-edge
    # geometry (packing.resolve_canvas_size) — 1344x768 for t2va 16:9, the
    # keyframe's aspect for fl2va, 16:9 for ref2va. Below the native canvas the
    # model visibly degrades, so the default is the native one.
    if args.height is None:
        aspect = (16, 9)
        if args.image:
            aspect = Image.open(args.image).size
        elif args.last_image:
            aspect = Image.open(args.last_image).size
        args.height, args.width = resolve_canvas_size(*aspect)
        print(f"canvas: {args.width}x{args.height} (native 768-short-edge for {aspect[0]}:{aspect[1]})")

    if attention_backend == "torch_sdpa":
        # Full self-attention over the packed sequence; torch SDPA without a
        # flash kernel materializes the per-head score matrix (≈ tokens² × 2 B).
        tokens = (args.height // 16) * (args.width // 16) * num_frames + 4096
        score_gb = tokens * tokens * 2 / 1e9
        if score_gb > 4:
            if args.attention_backend in (None, "auto"):
                # Automatic fallback: shrink to the small-canvas default rather
                # than let the run OOM mid-denoise.
                args.height, args.width = 320, 256
                print(f"warning: torch SDPA at the native canvas needs ~{score_gb:.0f} GB of per-head attention "
                      f"scores ({tokens} packed tokens) and would OOM; shrinking to {args.width}x{args.height}. "
                      "For native-canvas quality use a Hopper GPU with hub access (auto picks "
                      "_flash_3_hub) or pass an explicit backend.",
                      file=sys.stderr)
            else:
                raise SystemExit(f"~{score_gb:.0f} GB of per-head attention scores at {args.width}x{args.height} "
                                 f"({tokens} packed tokens) without a flash backend; shrink --height/--width "
                                 "or use --attention-backend _flash_3_hub on Hopper")

    if conditioner is not None:
        # Docs split: the conditioner (text encoder) runs first on its card and
        # hands the prompt_embeds to the rest pipeline via the state object.
        state = conditioner(prompt=args.prompt)
        kwargs = dict(state=state, num_frames=num_frames, height=args.height, width=args.width,
                      num_inference_steps=args.steps,
                      generator=torch.Generator().manual_seed(args.seed),
                      output_type=args.output_type,
                      output=["videos", "audio", "sampling_rate"])
    else:
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
                if not path.startswith(("http://", "https://")) and not Path(path).is_file():
                    raise SystemExit(f"--reference {kind}={path}: no such file "
                                     f"(resolved: {Path(path).resolve()}); relative paths are "
                                     "checked against the current working directory")
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
