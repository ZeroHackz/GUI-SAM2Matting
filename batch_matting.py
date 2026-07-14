"""Batch matting with SAM2Matting.

Takes a directory of frames, a single image, or a video file, and writes
three output folders with matching filenames:

    <output>/alpha/        - grayscale alpha matte (white = foreground)
    <output>/composite/    - foreground composited over a solid background
    <output>/transparent/  - RGBA PNG with the alpha applied

The prompt mask for the first frame is generated automatically with rembg
(u2net). For multi-frame inputs the mask is propagated through the video
predictor, which keeps the matte temporally consistent.

Usage:
    python batch_matting.py --input C:\\path\\to\\frames_dir
    python batch_matting.py --input C:\\path\\to\\video.mp4
    python batch_matting.py --input C:\\path\\to\\image.png --output D:\\out
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)  # hydra config paths in build_sam are repo-relative

def log(msg: str):
    print(msg, flush=True)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

HF_BASE = "https://huggingface.co/FudanCVL/SAM2Matting/resolve/main/checkpoints"

VARIANTS = {
    "sam2.1tiny": {
        "checkpoint": "checkpoints/SAM2Matting-SAM2.1Tiny.pt",
        "cfg": "configs/sam2matting-sam2.1tiny.yaml",
        "mask_size": 256,
        "url": f"{HF_BASE}/SAM2Matting-SAM2.1Tiny.pt",
        "size_mb": 206,
    },
    "sam2.1base+": {
        "checkpoint": "checkpoints/SAM2Matting-SAM2.1Base+.pt",
        "cfg": "configs/sam2matting-sam2.1base+.yaml",
        "mask_size": 256,
        "url": f"{HF_BASE}/SAM2Matting-SAM2.1Base%2B.pt",
        "size_mb": 366,
    },
    "sam3": {
        "checkpoint": "checkpoints/SAM2Matting-SAM3.pt",
        "cfg": None,  # sam3 builds without a hydra config
        "mask_size": 288,
        "url": f"{HF_BASE}/SAM2Matting-SAM3.pt",
        "size_mb": 3347,
    },
}


def ensure_checkpoint(variant: str):
    """Download the variant's checkpoint from Hugging Face if it's missing."""
    import urllib.request

    v = VARIANTS[variant]
    path = REPO_ROOT / v["checkpoint"]
    if path.exists():
        return
    log(f"Checkpoint for {variant} not found - downloading ~{v['size_mb']} MB from Hugging Face...")
    req = urllib.request.Request(v["url"])
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(req) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done, next_pct = 0, 10
            while chunk := r.read(1 << 22):
                f.write(chunk)
                done += len(chunk)
                if total and done * 100 // total >= next_pct:
                    log(f"      {done // (1 << 20)} / {total // (1 << 20)} MB")
                    next_pct += 10
        part.rename(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    log("      checkpoint downloaded")


def load_sam3_state_dict(checkpoint: str):
    """Extract the tracker weights sam3 matting needs (mirrors the repo's inference scripts)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    out = {}
    for k, v in ckpt["model"].items():
        if k.startswith("detector.backbone.vision_backbone."):
            out[k.removeprefix("detector.")] = v
        elif k.startswith("tracker."):
            out[k.removeprefix("tracker.")] = v
    return out


def make_first_mask(image_path: Path) -> np.ndarray:
    """Generate a rough binary foreground mask (uint8 0/255) with rembg."""
    from rembg import new_session, remove

    session = new_session("u2net")
    with Image.open(image_path) as im:
        result = remove(im.convert("RGB"), session=session, only_mask=True)
    mask = np.array(result.convert("L"))
    return ((mask > 25) * 255).astype(np.uint8)


def mask_to_inputs(mask_np: np.ndarray, size: int = 256):
    """Convert a uint8 mask to the (raw_mask, mask_input) pair the predictor expects."""
    raw_mask = (torch.from_numpy(mask_np) / 255) > 0
    mask_input = (torch.from_numpy(mask_np) > 0).float() * 20 - 10
    mask_input = mask_input.unsqueeze(0).unsqueeze(0)
    mask_input = torch.nn.functional.interpolate(
        mask_input, size=(size, size), mode="bilinear", align_corners=False
    )
    return raw_mask, mask_input


def save_outputs(out_dirs, stem: str, original: Image.Image, alpha01: np.ndarray, bg_rgb):
    """Write alpha / composite / transparent for one frame."""
    alpha_u8 = (alpha01 * 255).clip(0, 255).astype(np.uint8)
    rgb = np.array(original.convert("RGB"))

    Image.fromarray(alpha_u8, mode="L").save(out_dirs["alpha"] / f"{stem}.png")

    a = alpha01[..., None]
    bg = np.full_like(rgb, bg_rgb, dtype=np.uint8)
    comp = (rgb * a + bg * (1.0 - a)).astype(np.uint8)
    Image.fromarray(comp).save(out_dirs["composite"] / f"{stem}.png")

    rgba = np.dstack([rgb, alpha_u8])
    Image.fromarray(rgba, mode="RGBA").save(out_dirs["transparent"] / f"{stem}.png")


def collect_frames(input_path: Path, staging_dir: Path):
    """Return ordered list of (numeric_staged_path, original_stem).

    Multi-frame inputs are staged under pure-numeric names because the SAM2
    frame loader sorts by int(filename). Videos are decoded with OpenCV.
    """
    frames = []
    if input_path.is_dir():
        files = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            sys.exit(f"No images found in {input_path}")
        for i, src in enumerate(files):
            staged = staging_dir / f"{i:06d}{src.suffix.lower()}"
            try:
                os.link(src, staged)
            except OSError:
                shutil.copy2(src, staged)
            frames.append((staged, src.stem))
    elif input_path.suffix.lower() in VIDEO_EXTS:
        import cv2

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            sys.exit(f"Could not open video {input_path}")
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            staged = staging_dir / f"{i:06d}.jpg"
            cv2.imwrite(str(staged), frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
            frames.append((staged, f"{i:06d}"))
            i += 1
        cap.release()
        if not frames:
            sys.exit(f"No frames decoded from {input_path}")
    else:
        frames.append((input_path, input_path.stem))
    return frames


def alpha_to01(alpha) -> np.ndarray:
    if isinstance(alpha, torch.Tensor):
        return alpha.detach().cpu().squeeze().float().numpy().clip(0, 1)
    return np.asarray(alpha, dtype=np.float32).squeeze().clip(0, 1)


def run_single_image(image_path: Path, stem: str, mask_np, variant, out_dirs, bg_rgb):
    checkpoint = str(REPO_ROOT / VARIANTS[variant]["checkpoint"])
    log(f"Loading {variant} image predictor and checkpoint...")
    if variant == "sam3":
        from sam3.model.build_sam3matting import build_sam3matting
        from sam3.model.sam3matting_image_predictor import SAM3MattingImagePredictor

        model = build_sam3matting(checkpoint=None)
        model.load_state_dict(load_sam3_state_dict(checkpoint), strict=False)
        predictor = SAM3MattingImagePredictor(model)
    else:
        from sam2.build_sam import build_sam2matting
        from sam2.sam2matting_image_predictor import SAM2MattingImagePredictor

        predictor = SAM2MattingImagePredictor(
            build_sam2matting(VARIANTS[variant]["cfg"], checkpoint))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        image = Image.open(image_path).convert("RGB")
        img = predictor.set_image(image)
        raw_mask, mask_input = mask_to_inputs(mask_np, VARIANTS[variant]["mask_size"])
        result = predictor.predict(
            img=img, raw_mask=raw_mask, mask_input=mask_input, multimask_output=False
        )
    save_outputs(out_dirs, stem, image, alpha_to01(result[1]), bg_rgb)


def run_video(frames, mask_np, variant, out_dirs, bg_rgb, staging_dir, progress=False):
    from tqdm import tqdm

    checkpoint = str(REPO_ROOT / VARIANTS[variant]["checkpoint"])
    log(f"Loading {variant} video predictor and checkpoint...")
    if variant == "sam3":
        from sam3.model.sam3matting_video_predictor import build_sam3matting_video_predictor

        predictor = build_sam3matting_video_predictor(checkpoint=None, device="cuda")
        predictor.load_state_dict(load_sam3_state_dict(checkpoint), strict=False)
    else:
        from sam2.build_sam import build_sam2matting_video_predictor

        predictor = build_sam2matting_video_predictor(
            VARIANTS[variant]["cfg"], checkpoint, device="cuda")
    _, mask_input = mask_to_inputs(mask_np, VARIANTS[variant]["mask_size"])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        log("Loading frames into inference state (this can take a while for long clips)...")
        state = predictor.init_state(
            video_path=str(staging_dir), offload_video_to_cpu=True
        )
        predictor.add_new_mask(
            inference_state=state, frame_idx=0, obj_id=1, mask=mask_input.to("cuda")
        )
        log(f"Propagating matte through {len(frames)} frames...")
        iterator = predictor.propagate_in_video(state)
        if not progress:
            iterator = tqdm(iterator, total=len(frames), desc="Matting")
        done = 0
        for frame_idx, _, _, alpha, _ in iterator:
            staged_path, stem = frames[frame_idx]
            alpha01 = alpha_to01(alpha)
            with Image.open(staged_path) as original:
                save_outputs(out_dirs, stem, original, alpha01, bg_rgb)
            done += 1
            if progress:
                print(f"PROGRESS {done}/{len(frames)} {stem}", flush=True)


def load_hf_token():
    """Export the HF token from .env_huggingface_access_token if present."""
    token_file = REPO_ROOT / ".env_huggingface_access_token"
    if token_file.exists() and "HF_TOKEN" not in os.environ:
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token


def main():
    ap = argparse.ArgumentParser(description="SAM2Matting batch runner")
    ap.add_argument("--input", required=True, help="Frames directory, video file, or single image")
    ap.add_argument("--output", default=None, help="Output root (default: <input>_matting)")
    ap.add_argument("--variant", default="sam2.1base+", choices=list(VARIANTS))
    ap.add_argument("--bg", default="0,0,0", help="Composite background R,G,B (default black)")
    ap.add_argument("--mask", default=None, help="Optional first-frame mask PNG (skips rembg)")
    ap.add_argument("--progress", action="store_true",
                    help="Emit machine-readable PROGRESS/OUTPUT_ROOT lines (used by GUI.py)")
    args = ap.parse_args()

    load_hf_token()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        sys.exit(f"Input not found: {input_path}")

    ensure_checkpoint(args.variant)

    if args.output:
        out_root = Path(args.output).resolve()
    else:
        stamp = int(time.time())
        base = input_path.name if input_path.is_dir() else input_path.stem
        out_root = input_path.parent / f"{base}_{stamp}_matting"
    out_dirs = {name: out_root / name for name in ("alpha", "composite", "transparent")}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    bg_rgb = [int(v) for v in args.bg.split(",")]
    if len(bg_rgb) != 3:
        sys.exit("--bg must be R,G,B")

    log(f"Input:      {input_path}")
    log(f"Output:     {out_root}")
    log(f"Variant:    {args.variant}  |  composite bg: {tuple(bg_rgb)}")
    log(f"GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA - will fail'}")
    if args.progress:
        print(f"OUTPUT_ROOT {out_root}", flush=True)

    try:
        process(input_path, args, out_dirs, bg_rgb)
    except BaseException:
        # don't leave empty output folders behind on failure
        for d in out_dirs.values():
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        if out_root.exists() and not any(out_root.iterdir()):
            out_root.rmdir()
        raise

    log(f"[4/4] Done. Outputs in:\n  {out_dirs['alpha']}\n  {out_dirs['composite']}\n  {out_dirs['transparent']}")


def process(input_path, args, out_dirs, bg_rgb):
    with tempfile.TemporaryDirectory(prefix="sam2matting_") as tmp:
        staging_dir = Path(tmp)
        log("[1/4] Collecting frames...")
        frames = collect_frames(input_path, staging_dir)
        log(f"      {len(frames)} frame(s) found")

        first_frame = frames[0][0]
        if args.mask:
            log(f"[2/4] Using provided first-frame mask: {args.mask}")
            mask_np = np.array(Image.open(args.mask).convert("L"))
            mask_np = ((mask_np > 25) * 255).astype(np.uint8)
        else:
            log("[2/4] Generating first-frame mask (rembg u2net)...")
            mask_np = make_first_mask(first_frame)
        coverage = (mask_np > 0).mean() * 100
        log(f"      mask covers {coverage:.1f}% of frame")
        if mask_np.sum() == 0:
            sys.exit("First-frame mask is empty - no foreground detected")

        log("[3/4] Running SAM2Matting...")
        if len(frames) == 1:
            run_single_image(frames[0][0], frames[0][1], mask_np, args.variant, out_dirs, bg_rgb)
            if args.progress:
                print(f"PROGRESS 1/1 {frames[0][1]}", flush=True)
        else:
            run_video(frames, mask_np, args.variant, out_dirs, bg_rgb, staging_dir,
                      progress=args.progress)


if __name__ == "__main__":
    main()
