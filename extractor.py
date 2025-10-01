# extractor.py
import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_fps(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip()
    num, den = out.split("/")
    fps = float(num) / float(den)
    log(f"FFprobe reports input FPS ≈ {fps:.6g}")
    return fps


def run_ffmpeg_extract_all(video_path: str, out_dir: str, target_fps: float | None):
    """
    One pass extraction to imgs/<video_name>/frame%05d.png .
    Uses -n, so pre-existing frames are skipped on disk.
    Streams progress lines to the console.
    """
    out_pattern = str(Path(out_dir) / "frame%05d.png")
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-stats",  # print running counters
        "-loglevel",
        "info",
        "-nostdin",
        "-n",  # don't overwrite: skip files that already exist
        "-i",
        video_path,
    ]
    if target_fps is not None:
        ffmpeg_cmd += ["-vf", f"fps={target_fps}"]
    ffmpeg_cmd += [
        "-vsync",
        "0",
        "-start_number",
        "0",
        out_pattern,
    ]

    log("Launching ffmpeg for one-pass frame extraction…")
    log(" ".join(ffmpeg_cmd))
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # Forward ffmpeg's stderr progress lines live
    for line in proc.stderr:
        line = line.rstrip()
        if line.startswith("frame=") or "time=" in line or "fps=" in line:
            print(line, flush=True)
    ret = proc.wait()
    if ret != 0:
        # Surface ffmpeg error output if it failed
        err = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"ffmpeg failed with code {ret}\n{err}")

    log("ffmpeg finished frame extraction.")


def get_clip_embeddings(device, model, preprocess, frame_path: str, npy_path: str):
    image = preprocess(Image.open(frame_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image).cpu().numpy()
    np.save(npy_path, image_features)


def main(config):
    # Resolve paths and names
    video_file = config.video
    ann_file = config.annotations
    video_dir = str(Path(video_file).parent)
    video_stem = Path(video_file).stem

    # Output directories
    imgs_dir = os.path.join(video_dir, "imgs", video_stem)
    img_embs_dir = os.path.join(video_dir, "img_embeddings", video_stem)

    log(f"Video: {video_file}")
    log(f"Annotations: {ann_file}")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(img_embs_dir, exist_ok=True)
    log(f"Ensured dirs:\n  - frames: {imgs_dir}\n  - embeddings: {img_embs_dir}")

    # Read basic metadata
    try:
        video_fps = get_fps(video_file)
    except Exception as e:
        log(f"⚠️ FFprobe failed to read FPS: {e}")
        video_fps = None

    # Load annotations (to know the last annotated frame; helpful for sanity checks)
    try:
        annotations = pd.read_csv(ann_file)
        last_frame = int(annotations.iloc[-1]["Last frame"])
        log(f"Last annotated frame: {last_frame}")
    except Exception as e:
        log(f"⚠️ Could not read annotations ({e}). Continuing without last-frame bound.")
        last_frame = None

    # ---- 1) FRAME EXTRACTION (one pass) ----
    target_fps = float(config.frames_per_second) if config.frames_per_second > 0 else None
    run_ffmpeg_extract_all(video_file, imgs_dir, target_fps)

    # Count frames on disk
    frame_paths = sorted(Path(imgs_dir).glob("frame*.png"))
    if last_frame is not None and len(frame_paths) == 0:
        log("❌ No frames were written. Please check ffmpeg output above.")
        sys.exit(1)

    log(f"Frames on disk: {len(frame_paths)}")

    # ---- 2) CLIP EMBEDDINGS ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading CLIP (ViT-B/32) on device: {device}")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    log("CLIP loaded.")

    # Iterate frames -> embeddings (skip those already computed)
    log("Starting embedding computation…")
    done = 0
    skipped = 0
    for i, frame_path in enumerate(frame_paths):
        idx_str = frame_path.stem.replace("frame", "")
        npy_path = Path(img_embs_dir) / f"frame{idx_str}.npy"

        if npy_path.exists():
            skipped += 1
            if skipped % 1000 == 0:
                log(f"Skipped {skipped} embeddings already on disk…")
            continue

        try:
            get_clip_embeddings(device, clip_model, preprocess, str(frame_path), str(npy_path))
            done += 1
            if (done % 100 == 0) or (i == len(frame_paths) - 1):
                log(f"Embeddings: {done} written | {skipped} skipped | {i+1}/{len(frame_paths)} frames processed")
        except Exception as e:
            log(f"⚠️ Failed embedding for {frame_path.name}: {e}")

    log("All done ✅")
    log(f"Summary → Frames: {len(frame_paths)} | Embeddings written: {done} | Skipped: {skipped}")
    if video_fps and target_fps:
        log(f"(Input FPS ≈ {video_fps:.3f}, target FPS = {target_fps:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--annotations", type=str, required=True, help="CSV annotations path")
    parser.add_argument(
        "--frames-per-second",
        type=float,
        default=0.0,
        help="If >0, resample with fps=VALUE. If 0, keep original FPS.",
    )
    cfg = parser.parse_args()
    main(cfg)
