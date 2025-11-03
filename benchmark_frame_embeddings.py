#!/usr/bin/env python3
"""
Benchmark CLIP frame embedding extraction time across all videos.
Does not save embeddings or frames - purely for timing measurements.
Optimizes batch size for GPU memory.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm


# Video paths and frame counts
VIDEO_CONFIG = {
    "diving": {
        "path": "/mnt/Data/mrt/SportCLIP/diving.mp4",
        "frames": 8781,
    },
    "long_jump": {
        "path": "/mnt/Data/mrt/SportCLIP/long_jump.mp4",
        "frames": 4230,
    },
    "pole_vault": {
        "path": "/mnt/Data/mrt/SportCLIP/pole_vault.mp4",
        "frames": 4710,
    },
    "tumbling": {
        "path": "/mnt/Data/mrt/SportCLIP/tumbling.mp4",
        "frames": 18540,
    },
    "V1": {
        "path": "/mnt/Data/mrt/MATDAT/V1.mpeg",
        "frames": 37745,
    },
    "V2": {
        "path": "/mnt/Data/mrt/MATDAT/V2.mpeg",
        "frames": 39593,
    },
    "V3": {
        "path": "/mnt/Data/mrt/MATDAT/V3.mpeg",
        "frames": 18250,
    },
}


def find_optimal_batch_size(model, preprocess, device, test_image_path, safety_factor=0.85):
    """
    Binary search to find the largest batch size that fits in GPU memory.
    """
    print("Finding optimal batch size...")

    # Load a test image
    test_img = preprocess(Image.open(test_image_path))

    # Start with higher upper bound for high-memory GPUs
    max_batch = 8192
    min_batch = 1
    optimal_batch = 1

    while min_batch <= max_batch:
        mid_batch = (min_batch + max_batch) // 2

        try:
            # Clear cache
            torch.cuda.empty_cache()

            # Try to process a batch
            batch = test_img.unsqueeze(0).repeat(mid_batch, 1, 1, 1).to(device)

            with torch.no_grad():
                embeddings = model.encode_image(batch)
                # Force computation to complete
                _ = embeddings.cpu()

            # Success - try larger batch
            optimal_batch = mid_batch
            min_batch = mid_batch + 1
            print(f"  Batch size {mid_batch}: ✓")

            # Clean up
            del batch, embeddings
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                # Too large - try smaller
                max_batch = mid_batch - 1
                print(f"  Batch size {mid_batch}: ✗ (OOM)")
                torch.cuda.empty_cache()
            else:
                raise

    # Apply safety factor to account for other memory usage during actual processing
    final_batch = int(optimal_batch * safety_factor)

    print(f"Maximum batch size found: {optimal_batch}")
    print(f"Using batch size with safety factor ({safety_factor}): {final_batch}")
    return final_batch


def extract_frames_to_memory(video_path, num_frames, target_fps=None):
    """
    Extract frames from video directly to memory using ffmpeg pipe.
    Returns list of PIL Images.
    """
    import subprocess

    print(f"Extracting {num_frames} frames from {Path(video_path).name}...")

    # Build ffmpeg command to output raw RGB frames
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-f",
        "image2pipe",
        "-pix_fmt",
        "rgb24",
        "-vcodec",
        "rawvideo",
    ]

    if target_fps is not None:
        ffmpeg_cmd += ["-vf", f"fps={target_fps}"]

    ffmpeg_cmd += ["-"]

    # Get frame dimensions first
    probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path]

    dims = subprocess.check_output(probe_cmd).decode("utf-8").strip().split(",")
    width, height = int(dims[0]), int(dims[1])
    frame_size = width * height * 3  # RGB

    # Extract frames
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frames = []
    for _ in tqdm(range(num_frames), desc="Loading frames"):
        raw_frame = proc.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            break

        # Convert raw bytes to PIL Image
        frame_array = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
        img = Image.fromarray(frame_array)
        frames.append(img)

    proc.terminate()
    proc.wait()

    print(f"Loaded {len(frames)} frames into memory")
    return frames


def benchmark_video(video_name, video_info, model, preprocess, device, batch_size, num_runs=3, max_frames=None):
    """
    Benchmark CLIP embedding extraction for a single video.
    Returns timing statistics.
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking: {video_name}")
    print(f"{'='*80}")

    video_path = video_info["path"]
    num_frames = video_info["frames"]

    # Check if we should load all frames or sample
    if max_frames is not None and num_frames > max_frames:
        # For very long videos, sample frames
        sample_frames = max_frames
        print(f"Video has {num_frames} frames, sampling {sample_frames} for benchmark")
        # Extract evenly spaced frames
        frame_indices = np.linspace(0, num_frames - 1, sample_frames, dtype=int)
    else:
        sample_frames = num_frames
        frame_indices = None

    # Load frames into memory (to avoid I/O overhead in timing)
    # For this benchmark, we'll actually use existing frame files if available
    imgs_dir = f"/mnt/Data/mrt/SportCLIP/imgs/{video_name}"
    if not os.path.exists(imgs_dir):
        # Try alternative path
        imgs_dir = f"/mnt/Data/mrt/MATDAT/imgs/{video_name}"

    if not os.path.exists(imgs_dir):
        print(f"Warning: Frame directory not found: {imgs_dir}")
        print("Skipping this video")
        return None

    frame_paths = sorted(Path(imgs_dir).glob("frame*.png"))

    if len(frame_paths) == 0:
        print(f"Warning: No frames found in {imgs_dir}")
        return None

    # Sample frames if needed
    if frame_indices is not None:
        frame_paths = [frame_paths[i] for i in frame_indices if i < len(frame_paths)]

    actual_frames = len(frame_paths)
    print(f"Using {actual_frames} frames for benchmark")

    # Pre-load and preprocess all frames
    print("Pre-loading frames...")
    preprocessed_frames = []
    for frame_path in tqdm(frame_paths, desc="Preprocessing"):
        img = Image.open(frame_path)
        preprocessed_frames.append(preprocess(img))

    # Stack into tensor
    all_frames_tensor = torch.stack(preprocessed_frames)
    print(f"Preprocessed tensor shape: {all_frames_tensor.shape}")

    # Warm up GPU
    print("Warming up GPU...")
    with torch.no_grad():
        warmup_batch = all_frames_tensor[:batch_size].to(device)
        _ = model.encode_image(warmup_batch)
    torch.cuda.synchronize()

    # Run benchmark multiple times
    print(f"Running benchmark ({num_runs} runs)...")
    run_times = []

    for run in range(num_runs):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        start_time = time.time()

        # Process in batches
        num_batches = (actual_frames + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, actual_frames)

            batch = all_frames_tensor[start_idx:end_idx].to(device)

            with torch.no_grad():
                _ = model.encode_image(batch)

        torch.cuda.synchronize()
        end_time = time.time()

        elapsed = end_time - start_time
        run_times.append(elapsed)

        print(f"  Run {run + 1}: {elapsed:.4f} s ({actual_frames / elapsed:.2f} frames/s)")

    # Compute statistics
    mean_time = np.mean(run_times)
    std_time = np.std(run_times, ddof=1)
    mean_time_per_frame_ms = (mean_time / actual_frames) * 1000
    std_time_per_frame_ms = (std_time / actual_frames) * 1000

    return {
        "video_name": video_name,
        "num_frames": num_frames,
        "frames_tested": actual_frames,
        "batch_size": batch_size,
        "num_runs": num_runs,
        "mean_total_time": mean_time,
        "std_total_time": std_time,
        "mean_time_per_frame_ms": mean_time_per_frame_ms,
        "std_time_per_frame_ms": std_time_per_frame_ms,
        "throughput_fps": actual_frames / mean_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark CLIP frame embedding extraction")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use (default: 0)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto-detect if not specified)")
    parser.add_argument("--safety-factor", type=float, default=0.85, help="Safety factor for batch size (default: 0.85)")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of benchmark runs per video (default: 3)")
    parser.add_argument("--max-frames", type=int, default=10000, help="Maximum frames to process per video (default: None = all frames)")
    parser.add_argument("--output", type=str, default="paper_outputs/frame_embedding_benchmark.md", help="Output markdown file")
    args = parser.parse_args()

    # Set GPU device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(args.gpu)

    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.1f} GB")

    # Load CLIP model
    print("\nLoading CLIP (ViT-B/32)...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print("CLIP loaded")

    # Find optimal batch size if not specified
    if args.batch_size is None:
        # Find a test image
        test_img_path = None
        for video_name in VIDEO_CONFIG:
            imgs_dir = f"/mnt/Data/mrt/SportCLIP/imgs/{video_name}"
            if not os.path.exists(imgs_dir):
                imgs_dir = f"/mnt/Data/mrt/MATDAT/imgs/{video_name}"

            if os.path.exists(imgs_dir):
                frames = list(Path(imgs_dir).glob("frame*.png"))
                if frames:
                    test_img_path = str(frames[0])
                    break

        if test_img_path:
            batch_size = find_optimal_batch_size(model, preprocess, device, test_img_path, args.safety_factor)
        else:
            print("Warning: Could not find test image, using batch_size=512")
            batch_size = 512
    else:
        batch_size = args.batch_size

    print(f"\nUsing batch size: {batch_size}")

    # Benchmark all videos
    results = []
    for video_name in sorted(VIDEO_CONFIG.keys()):
        video_info = VIDEO_CONFIG[video_name]
        result = benchmark_video(video_name, video_info, model, preprocess, device, batch_size, args.num_runs, args.max_frames)
        if result:
            results.append(result)

    # Generate report
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")

    # Compute aggregate statistics
    all_times_per_frame = [r["mean_time_per_frame_ms"] for r in results]
    overall_mean = np.mean(all_times_per_frame)
    overall_std = np.std(all_times_per_frame, ddof=1)

    # Write markdown report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        f.write("# Frame Embedding Extraction Benchmark\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- **Model**: CLIP ViT-B/32\n")
        f.write(f"- **Device**: {torch.cuda.get_device_name(args.gpu)}\n")
        f.write(f"- **Batch size**: {batch_size}\n")
        f.write(f"- **Runs per video**: {args.num_runs}\n\n")

        f.write("---\n\n")
        f.write("## Summary Statistics\n\n")
        f.write("| Component | Mean | Std Dev | Unit |\n")
        f.write("|-----------|------|---------|------|\n")
        f.write(f"| Frame embedding extraction | {overall_mean:.4f} | {overall_std:.4f} | ms/frame |\n\n")
        f.write(f"*Batch size: {batch_size} frames/batch*\n\n")
        f.write(f"*Time per batch: {overall_mean * batch_size:.2f} ms (approx)*\n\n")

        f.write("---\n\n")
        f.write("## Per-Video Results\n\n")
        f.write("| Video | Frames | Time/Frame (ms) | Throughput (fps) |\n")
        f.write("|-------|--------|-----------------|------------------|\n")

        for r in results:
            f.write(f"| {r['video_name']} | {r['num_frames']:,} | {r['mean_time_per_frame_ms']:.4f} ± {r['std_time_per_frame_ms']:.4f} | {r['throughput_fps']:.1f} |\n")

        f.write("\n---\n\n")
        f.write("## Detailed Timing\n\n")
        f.write("| Video | Frames Tested | Total Time (s) | Runs |\n")
        f.write("|-------|---------------|----------------|------|\n")

        for r in results:
            f.write(f"| {r['video_name']} | {r['frames_tested']:,} | {r['mean_total_time']:.2f} ± {r['std_total_time']:.2f} | {r['num_runs']} |\n")

        f.write("\n---\n\n")
        f.write("## Notes\n\n")
        f.write(f"- Batch size was {'auto-detected' if args.batch_size is None else 'manually set'} at {batch_size}\n")
        if args.batch_size is None:
            f.write(f"- Safety factor applied: {args.safety_factor} (batch size = {args.safety_factor} × max batch size that fits)\n")
        f.write("- Times include GPU computation only (frames pre-loaded into memory)\n")
        f.write("- Each video was benchmarked with multiple runs to compute mean and standard deviation\n")
        if args.max_frames is not None:
            f.write(f"- Videos with more than {args.max_frames} frames were sampled to {args.max_frames} frames\n")
        else:
            f.write("- All available frames were processed for each video\n")

    print(f"\nResults written to: {args.output}")
    print(f"\nOverall: {overall_mean:.4f} ± {overall_std:.4f} ms/frame")
    print(f"Throughput: ~{1000/overall_mean:.1f} frames/second")


if __name__ == "__main__":
    main()
