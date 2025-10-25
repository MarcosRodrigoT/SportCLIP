#!/usr/bin/env python3
"""
Plot timing analysis from timing_log.json

This script creates professional visualizations of computational costs
for the SportCLIP pipeline, with normalization by video duration.

HOW TIMINGS ARE COMPUTED:
==========================

1. Raw Timing Collection:
   - Each component in the pipeline (e.g., "Create class embeddings", "Compute similarities")
     is timed individually using Python's timing utilities
   - For multi_sentences.py: Timings are recorded for EACH sentence pair processed
   - For summarize.py: Timings are recorded once per video
   - All timings include CPU usage and memory consumption

2. Normalization:
   - Raw timings vary based on video length (longer videos take more time to process)
   - To enable fair comparison, timings are normalized to "seconds per minute of video"
   - Formula: normalized_time = (raw_time / video_duration_seconds) x 60
   - Video duration and FPS are extracted automatically using ffprobe

3. Aggregation:
   - For each video with multiple sentence sets (e.g., diving with sets 6-10):
     * Timings across all sentence pairs are AVERAGED
     * This gives you the "average time per sentence pair" for that video
   - Across videos: Mean and standard deviation are computed for comparison

4. Grouping:
   - Components are grouped into categories (e.g., "Embeddings & Similarity",
     "Post-Processing") for high-level visualization

Example:
--------
If "Create class embeddings" took:
  - Video A (3 min): 1.5s total for 64 pairs = ~0.023s per pair
  - Video B (5 min): 2.5s total for 64 pairs = ~0.039s per pair

Normalized (per minute of video):
  - Video A: (1.5 / 180) x 60 = 0.5 s/min
  - Video B: (2.5 / 300) x 60 = 0.5 s/min

Both videos have the same normalized processing time despite different raw times!
"""

import argparse
import json
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# ============================================================================
# Configuration
# ============================================================================

# Cache for video metadata (populated automatically)
VIDEO_METADATA = {}

# Default data directory (can be overridden via command line)
DEFAULT_DATA_DIR = "/mnt/Data/mrt/MATDAT"

# Component grouping for better visualization
COMPONENT_GROUPS = {
    "Data Loading": [
        "multi_sentences.py - Setup",
        "Load ground truth annotations",
        "Load highlight scores from filtered pairs",
    ],
    "Embeddings & Similarity": [
        "Create class embeddings (sentences)",
        "Collect predictions (compute similarities)",
    ],
    "Post-Processing": [
        "Convert predictions to lists",
        "Convert ground truth to list",
        "Compute rolling averages",
        "Compute coarse predictions",
        "Closing operation (dilate/erode)",
        "Compute prediction areas",
    ],
    "Event Detection": [
        "Detect events",
        "Filter events by duration",
        "Filter events by area",
    ],
    "Metrics & Analysis": [
        "Compute event statistics",
        "Compute frame-level results",
        "Compute final frame-level results",
        "Compute ablation metrics (frame & event level)",
        "Compute intermediate variables (histogram stats)",
        "Filter pairs by histogram and area",
        "Compute median predictions",
    ],
    "Visualization": [
        "Plot predictions vs ground truth",
        "Draw histograms",
        "Compute crossing point",
        "Stitch images",
        "Plot final predictions vs ground truth",
    ],
    "I/O Operations": [
        "Save logs",
        "Save pairs used",
        "Export highlight reel video",
    ],
}

# Seaborn style configuration
STYLE_CONFIG = {
    "style": "whitegrid",
    "palette": "husl",
    "context": "paper",
    "font_scale": 1.2,
}


# ============================================================================
# Video Metadata Extraction
# ============================================================================


def get_video_metadata_ffprobe(video_path: str) -> Optional[Dict]:
    """
    Extract video metadata (duration, fps, frame count) using ffprobe.
    Returns None if ffprobe fails.
    """
    try:
        # Run ffprobe to get video metadata
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            return None

        # Extract metadata
        duration = float(data.get("format", {}).get("duration", 0))
        fps_str = video_stream.get("r_frame_rate", "30/1")
        num, denom = map(int, fps_str.split("/"))
        fps = num / denom if denom != 0 else 30.0
        frame_count = int(video_stream.get("nb_frames", 0))

        # If frame count not available, estimate from duration and fps
        if frame_count == 0 and duration > 0:
            frame_count = int(duration * fps)

        return {
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
        }

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        print(f"Warning: ffprobe failed for {video_path}: {e}")
        return None


def find_video_file(video_name: str, data_dir: str) -> Optional[str]:
    """
    Find the video file for a given video name.
    Searches for common video extensions in the data directory.
    """
    extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg"]

    for ext in extensions:
        video_path = os.path.join(data_dir, f"{video_name}{ext}")
        if os.path.exists(video_path):
            return video_path

    return None


def get_video_metadata(video_name: str, data_dir: str) -> Dict:
    """
    Get video metadata (duration, fps, frames).
    Uses ffprobe to extract metadata. Raises error if fails.
    """
    # Check cache first
    if video_name in VIDEO_METADATA:
        return VIDEO_METADATA[video_name]

    # Try to find video file
    video_path = find_video_file(video_name, data_dir)

    if not video_path:
        raise FileNotFoundError(f"Could not find video file for '{video_name}' in '{data_dir}'. Searched for extensions: .mp4, .avi, .mov, .mkv, .webm, .mpeg")

    # Try to probe video file
    metadata = get_video_metadata_ffprobe(video_path)

    if not metadata:
        raise RuntimeError(f"Failed to extract metadata from video '{video_path}' using ffprobe. Ensure ffprobe is installed and the video file is valid.")

    # Cache and return metadata
    VIDEO_METADATA[video_name] = metadata
    print(f"Loaded metadata for '{video_name}': {metadata['duration']:.1f}s @ {metadata['fps']:.2f}fps ({metadata['frame_count']} frames)")
    return metadata


# ============================================================================
# Data Loading & Processing
# ============================================================================


def load_timing_data(log_file: str) -> List[Dict]:
    """Load timing data from JSON file."""
    with open(log_file, "r") as f:
        data = json.load(f)
    return data


def normalize_video_name(video_name: str) -> str:
    """Normalize video name for consistent grouping."""
    # Handle tricking videos (V1, V2, V3)
    if video_name in ["V1", "V2", "V3"]:
        return video_name
    # For other sports, remove any suffixes
    return video_name.split("_")[0] if "_" in video_name else video_name


def assign_component_group(component_name: str) -> str:
    """Assign a component to its group."""
    for group_name, components in COMPONENT_GROUPS.items():
        if component_name in components:
            return group_name
    return "Other"


def process_timing_data(timing_data: List[Dict], data_dir: str) -> pd.DataFrame:
    """Process raw timing data into a structured DataFrame."""
    records = []

    # Get unique video names first
    unique_videos = set(entry.get("video_name", "unknown") for entry in timing_data)

    # Load metadata for all videos
    print("\nLoading video metadata...")
    for video_name in unique_videos:
        if video_name != "unknown":
            get_video_metadata(video_name, data_dir)

    # Process timing entries
    for entry in timing_data:
        video_name = entry.get("video_name", "unknown")
        sport = entry.get("sport", "unknown")
        component = entry["component"]
        duration = entry["duration_seconds"]

        # Get video metadata
        metadata = get_video_metadata(video_name, data_dir)
        video_duration = metadata["duration"]

        # Normalize timing: time per minute of video
        normalized_duration = (duration / video_duration) * 60.0 if video_duration > 0 else 0

        records.append(
            {
                "video_name": video_name,
                "sport": sport,
                "component": component,
                "duration_seconds": duration,
                "normalized_duration": normalized_duration,  # seconds per minute of video
                "video_duration": video_duration,
                "video_fps": metadata["fps"],
                "video_frames": metadata["frame_count"],
                "group": assign_component_group(component),
            }
        )

    return pd.DataFrame(records)


def aggregate_by_video(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate timings by video and component (average across sentence sets)."""
    agg_df = (
        df.groupby(["video_name", "component", "group"])
        .agg(
            {
                "duration_seconds": "mean",
                "normalized_duration": "mean",
                "video_duration": "first",
            }
        )
        .reset_index()
    )

    return agg_df


def aggregate_across_videos(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate timings across all videos by component."""
    agg_df = (
        df.groupby(["component", "group"])
        .agg(
            {
                "duration_seconds": ["mean", "std"],
                "normalized_duration": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Flatten column names
    agg_df.columns = ["component", "group", "duration_mean", "duration_std", "normalized_mean", "normalized_std"]

    return agg_df


# ============================================================================
# Plotting Functions
# ============================================================================


def setup_style():
    """Set up seaborn style for beautiful plots."""
    sns.set_style(STYLE_CONFIG["style"])
    sns.set_context(STYLE_CONFIG["context"], font_scale=STYLE_CONFIG["font_scale"])
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def plot_per_video_timings(df: pd.DataFrame, output_file: str):
    """
    Create a stacked bar chart showing normalized timings per video,
    grouped by component category.
    """
    setup_style()

    # Aggregate by video and group
    video_group_df = df.groupby(["video_name", "group"]).agg({"normalized_duration": "sum"}).reset_index()

    # Pivot for stacked bar chart
    pivot_df = video_group_df.pivot(index="video_name", columns="group", values="normalized_duration").fillna(0)

    # Sort videos alphabetically (case-insensitive: diving, long_jump, pole_vault, tumbling, V1, V2, V3)
    pivot_df = pivot_df.sort_index(key=lambda x: x.str.lower())

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create stacked bar chart
    pivot_df.plot(kind="bar", stacked=True, ax=ax, colormap="Set3", edgecolor="black", linewidth=0.5)

    # Styling
    ax.set_xlabel("Video", fontsize=14, fontweight="bold")
    ax.set_ylabel("Processing Time (seconds per minute of video)", fontsize=14, fontweight="bold")
    ax.set_title("Processing Time by Component Group (Normalized per Video)\nAverage across sentence sets", fontsize=16, fontweight="bold", pad=20)
    ax.legend(title="Component Group", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_average_component_timings(df: pd.DataFrame, output_file: str):
    """
    Create a horizontal bar chart showing average normalized timings
    for each component across all videos, with error bars.
    """
    setup_style()

    # Aggregate across videos
    agg_df = aggregate_across_videos(df)

    # Sort by mean time
    agg_df = agg_df.sort_values("normalized_mean", ascending=True)

    # Filter out very small components (< 0.01s per minute)
    agg_df = agg_df[agg_df["normalized_mean"] > 0.01]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create color mapping by group
    groups = agg_df["group"].unique()
    palette = sns.color_palette(STYLE_CONFIG["palette"], n_colors=len(groups))
    group_colors = {group: palette[i] for i, group in enumerate(groups)}
    colors = [group_colors[group] for group in agg_df["group"]]

    # Create horizontal bar chart with error bars
    bars = ax.barh(range(len(agg_df)), agg_df["normalized_mean"], xerr=agg_df["normalized_std"], color=colors, edgecolor="black", linewidth=0.5, capsize=3)

    # Styling
    ax.set_yticks(range(len(agg_df)))
    ax.set_yticklabels(agg_df["component"], fontsize=10)
    ax.set_xlabel("Processing Time (seconds per minute of video)", fontsize=14, fontweight="bold")
    ax.set_title("Average Processing Time by Component\nNormalized across all videos (mean ± std)", fontsize=16, fontweight="bold", pad=20)
    ax.grid(axis="x", alpha=0.3)

    # Add legend for groups
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=group_colors[group], edgecolor="black", label=group) for group in groups]
    ax.legend(handles=legend_elements, title="Component Group", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_group_pie_chart(df: pd.DataFrame, output_file: str):
    """
    Create a pie chart showing the proportion of time spent in each
    component group, averaged across all videos.
    """
    setup_style()

    # Aggregate by group
    group_df = df.groupby("group").agg({"normalized_duration": "sum"}).reset_index()

    # Sort by time
    group_df = group_df.sort_values("normalized_duration", ascending=False)

    # Create figure with more space for legend
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create pie chart - only show labels for slices > 3%
    colors = sns.color_palette(STYLE_CONFIG["palette"], n_colors=len(group_df))
    total = group_df["normalized_duration"].sum()

    def label_func(pct, group_name):
        # Only show label if slice is > 3%
        if pct > 3.0:
            return group_name
        return ""

    def autopct_func(pct):
        # Only show percentage if slice is > 3%
        if pct > 3.0:
            return f"{pct:.1f}%"
        return ""

    wedges, texts, autotexts = ax.pie(
        group_df["normalized_duration"],
        labels=[label_func(100 * val / total, name) for val, name in zip(group_df["normalized_duration"], group_df["group"])],
        autopct=autopct_func,
        startangle=90,
        colors=colors,
        textprops={"fontsize": 12, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.85,
    )

    # Styling
    ax.set_title("Distribution of Processing Time by Component Group\nNormalized average across all videos", fontsize=16, fontweight="bold", pad=20)

    # Add legend with all groups and their percentages
    legend_labels = [f"{row['group']}: {100*row['normalized_duration']/total:.1f}% ({row['normalized_duration']:.1f}s)" for _, row in group_df.iterrows()]
    ax.legend(legend_labels, title="Component Groups", bbox_to_anchor=(1.3, 1), loc="upper left", frameon=True, fancybox=True, shadow=True, fontsize=11)

    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_top_components_comparison(df: pd.DataFrame, output_file: str, top_n: int = 10):
    """
    Create a grouped bar chart comparing the top N most expensive components
    across different videos.
    """
    setup_style()

    # Aggregate by video
    video_df = aggregate_by_video(df)

    # Find top N components by average normalized duration
    top_components = video_df.groupby("component")["normalized_duration"].mean().nlargest(top_n).index.tolist()

    # Filter to top components
    filtered_df = video_df[video_df["component"].isin(top_components)]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Create grouped bar chart
    pivot_df = filtered_df.pivot(index="component", columns="video_name", values="normalized_duration").fillna(0)

    # Sort columns (videos) alphabetically (case-insensitive) for consistent legend ordering
    pivot_df = pivot_df[sorted(pivot_df.columns, key=str.lower)]

    pivot_df.plot(kind="bar", ax=ax, colormap="tab20", edgecolor="black", linewidth=0.5, width=0.8)

    # Styling
    ax.set_xlabel("Component", fontsize=14, fontweight="bold")
    ax.set_ylabel("Processing Time (seconds per minute of video)", fontsize=14, fontweight="bold")
    ax.set_title(f"Top {top_n} Most Expensive Components by Video\nNormalized per video duration", fontsize=16, fontweight="bold", pad=20)
    ax.legend(title="Video", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True, fancybox=True, shadow=True, ncol=1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Create professional timing analysis plots for SportCLIP pipeline")
    parser.add_argument("timing_log", type=str, help="Path to timing_log.json file")
    parser.add_argument("--output-dir", type=str, default="timing_plots", help="Output directory for plots (default: timing_plots)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help=f"Data directory containing video files (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top components to show in comparison plot (default: 10)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and process data
    print("Loading timing data...")
    timing_data = load_timing_data(args.timing_log)
    df = process_timing_data(timing_data, args.data_dir)

    print(f"\nLoaded {len(timing_data)} timing entries")
    print(f"Found {df['video_name'].nunique()} unique videos")
    print(f"Found {df['component'].nunique()} unique components")

    # Aggregate by video (average across sentence sets)
    video_df = aggregate_by_video(df)

    # Generate plots
    print("\nGenerating plots...")

    plot_per_video_timings(video_df, os.path.join(args.output_dir, "timing_per_video.png"))

    plot_average_component_timings(video_df, os.path.join(args.output_dir, "timing_average_components.png"))

    plot_group_pie_chart(video_df, os.path.join(args.output_dir, "timing_group_distribution.png"))

    plot_top_components_comparison(video_df, os.path.join(args.output_dir, "timing_top_components_comparison.png"), top_n=args.top_n)

    print(f"\nAll plots saved to: {args.output_dir}/")
    print("\nGenerated plots:")
    print("  1. timing_per_video.png - Stacked bar chart by video")
    print("  2. timing_average_components.png - Average component timings with error bars")
    print("  3. timing_group_distribution.png - Pie chart of component group distribution")
    print("  4. timing_top_components_comparison.png - Top components across videos")


if __name__ == "__main__":
    main()
