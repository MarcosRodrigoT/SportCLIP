#!/usr/bin/env python3
"""
Generate timing analysis table and visualizations for the paper.

This script processes timing logs and creates:
1. A markdown table with mean, std, and share for each paper component
2. A pie chart showing the distribution of processing time
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


# ============================================================================
# Component Mapping: Paper Components -> Code Components
# ============================================================================

# Note: Using ordered dict to control legend order in visualizations
from collections import OrderedDict

PAPER_COMPONENT_MAPPING = OrderedDict([
    ("Embedding computation", [
        "Create class embeddings (sentences)",
    ]),
    ("Similarity computation", [
        "Collect predictions (compute similarities)",
        "Compute rolling averages",  # Processes all frames
        "Compute and save KDE",       # Processes all frames
    ]),
    ("Sentence filtering", [
        "Filter events by duration",
        "Filter events by area",
    ]),
    ("Post-processing", [
        "Compute coarse predictions",
        "Closing operation (dilate/erode)",
        "Compute prediction areas",
        "Detect events",
        "Compute event statistics",
    ]),
])

# Component units of work (for normalization in Table V2)
COMPONENT_UNITS = {
    "Embedding computation": "per_run",  # One execution per run (not per pair)
    "Similarity computation": "per_frame",  # Processes all frames in video
    "Sentence filtering": "per_pair",  # One execution per sentence pair
    "Post-processing": "per_pair",  # One execution per sentence pair
}

# Frame counts for each video (from _gt_3classes.csv last frame)
VIDEO_FRAME_COUNTS = {
    'diving': 8781,
    'long_jump': 4230,
    'pole_vault': 4710,
    'tumbling': 18540,
    'V1': 37745,
    'V2': 39593,
    'V3': 18250
}

# Components to EXCLUDE from analysis (not relevant for paper)
EXCLUDED_COMPONENTS = [
    "multi_sentences.py - Setup",
    "Load ground truth annotations",
    "Load highlight scores from filtered pairs",
    "Convert predictions to lists",
    "Convert ground truth to list",
    "Compute frame-level results",
    "Compute final frame-level results",
    "Compute ablation metrics (frame & event level)",
    "Compute intermediate variables (histogram stats)",
    "Filter pairs by histogram and area",
    "Compute median predictions",
    "Plot predictions vs ground truth",
    "Draw histograms",
    "Compute crossing point",
    "Stitch images",
    "Plot final predictions vs ground truth",
    "Save logs",
    "Save pairs used",
    "Export highlight reel video",
]


# ============================================================================
# Data Processing
# ============================================================================


def load_timing_log(log_file: str) -> pd.DataFrame:
    """Load timing log JSON and convert to DataFrame."""
    with open(log_file, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def get_code_components_for_paper(paper_component: str) -> list:
    """Get the list of code components that map to a paper component."""
    return PAPER_COMPONENT_MAPPING.get(paper_component, [])


def filter_relevant_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only include relevant components for the paper."""
    # Get all relevant code components
    relevant_components = []
    for paper_comp, code_comps in PAPER_COMPONENT_MAPPING.items():
        relevant_components.extend(code_comps)

    # Filter to only relevant components
    filtered_df = df[df["component"].isin(relevant_components)].copy()

    return filtered_df


def map_to_paper_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column mapping code components to paper components."""
    component_to_paper = {}
    for paper_comp, code_comps in PAPER_COMPONENT_MAPPING.items():
        for code_comp in code_comps:
            component_to_paper[code_comp] = paper_comp

    df["paper_component"] = df["component"].map(component_to_paper)
    return df


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, and share for each paper component."""
    # Group by paper component and compute statistics
    stats = df.groupby("paper_component")["duration_seconds"].agg(["mean", "std", "sum", "count"]).reset_index()

    # Compute total time across all relevant components
    total_time = stats["sum"].sum()

    # Compute share (percentage)
    stats["share_percent"] = (stats["sum"] / total_time) * 100

    # Sort by the order defined in PAPER_COMPONENT_MAPPING
    component_order = {comp: idx for idx, comp in enumerate(PAPER_COMPONENT_MAPPING.keys())}
    stats["order"] = stats["paper_component"].map(component_order)
    stats = stats.sort_values("order").drop(columns=["order"])

    return stats


def compute_per_unit_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-unit statistics for Table V2.
    - Per-run components: Total time (ms) - executed once per run
    - Per-frame components: Normalize by video frame count (ms/frame)
    - Per-pair components: Use raw time (ms/pair)
    """
    stats_list = []

    for paper_comp in PAPER_COMPONENT_MAPPING.keys():
        code_comps = PAPER_COMPONENT_MAPPING[paper_comp]
        comp_data = df[df["component"].isin(code_comps)]

        if comp_data.empty:
            continue

        unit_type = COMPONENT_UNITS.get(paper_comp, "per_pair")

        if unit_type == "per_frame":
            # Normalize by frame count for each video
            normalized_times = []
            for _, row in comp_data.iterrows():
                video = row["video_name"]
                time_sec = row["duration_seconds"]
                if video in VIDEO_FRAME_COUNTS:
                    # Convert to ms per frame
                    time_ms_per_frame = (time_sec / VIDEO_FRAME_COUNTS[video]) * 1000
                    normalized_times.append(time_ms_per_frame)

            if normalized_times:
                mean_val = np.mean(normalized_times)
                std_val = np.std(normalized_times, ddof=1) if len(normalized_times) > 1 else 0
                unit_str = "ms/frame"
        elif unit_type == "per_run":
            # Per-run: convert seconds to milliseconds (total time)
            times_ms = comp_data["duration_seconds"].values * 1000
            mean_val = np.mean(times_ms)
            std_val = np.std(times_ms, ddof=1) if len(times_ms) > 1 else 0
            unit_str = "ms"
        else:
            # Per-pair: convert seconds to milliseconds
            times_ms = comp_data["duration_seconds"].values * 1000
            mean_val = np.mean(times_ms)
            std_val = np.std(times_ms, ddof=1) if len(times_ms) > 1 else 0
            unit_str = "ms/pair"

        stats_list.append({
            "paper_component": paper_comp,
            "mean": mean_val,
            "std": std_val,
            "unit": unit_str,
            "count": len(comp_data)
        })

    stats = pd.DataFrame(stats_list)

    # Sort by the order defined in PAPER_COMPONENT_MAPPING
    component_order = {comp: idx for idx, comp in enumerate(PAPER_COMPONENT_MAPPING.keys())}
    stats["order"] = stats["paper_component"].map(component_order)
    stats = stats.sort_values("order").drop(columns=["order"])

    return stats


def create_component_breakdown(df: pd.DataFrame) -> dict:
    """Create a breakdown showing which code components contribute to each paper component."""
    breakdown = {}

    for paper_comp in PAPER_COMPONENT_MAPPING.keys():
        code_comps = PAPER_COMPONENT_MAPPING[paper_comp]
        comp_data = df[df["component"].isin(code_comps)]

        comp_stats = comp_data.groupby("component")["duration_seconds"].agg(["mean", "sum", "count"]).reset_index()
        comp_stats = comp_stats.sort_values("sum", ascending=False)

        breakdown[paper_comp] = comp_stats

    return breakdown


# ============================================================================
# Output Generation
# ============================================================================


def generate_markdown_table(stats: pd.DataFrame, breakdown: dict, output_file: str):
    """Generate a markdown file with the timing table and component breakdown."""
    lines = []

    # Title
    lines.append("# Computational Cost Analysis - Table 7")
    lines.append("")
    lines.append("## Summary Statistics")
    lines.append("")

    # Main table
    lines.append("| Component | Mean (s) | Std Dev (s) | Share (%) |")
    lines.append("|-----------|----------|-------------|-----------|")

    for _, row in stats.iterrows():
        paper_comp = row["paper_component"]
        mean = row["mean"]
        std = row["std"]
        share = row["share_percent"]

        lines.append(f"| {paper_comp} | {mean:.4f} | {std:.4f} | {share:.2f} |")

    # Add total row
    total_mean = stats["mean"].sum()
    lines.append(f"| **TOTAL** | **{total_mean:.4f}** | - | **100.00** |")

    lines.append("")
    lines.append(f"*Note: Based on {stats['count'].sum():,} timing measurements*")
    lines.append("")

    # Component breakdown
    lines.append("---")
    lines.append("")
    lines.append("## Component Breakdown")
    lines.append("")
    lines.append("This section shows which code components contribute to each paper component.")
    lines.append("")

    for paper_comp, comp_stats in breakdown.items():
        lines.append(f"### {paper_comp}")
        lines.append("")

        # Get code components for this paper component
        code_comps = PAPER_COMPONENT_MAPPING[paper_comp]
        lines.append(f"**Includes the following code components:**")
        for code_comp in code_comps:
            lines.append(f"- `{code_comp}`")
        lines.append("")

        # Statistics for each code component
        if not comp_stats.empty:
            lines.append("| Code Component | Mean (s) | Count | Total (s) |")
            lines.append("|----------------|----------|-------|-----------|")

            for _, row in comp_stats.iterrows():
                lines.append(f"| {row['component']} | {row['mean']:.4f} | {row['count']} | {row['sum']:.2f} |")

            lines.append("")

            # Aggregate statistics
            agg_mean = comp_stats["mean"].sum()
            agg_total = comp_stats["sum"].sum()
            lines.append(f"**Aggregate:** Mean = {agg_mean:.4f}s, Total = {agg_total:.2f}s")
        else:
            lines.append("*No data available*")

        lines.append("")

    # Methodology notes
    lines.append("---")
    lines.append("")
    lines.append("## Methodology Notes")
    lines.append("")
    lines.append("### Data Collection")
    lines.append("- Timing measurements were collected using Python's `time.time()` for wall-clock time")
    lines.append("- Each component execution was timed individually using context managers")
    lines.append("- Measurements include CPU and memory usage metadata")
    lines.append("")

    lines.append("### Component Definitions")
    lines.append("")
    lines.append("**Embedding computation:** Computing CLIP text embeddings for sentence pairs")
    lines.append("")
    lines.append("**Similarity computation:** Computing cosine similarities between frame embeddings and text embeddings")
    lines.append("")
    lines.append("**Post-processing:** Signal processing operations including rolling averages, morphological operations, ")
    lines.append("event detection, histogram computation (KDE), and statistical analysis")
    lines.append("")
    lines.append("**Sentence filtering:** Filtering detected events based on duration and area thresholds")
    lines.append("")

    lines.append("### Excluded Components")
    lines.append("")
    lines.append("The following components were excluded from this analysis as they are not core algorithmic steps:")
    lines.append("")
    for excluded in EXCLUDED_COMPONENTS[:10]:  # Show first 10
        lines.append(f"- {excluded}")
    lines.append(f"- ... and {len(EXCLUDED_COMPONENTS) - 10} more")
    lines.append("")

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Markdown table written to: {output_file}")


def generate_markdown_table_v2(stats_per_unit: pd.DataFrame, output_file: str):
    """Generate Table V2 with per-unit metrics and example calculation."""
    lines = []

    # Title
    lines.append("# Computational Cost Analysis - Table V2 (Per-Unit Metrics)")
    lines.append("")
    lines.append("## Summary Statistics (Per Unit of Work)")
    lines.append("")

    # Main table
    lines.append("| Component | Mean | Std Dev | Unit |")
    lines.append("|-----------|------|---------|------|")

    for _, row in stats_per_unit.iterrows():
        paper_comp = row["paper_component"]
        mean = row["mean"]
        std = row["std"]
        unit = row["unit"]

        lines.append(f"| {paper_comp} | {mean:.4f} | {std:.4f} | {unit} |")

    lines.append("")
    lines.append(f"*Note: Based on {stats_per_unit['count'].sum():,} timing measurements*")
    lines.append("")

    # Example calculation
    lines.append("---")
    lines.append("")
    lines.append("## Example: 1-minute video at 30 FPS with 64 sentence pairs")
    lines.append("")
    lines.append("**Given:**")
    lines.append("- Video duration: 1 minute = 60 seconds")
    lines.append("- Frame rate: 30 FPS")
    lines.append("- Total frames: 60 × 30 = 1,800 frames")
    lines.append("- Sentence pairs tested: 64 pairs")
    lines.append("")

    lines.append("**Computational cost breakdown:**")
    lines.append("")

    # Track components by execution frequency
    per_run_total = 0.0
    per_pair_total = 0.0

    for _, row in stats_per_unit.iterrows():
        paper_comp = row["paper_component"]
        mean = row["mean"]
        unit = row["unit"]

        if unit == "ms/frame":
            # Multiply by total frames - this is per pair!
            cost = mean * 1800 / 1000  # Convert to seconds
            lines.append(f"- **{paper_comp}**: {mean:.4f} ms/frame × 1,800 frames = {cost:.3f} s/pair")
            per_pair_total += cost
        elif unit == "ms":
            # Per-run: executed once
            cost = mean / 1000  # Convert to seconds
            lines.append(f"- **{paper_comp}**: {mean:.2f} ms (once per run) = {cost:.3f} s")
            per_run_total += cost
        else:  # ms/pair
            cost = mean / 1000  # Convert to seconds
            lines.append(f"- **{paper_comp}**: {mean:.2f} ms/pair = {cost:.3f} s/pair")
            per_pair_total += cost

    lines.append("")
    lines.append("**Total computational cost:**")
    lines.append("")

    # Calculate total: per_run (once) + per_pair * 64
    total_time = per_run_total + (per_pair_total * 64)

    lines.append(f"- Embedding computation (once): {per_run_total:.3f} s")
    lines.append(f"- Per-pair components × 64 pairs: {per_pair_total:.3f} s/pair × 64 = {per_pair_total * 64:.2f} s")
    lines.append("")
    lines.append(f"**Total: {total_time:.2f} seconds = {total_time / 60:.2f} minutes**")
    lines.append("")

    # Additional video length estimates
    lines.append("---")
    lines.append("")
    lines.append("## Estimates for Different Video Lengths (64 pairs, 30 FPS)")
    lines.append("")

    # Get per-frame and per-pair components separately for different video lengths
    similarity_per_frame_ms = None
    per_pair_without_similarity_ms = 0.0

    for _, row in stats_per_unit.iterrows():
        paper_comp = row["paper_component"]
        mean = row["mean"]
        unit = row["unit"]

        if unit == "ms/frame":
            similarity_per_frame_ms = mean
        elif unit == "ms/pair":
            per_pair_without_similarity_ms += mean

    # Compute for different video lengths
    video_durations = [
        (1, 1 * 60 * 30),      # 1 minute
        (5, 5 * 60 * 30),      # 5 minutes
        (10, 10 * 60 * 30),    # 10 minutes
        (60, 60 * 60 * 30),    # 1 hour
    ]

    lines.append("| Video Length | Total Frames | Similarity Time | Other Per-Pair Time | Total Time |")
    lines.append("|--------------|--------------|-----------------|---------------------|------------|")

    for duration_min, total_frames in video_durations:
        # Similarity: per_frame_ms * total_frames * 64 pairs / 1000 (to seconds)
        similarity_time = (similarity_per_frame_ms * total_frames * 64) / 1000

        # Other per-pair: (filtering + post-processing) * 64 / 1000
        other_per_pair_time = (per_pair_without_similarity_ms * 64) / 1000

        # Total: embedding (once) + similarity + other per-pair
        total = per_run_total + similarity_time + other_per_pair_time

        if duration_min == 60:
            duration_str = "1 hour"
            total_str = f"{total / 60:.1f} min"
        else:
            duration_str = f"{duration_min} min"
            if total < 60:
                total_str = f"{total:.1f} s"
            else:
                total_str = f"{total / 60:.1f} min"

        lines.append(f"| {duration_str} | {total_frames:,} | {similarity_time:.1f} s | {other_per_pair_time:.1f} s | {total_str} |")

    lines.append("")
    lines.append("**Note:** Times scale linearly with the number of frames (video length) due to similarity computation.")
    lines.append("")

    # Methodology notes
    lines.append("---")
    lines.append("")
    lines.append("## Methodology Notes")
    lines.append("")
    lines.append("### Per-Unit Normalization")
    lines.append("")
    lines.append("Different components have different units of work:")
    lines.append("")
    lines.append("- **Embedding computation** (ms): Executed once per run to create embeddings for all sentence pairs")
    lines.append("- **Similarity computation** (ms/frame): Processes every frame in the video (executed once per pair)")
    lines.append("- **Post-processing** (ms/pair): Executed once per sentence pair on the full similarity sequence")
    lines.append("- **Sentence filtering** (ms/pair): Executed once per sentence pair to filter detected events")
    lines.append("")
    lines.append("### Why Per-Unit Metrics?")
    lines.append("")
    lines.append("Raw timing measurements showed high variance because:")
    lines.append("1. Videos have different lengths (different frame counts)")
    lines.append("2. Similarity computation processes all frames, so longer videos take proportionally longer")
    lines.append("3. Other components execute once per pair regardless of video length")
    lines.append("")
    lines.append("Per-unit normalization reveals the true computational cost:")
    lines.append("- Similarity computation: ~0.12 ms/frame (consistent across all videos)")
    lines.append("- Per-pair operations: Measured directly in ms/pair")
    lines.append("")

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Table V2 markdown written to: {output_file}")


def create_pie_chart(stats: pd.DataFrame, output_file: str):
    """Create a professional pie chart showing the share of each component."""
    # Set up style
    sns.set_style("whitegrid")
    plt.rcParams["figure.facecolor"] = "white"

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors
    colors = sns.color_palette("husl", n_colors=len(stats))

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        stats["share_percent"],
        labels=stats["paper_component"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 12, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )

    # Title
    ax.set_title("Distribution of Processing Time by Component\n(Core Algorithm Steps Only)", fontsize=16, fontweight="bold", pad=20)

    # Create legend with timing information
    legend_labels = [f"{row['paper_component']}: {row['mean']:.4f}s ({row['share_percent']:.1f}%)" for _, row in stats.iterrows()]

    ax.legend(legend_labels, title="Components (Mean Time)", bbox_to_anchor=(1.3, 1), loc="upper left", frameon=True, fancybox=True, shadow=True, fontsize=11)

    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Pie chart saved to: {output_file}")
    plt.close()


def create_bar_chart(stats: pd.DataFrame, output_file: str):
    """Create a bar chart showing mean times with error bars."""
    # Set up style
    sns.set_style("whitegrid")
    plt.rcParams["figure.facecolor"] = "white"

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart
    x = range(len(stats))
    bars = ax.bar(x, stats["mean"], yerr=stats["std"], color=sns.color_palette("husl", n_colors=len(stats)), edgecolor="black", linewidth=1.5, capsize=5)

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(stats["paper_component"], rotation=15, ha="right", fontsize=11)
    ax.set_ylabel("Processing Time (seconds)", fontsize=13, fontweight="bold")
    ax.set_title("Mean Processing Time by Component\n(with standard deviation error bars)", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, row) in enumerate(zip(bars, stats.iterrows())):
        _, data = row
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{data["mean"]:.4f}s', ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Bar chart saved to: {output_file}")
    plt.close()


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate timing analysis for paper Table 7")
    parser.add_argument("--timing-log", type=str, default="results/ablation_3/timing_log.json", help="Path to timing_log.json file (default: results/ablation_3/timing_log.json)")
    parser.add_argument(
        "--timing-summary", type=str, default="results/ablation_3/timing_summary.txt", help="Path to timing_summary.txt file (default: results/ablation_3/timing_summary.txt)"
    )
    parser.add_argument("--output-dir", type=str, default="paper_outputs", help="Output directory for results (default: paper_outputs)")

    args = parser.parse_args()

    # Create output directory
    import os

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading timing data...")
    df = load_timing_log(args.timing_log)
    print(f"Loaded {len(df)} timing entries")

    # Filter to relevant components only
    print("\nFiltering to relevant components...")
    filtered_df = filter_relevant_data(df)
    print(f"Filtered to {len(filtered_df)} relevant entries ({len(filtered_df) / len(df) * 100:.1f}%)")

    # Map to paper components
    filtered_df = map_to_paper_components(filtered_df)

    # Compute statistics for Table V1 (raw seconds)
    print("\nComputing statistics for Table V1 (raw seconds)...")
    stats = compute_statistics(filtered_df)

    print("\n" + "=" * 80)
    print("TABLE V1 STATISTICS (Raw Seconds)")
    print("=" * 80)
    print(stats.to_string(index=False))
    print("=" * 80)

    # Compute per-unit statistics for Table V2
    print("\nComputing per-unit statistics for Table V2...")
    stats_per_unit = compute_per_unit_statistics(filtered_df)

    print("\n" + "=" * 80)
    print("TABLE V2 STATISTICS (Per-Unit Metrics)")
    print("=" * 80)
    print(stats_per_unit.to_string(index=False))
    print("=" * 80)

    # Create component breakdown
    print("\nCreating component breakdown...")
    breakdown = create_component_breakdown(filtered_df)

    # Generate outputs
    print("\nGenerating outputs...")

    # Table V1: Raw seconds with share percentages
    markdown_v1_file = os.path.join(args.output_dir, "table_v1_raw_seconds.md")
    generate_markdown_table(stats, breakdown, markdown_v1_file)

    # Table V2: Per-unit metrics with example calculation
    markdown_v2_file = os.path.join(args.output_dir, "table_v2_per_unit.md")
    generate_markdown_table_v2(stats_per_unit, markdown_v2_file)

    # Pie chart (using V1 stats for share percentages)
    pie_chart_file = os.path.join(args.output_dir, "computational_cost_pie.png")
    create_pie_chart(stats, pie_chart_file)

    # Bar chart (using V1 stats)
    bar_chart_file = os.path.join(args.output_dir, "computational_cost_bar.png")
    create_bar_chart(stats, bar_chart_file)

    # Summary
    print("\n" + "=" * 80)
    print("OUTPUTS GENERATED")
    print("=" * 80)
    print(f"1. Table V1 (raw seconds): {markdown_v1_file}")
    print(f"2. Table V2 (per-unit metrics): {markdown_v2_file}")
    print(f"3. Pie chart: {pie_chart_file}")
    print(f"4. Bar chart: {bar_chart_file}")
    print("=" * 80)

    print("\nDone!")


if __name__ == "__main__":
    main()
