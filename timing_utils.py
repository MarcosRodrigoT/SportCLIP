"""
Timing and resource tracking utilities for the SportCLIP pipeline.
"""
import time
import json
import psutil
import os
from contextlib import contextmanager
from typing import Dict, List, Optional
from collections import defaultdict


class TimingTracker:
    """Track timing and resource usage for pipeline components."""

    def __init__(self, log_file: str = "timing_log.json"):
        self.log_file = log_file
        self.timings = []
        self.current_context = []
        self.process = psutil.Process(os.getpid())

    @contextmanager
    def track(self, component_name: str, metadata: Optional[Dict] = None):
        """Context manager to track timing and resources for a component."""
        # Record start state
        start_time = time.time()
        start_cpu = self.process.cpu_percent()
        start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB

        # Small delay to get accurate CPU reading
        time.sleep(0.01)

        try:
            yield
        finally:
            # Record end state
            end_time = time.time()
            end_cpu = self.process.cpu_percent()
            end_memory = self.process.memory_info().rss / (1024 * 1024)  # MB

            duration = end_time - start_time
            avg_cpu = (start_cpu + end_cpu) / 2
            memory_delta = end_memory - start_memory

            # Store timing data
            timing_entry = {
                "component": component_name,
                "duration_seconds": round(duration, 4),
                "cpu_percent": round(avg_cpu, 2),
                "memory_mb": round(end_memory, 2),
                "memory_delta_mb": round(memory_delta, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Add metadata if provided
            if metadata:
                timing_entry.update(metadata)

            self.timings.append(timing_entry)

    def save(self):
        """Save all timing data to the log file."""
        # Load existing data if file exists
        existing_data = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []

        # Append new timings
        existing_data.extend(self.timings)

        # Save to file
        with open(self.log_file, "w") as f:
            json.dump(existing_data, f, indent=2)

        # Clear current timings
        self.timings = []

    def print_summary(self, title: str = "Timing Summary"):
        """Print a summary of timings for the current run."""
        if not self.timings:
            return

        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print(f"{'='*80}")

        # Group by component
        component_times = defaultdict(list)
        for entry in self.timings:
            component_times[entry["component"]].append(entry["duration_seconds"])

        # Print summary table
        print(f"\n{'Component':<50} {'Time (s)':>12} {'Avg (s)':>12}")
        print(f"{'-'*80}")

        total_time = 0
        for component, times in sorted(component_times.items()):
            total = sum(times)
            avg = total / len(times)
            count = len(times)
            total_time += total

            if count > 1:
                print(f"{component:<50} {total:>12.4f} {avg:>12.4f} (n={count})")
            else:
                print(f"{component:<50} {total:>12.4f} {'-':>12}")

        print(f"{'-'*80}")
        print(f"{'TOTAL':<50} {total_time:>12.4f}")
        print(f"{'='*80}\n")


def aggregate_timing_logs(log_file: str = "timing_log.json", output_file: str = "timing_summary.txt"):
    """Aggregate all timing logs and create a comprehensive summary."""

    if not os.path.exists(log_file):
        print(f"No timing log found at {log_file}")
        return

    # Load all timing data
    with open(log_file, "r") as f:
        all_timings = json.load(f)

    if not all_timings:
        print("No timing data to aggregate")
        return

    # Group by component
    component_stats = defaultdict(lambda: {
        "count": 0,
        "total_time": 0.0,
        "times": [],
        "cpu_percents": [],
        "memory_deltas": [],
    })

    for entry in all_timings:
        component = entry["component"]
        stats = component_stats[component]

        stats["count"] += 1
        stats["total_time"] += entry["duration_seconds"]
        stats["times"].append(entry["duration_seconds"])
        stats["cpu_percents"].append(entry.get("cpu_percent", 0))
        stats["memory_deltas"].append(entry.get("memory_delta_mb", 0))

    # Calculate statistics
    import statistics

    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append("COMPREHENSIVE TIMING SUMMARY")
    summary_lines.append("=" * 100)
    summary_lines.append("")
    summary_lines.append(f"Total measurements: {len(all_timings)}")
    summary_lines.append(f"Unique components: {len(component_stats)}")
    summary_lines.append("")
    summary_lines.append("-" * 100)
    summary_lines.append(f"{'Component':<50} {'Count':>8} {'Total(s)':>12} {'Mean(s)':>12} {'StdDev(s)':>12} {'CPU%':>8}")
    summary_lines.append("-" * 100)

    # Sort by total time (descending)
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["total_time"], reverse=True)

    grand_total = 0
    for component, stats in sorted_components:
        count = stats["count"]
        total = stats["total_time"]
        mean = total / count
        stddev = statistics.stdev(stats["times"]) if count > 1 else 0
        avg_cpu = sum(stats["cpu_percents"]) / count if stats["cpu_percents"] else 0

        grand_total += total

        summary_lines.append(
            f"{component:<50} {count:>8} {total:>12.4f} {mean:>12.4f} {stddev:>12.4f} {avg_cpu:>8.2f}"
        )

    summary_lines.append("-" * 100)
    summary_lines.append(f"{'GRAND TOTAL':<50} {len(all_timings):>8} {grand_total:>12.4f}")
    summary_lines.append("=" * 100)
    summary_lines.append("")

    # Add detailed breakdown by video/sport if metadata exists
    video_stats = defaultdict(lambda: {"total_time": 0.0, "component_times": defaultdict(float)})

    for entry in all_timings:
        video = entry.get("video_name", "unknown")
        sport = entry.get("sport", "unknown")
        key = f"{sport}/{video}"

        video_stats[key]["total_time"] += entry["duration_seconds"]
        video_stats[key]["component_times"][entry["component"]] += entry["duration_seconds"]

    if len(video_stats) > 1:
        summary_lines.append("")
        summary_lines.append("=" * 100)
        summary_lines.append("PER-VIDEO BREAKDOWN")
        summary_lines.append("=" * 100)
        summary_lines.append("")

        for video_key in sorted(video_stats.keys()):
            stats = video_stats[video_key]
            summary_lines.append(f"\n{video_key}:")
            summary_lines.append(f"  Total time: {stats['total_time']:.4f}s")
            summary_lines.append(f"  Top components:")

            top_components = sorted(stats["component_times"].items(), key=lambda x: x[1], reverse=True)[:5]
            for comp, comp_time in top_components:
                pct = (comp_time / stats['total_time']) * 100
                summary_lines.append(f"    - {comp}: {comp_time:.4f}s ({pct:.1f}%)")

    # Write to file
    summary_text = "\n".join(summary_lines)
    with open(output_file, "w") as f:
        f.write(summary_text)

    # Also print to console
    print(summary_text)

    return summary_text


# Global tracker instance
_global_tracker = None


def get_tracker(log_file: str = "timing_log.json") -> TimingTracker:
    """Get or create the global timing tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TimingTracker(log_file)
    return _global_tracker
