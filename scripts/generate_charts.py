#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Chart generator for dispenso benchmarks.

This script takes benchmark results (JSON from run_benchmarks.py) and generates
comparison charts and a markdown report.

Usage:
    python generate_charts.py --input results.json --output-dir charts/

Requirements:
    pip install matplotlib pandas
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib.pyplot as plt
    import pandas as pd

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Error: matplotlib/pandas required for chart generation")
    print("Install with: pip install matplotlib pandas")


# Consistent color palette for libraries across all charts
LIBRARY_COLORS = {
    # Primary libraries
    "dispenso": "#2ecc71",  # Green
    "tbb": "#3498db",  # Blue
    "omp": "#e74c3c",  # Red
    "folly": "#9b59b6",  # Purple
    "taskflow": "#f39c12",  # Orange
    # Standard library variants
    "std": "#1abc9c",  # Teal
    "vector": "#1abc9c",  # Teal (same as std)
    "deque": "#e67e22",  # Dark orange
    "serial": "#95a5a6",  # Gray
    # Dispenso variants
    "dispenso2": "#27ae60",  # Darker green
    "dispenso auto chunk": "#58d68d",  # Lighter green
    "dispenso static chunk": "#1e8449",  # Dark green
    "dispenso mostly idle": "#82e0aa",  # Pale green
    "dispenso very idle": "#abebc6",  # Very pale green
    # TBB variants
    "tbb concurrent vector": "#5dade2",  # Light blue
    # Async
    "async": "#f1c40f",  # Yellow
}


def get_library_color(name: str) -> str:
    """Get consistent color for a library name."""
    name_lower = name.lower().replace("_", " ")

    # Try exact match first
    if name_lower in LIBRARY_COLORS:
        return LIBRARY_COLORS[name_lower]

    # Try prefix matching
    for key, color in LIBRARY_COLORS.items():
        if name_lower.startswith(key) or key in name_lower:
            return color

    # Default color for unknown libraries
    return "#7f8c8d"  # Gray


def get_time_scale(values, original_unit: str = "ns"):
    """Determine appropriate time unit and scale factor for display.

    Returns (scale_factor, display_unit) where:
    - scale_factor: divide values by this to get display values
    - display_unit: string label for the axis (e.g., "ms", "µs")

    This avoids matplotlib's scientific notation by choosing appropriate units.
    """
    max_val = max(values) if len(values) > 0 else 0

    if original_unit == "ns":
        if max_val >= 1e9:
            return 1e9, "s"
        elif max_val >= 1e6:
            return 1e6, "ms"
        elif max_val >= 1e3:
            return 1e3, "µs"
        else:
            return 1, "ns"
    elif original_unit == "us":
        if max_val >= 1e6:
            return 1e6, "s"
        elif max_val >= 1e3:
            return 1e3, "ms"
        else:
            return 1, "µs"
    elif original_unit == "ms":
        if max_val >= 1e3:
            return 1e3, "s"
        else:
            return 1, "ms"
    else:
        return 1, original_unit


def extract_benchmark_data(results: List[Dict]) -> "pd.DataFrame":
    """Extract benchmark data into a DataFrame for plotting."""
    rows = []

    for result in results:
        if not result.get("success") or "data" not in result:
            continue

        data = result["data"]
        benchmarks = data.get("benchmarks", [])

        for bm in benchmarks:
            row = {
                "suite": result["name"].replace("_benchmark", ""),
                "name": bm.get("name", "unknown"),
                "real_time": bm.get("real_time", 0),
                "cpu_time": bm.get("cpu_time", 0),
                "time_unit": bm.get("time_unit", "ns"),
                "iterations": bm.get("iterations", 0),
                "skipped": bm.get("error_occurred", False),
                "error_message": bm.get("error_message", ""),
            }

            # Extract any custom counters
            for key, value in bm.items():
                if key not in row and isinstance(value, (int, float)):
                    row[key] = value

            rows.append(row)

    return pd.DataFrame(rows)


def parse_benchmark_name(name: str) -> Dict[str, Any]:
    """Parse benchmark name to extract library, threads, worksize from patterns like:
    BM_library/threads/worksize/real_time or BM_library<template>
    """
    result = {"library": name, "threads": None, "worksize": None, "raw": name}

    # Handle template-style names like BM_serial<kSmallSize>
    if "<" in name and ">" in name:
        match = re.match(r"([^<]+)<([^>]+)>", name)
        if match:
            result["library"] = match.group(1)
            result["worksize"] = match.group(2)
        return result

    # Handle slash-delimited names like BM_dispenso/8/1000/real_time
    parts = name.split("/")
    # Filter out "real_time" suffix if present
    if parts and parts[-1] == "real_time":
        parts = parts[:-1]

    if len(parts) >= 3:
        # Pattern: library/threads/worksize
        result["library"] = parts[0]
        try:
            result["threads"] = int(parts[1])
            result["worksize"] = parts[2]
        except (ValueError, IndexError):
            pass
    elif len(parts) == 2:
        # Pattern: library/threads (no worksize, e.g., BM_dispenso_very_idle/8)
        result["library"] = parts[0]
        try:
            result["threads"] = int(parts[1])
            result["worksize"] = "default"
        except ValueError:
            pass

    return result


def generate_line_chart(
    suite: str,
    suite_df: "pd.DataFrame",
    output_dir: Path,
    worksize: str,
) -> Path:
    """Generate a line chart for thread scaling benchmarks."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get data for this worksize
    ws_df = suite_df[suite_df["worksize"] == worksize].copy()
    if ws_df.empty:
        plt.close()
        return None

    # Determine appropriate time units
    original_unit = ws_df["time_unit"].iloc[0]
    scale, display_unit = get_time_scale(ws_df["real_time"].values, original_unit)

    # Group by library and plot each as a line
    libraries = sorted(ws_df["library"].unique())

    has_data = False
    for lib in libraries:
        lib_df = ws_df[ws_df["library"] == lib].sort_values("threads")
        # Filter out invalid results (time = 0 or near 0, often indicates failed runs)
        lib_df = lib_df[lib_df["real_time"] > 0.01]
        if not lib_df.empty and lib_df["threads"].notna().any():
            # Clean up library name for legend
            label = lib.replace("BM_", "").replace("_", " ")
            color = get_library_color(label)
            ax.plot(
                lib_df["threads"],
                lib_df["real_time"] / scale,
                marker="o",
                label=label,
                color=color,
                linewidth=2,
                markersize=6,
            )
            has_data = True

    if not has_data:
        plt.close()
        return None

    # Format worksize for title
    ws_display = worksize
    if worksize.isdigit():
        ws_int = int(worksize)
        if ws_int >= 1000000:
            ws_display = f"{ws_int // 1000000}M"
        elif ws_int >= 1000:
            ws_display = f"{ws_int // 1000}K"

    # Clean up suite name for title
    suite_display = suite.replace("_", " ").title()

    ax.set_xlabel("Threads", fontsize=12)
    ax.set_ylabel(f"Time ({display_unit})", fontsize=12)
    ax.set_title(f"{suite_display} - {ws_display} Elements", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(5, len(libraries)),
        fontsize=11,
    )

    plt.tight_layout()
    chart_path = output_dir / f"{suite}_{worksize}_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    return chart_path


def generate_bar_chart(suite: str, suite_df: "pd.DataFrame", output_dir: Path) -> Path:
    """Generate a vertical bar chart for simple benchmarks."""
    import numpy as np

    # For simple benchmarks, just show each benchmark as its own bar
    # Skip the complex operation/library parsing that causes issues

    num_bars = len(suite_df)
    fig_width = max(10, min(20, num_bars * 1.2))
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    # Sort for logical grouping: serial implementations first, then parallel
    # Within each group: serial baseline first, then dispenso, then others alphabetically
    def library_order(lib):
        lib_lower = lib.lower()
        if lib_lower == "serial":
            return (0, lib_lower)  # Serial baseline first
        elif lib_lower.startswith("dispenso"):
            return (1, lib_lower)  # Dispenso second
        elif lib_lower.startswith("taskflow"):
            return (2, lib_lower)  # Taskflow third
        else:
            return (3, lib_lower)  # Others (tbb, etc.) alphabetically after

    def sort_key(name):
        clean = name.replace("BM_", "").replace("/real_time", "")
        # Group parallel implementations after serial ones
        is_parallel = "_par" in clean or clean.endswith("par")
        # Extract library name for ordering
        lib = clean.replace("_par", "").replace(" par", "")
        return (1 if is_parallel else 0, library_order(lib))

    suite_df_sorted = suite_df.copy()
    suite_df_sorted["sort_key"] = suite_df_sorted["name"].apply(sort_key)
    suite_df_sorted = suite_df_sorted.sort_values("sort_key")
    suite_df_sorted = suite_df_sorted.drop(columns=["sort_key"])

    # Clean up labels and get colors
    labels = suite_df_sorted["name"].apply(
        lambda x: x.replace("BM_", "").replace("/real_time", "").replace("_", " ")
    )
    colors = [get_library_color(label) for label in labels]

    # Convert to sensible units based on max value
    max_val = suite_df_sorted["real_time"].max()
    original_unit = suite_df_sorted["time_unit"].iloc[0]

    # Determine best display unit and scale factor
    if original_unit == "ns":
        if max_val >= 1e9:
            scale = 1e9
            display_unit = "s"
        elif max_val >= 1e6:
            scale = 1e6
            display_unit = "ms"
        elif max_val >= 1e3:
            scale = 1e3
            display_unit = "µs"
        else:
            scale = 1
            display_unit = "ns"
    else:
        scale = 1
        display_unit = original_unit

    scaled_values = suite_df_sorted["real_time"] / scale

    x = np.arange(len(labels))
    bars = ax.bar(x, scaled_values, color=colors)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)

    # Clean up suite name for title
    suite_display = suite.replace("_", " ").title()

    ax.set_ylabel(f"Time ({display_unit})", fontsize=12)
    ax.set_title(f"{suite_display} Benchmark Results", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=10)

    # Add value labels on top of bars
    max_scaled = scaled_values.max()
    for bar, val in zip(bars, scaled_values):
        # Format value for display
        if val >= 1000:
            label = f"{val:.0f}"
        elif val >= 100:
            label = f"{val:.1f}"
        elif val >= 10:
            label = f"{val:.1f}"
        else:
            label = f"{val:.2f}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90 if num_bars > 15 else 0,
        )

    # Add some headroom for labels
    ax.set_ylim(0, max_scaled * 1.15)

    plt.tight_layout()
    chart_path = output_dir / f"{suite}_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    return chart_path


def generate_grouped_bar_chart(
    suite: str,
    suite_df: "pd.DataFrame",
    output_dir: Path,
    operations: list,
    libraries: list,
) -> Path:
    """Generate a grouped vertical bar chart comparing libraries per operation."""
    import numpy as np

    num_ops = len(operations)
    num_libs = len(libraries)
    fig_width = max(12, min(24, num_ops * num_libs * 0.4))
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    x = np.arange(num_ops)
    width = 0.8 / num_libs

    max_val = 0
    for i, lib in enumerate(libraries):
        lib_df = suite_df[suite_df["lib"] == lib]
        values = []
        for op in operations:
            op_df = lib_df[lib_df["operation"] == op]
            if not op_df.empty:
                values.append(op_df["real_time"].iloc[0])
            else:
                values.append(0)

        values = np.array(values)
        max_val = max(max_val, values.max())
        color = get_library_color(lib)
        bars = ax.bar(
            x + i * width - width * num_libs / 2 + width / 2,
            values,
            width,
            label=lib,
            color=color,
        )

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                if val >= 1e9:
                    label = f"{val / 1e9:.1f}B"
                elif val >= 1e6:
                    label = f"{val / 1e6:.1f}M"
                elif val >= 1e3:
                    label = f"{val / 1e3:.1f}K"
                else:
                    label = f"{val:.0f}"

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(operations, rotation=45, ha="right", fontsize=10)

    suite_display = suite.replace("_", " ").title()
    ax.set_ylabel(f"Time ({suite_df['time_unit'].iloc[0]})", fontsize=12)
    ax.set_title(f"{suite_display} Benchmark Results", fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0, max_val * 1.2)

    plt.tight_layout()
    chart_path = output_dir / f"{suite}_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    return chart_path


def generate_horizontal_grouped_bar_chart(
    title: str,
    suite_df: "pd.DataFrame",
    output_dir: Path,
    chart_name: str,
) -> Path:
    """Generate a horizontal grouped bar chart with log scale for comparing libraries."""
    import numpy as np

    # Parse benchmark names to extract operation and library
    def parse_cv_benchmark(name: str):
        """Parse concurrent_vector benchmark name into operation and library."""
        clean = name.replace("BM_", "").replace("/real_time", "")
        # Library prefixes
        lib_prefixes = ["std_", "deque_", "tbb_", "dispenso_"]
        for prefix in lib_prefixes:
            if clean.startswith(prefix):
                lib = prefix.rstrip("_")
                op = clean[len(prefix) :]
                # Map library names to display names
                lib_map = {
                    "std": "std::vector",
                    "deque": "std::deque",
                    "tbb": "tbb::concurrent_vector",
                    "dispenso": "dispenso::ConcurrentVector",
                }
                return op, lib_map.get(lib, lib)
        return clean, "unknown"

    suite_df = suite_df.copy()
    parsed = suite_df["name"].apply(parse_cv_benchmark)
    suite_df["operation"] = parsed.apply(lambda x: x[0])
    suite_df["lib"] = parsed.apply(lambda x: x[1])

    # Get unique operations and libraries
    operations = suite_df["operation"].unique().tolist()
    libraries = [
        "std::vector",
        "std::deque",
        "tbb::concurrent_vector",
        "dispenso::ConcurrentVector",
    ]
    libraries = [lib for lib in libraries if lib in suite_df["lib"].values]

    num_ops = len(operations)
    num_libs = len(libraries)
    fig_height = max(6, num_ops * 0.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y = np.arange(num_ops)
    height = 0.8 / num_libs

    # Library colors
    lib_colors = {
        "std::vector": "#808080",  # gray
        "std::deque": "#FF8C00",  # orange
        "tbb::concurrent_vector": "#4285F4",  # blue
        "dispenso::ConcurrentVector": "#34A853",  # green
    }

    for i, lib in enumerate(libraries):
        lib_df = suite_df[suite_df["lib"] == lib]
        values = []
        for op in operations:
            op_df = lib_df[lib_df["operation"] == op]
            if not op_df.empty:
                values.append(op_df["real_time"].iloc[0])
            else:
                values.append(0)

        values = np.array(values)
        color = lib_colors.get(lib, "#888888")
        ax.barh(
            y + i * height - height * num_libs / 2 + height / 2,
            values,
            height,
            label=lib,
            color=color,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(operations, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Time (ns) - log scale", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / f"{chart_name}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    return chart_path


def generate_concurrent_vector_charts(
    df: "pd.DataFrame", output_dir: Path, suffix: str = ""
):
    """Generate specialized charts for concurrent_vector benchmarks.

    Args:
        df: DataFrame with benchmark data
        output_dir: Output directory for charts
        suffix: Optional suffix for chart names (e.g., "_tcmalloc")
    """
    # Define operation categories
    serial_ops = [
        "push_back_serial",
        "push_back_serial_reserve",
        "iterate",
        "iterate_reverse",
        "lower_bound",
        "index",
        "random",
    ]
    parallel_ops = [
        "parallel",
        "parallel_reserve",
        "parallel_clear",
        "parallel_grow_by_10",
        "parallel_grow_by_100",
        "parallel_grow_by_max",
    ]

    # More precise filtering
    def get_operation(name):
        clean = name.replace("BM_", "").replace("/real_time", "")
        for prefix in ["std_", "deque_", "tbb_", "dispenso_"]:
            if clean.startswith(prefix):
                return clean[len(prefix) :]
        return clean

    df_with_op = df.copy()
    df_with_op["operation"] = df_with_op["name"].apply(get_operation)

    serial_df = df_with_op[df_with_op["operation"].isin(serial_ops)]
    parallel_df = df_with_op[df_with_op["operation"].isin(parallel_ops)]

    # Title suffix for tcmalloc variant
    title_suffix = " (tcmalloc)" if suffix else ""

    charts = []

    if not serial_df.empty:
        chart = generate_horizontal_grouped_bar_chart(
            f"Concurrent Vector - Serial/Access Operations{title_suffix}",
            serial_df,
            output_dir,
            f"concurrent_vector_serial{suffix}_chart",
        )
        charts.append(chart)
        print(f"Generated: {chart}")

    if not parallel_df.empty:
        chart = generate_horizontal_grouped_bar_chart(
            f"Concurrent Vector - Parallel Operations{title_suffix}",
            parallel_df,
            output_dir,
            f"concurrent_vector_parallel{suffix}_chart",
        )
        charts.append(chart)
        print(f"Generated: {chart}")

    return charts


def generate_simple_for_charts(
    suite_df: "pd.DataFrame", output_dir: Path, suite: str = "simple_for"
):
    """Generate specialized charts for simple_for/summing_for/trivial_compute benchmarks.

    - Filters out auto_chunk and static_chunk variants (just uses 'dispenso')
    - Creates zoomed charts focused on competitive range
    - No max_threads comparison chart
    """

    # Filter out auto_chunk and static_chunk - just keep base dispenso
    def should_keep(lib):
        lib_lower = lib.lower()
        if "auto" in lib_lower or "static" in lib_lower:
            return False
        return True

    suite_df = suite_df[suite_df["library"].apply(should_keep)].copy()

    worksizes = suite_df["worksize"].dropna().unique()
    charts = []

    for worksize in sorted(
        worksizes,
        key=lambda x: (not str(x).isdigit(), int(x) if str(x).isdigit() else 0),
    ):
        ws_df = suite_df[suite_df["worksize"] == worksize].copy()
        if ws_df.empty:
            continue

        # Format worksize for title
        ws_display = worksize
        if str(worksize).isdigit():
            ws_int = int(worksize)
            if ws_int >= 1000000000:
                ws_display = f"{ws_int // 1000000000}B"
            elif ws_int >= 1000000:
                ws_display = f"{ws_int // 1000000}M"
            elif ws_int >= 1000:
                ws_display = f"{ws_int // 1000}K"

        # Generate main chart
        fig, ax = plt.subplots(figsize=(12, 8))

        # Determine appropriate time units for this worksize
        original_unit = ws_df["time_unit"].iloc[0]
        scale, display_unit = get_time_scale(ws_df["real_time"].values, original_unit)

        libraries = sorted(ws_df["library"].unique())
        has_data = False
        for lib in libraries:
            lib_df = ws_df[ws_df["library"] == lib].sort_values("threads")
            if not lib_df.empty and lib_df["threads"].notna().any():
                label = lib.replace("BM_", "").replace("_", " ")
                color = get_library_color(label)
                ax.plot(
                    lib_df["threads"],
                    lib_df["real_time"] / scale,
                    marker="o",
                    label=label,
                    color=color,
                    linewidth=2,
                    markersize=6,
                )
                has_data = True

        if not has_data:
            plt.close()
            continue

        suite_display = suite.replace("_", " ").title()
        ax.set_xlabel("Threads", fontsize=12)
        ax.set_ylabel(f"Time ({display_unit})", fontsize=12)
        ax.set_title(f"{suite_display} - {ws_display} Elements", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=min(5, len(libraries)),
            fontsize=11,
        )

        plt.tight_layout()
        chart_path = output_dir / f"{suite}_{worksize}_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        charts.append(chart_path)
        print(f"Generated: {chart_path}")

        # Generate zoomed chart focusing on competitive libraries (dispenso, tbb, omp)
        # Find the max value among competitive libraries to set appropriate y-limit
        competitive_libs = ["BM_dispenso", "BM_tbb", "BM_omp"]
        competitive_df = ws_df[ws_df["library"].isin(competitive_libs)]

        if not competitive_df.empty and len(competitive_df["library"].unique()) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Determine y-limit: use values from mid-range thread counts
            # Use data-driven percentile-based range, zoom aggressively
            thread_min = competitive_df["threads"].quantile(0.25)
            thread_max = competitive_df["threads"].quantile(0.75)
            mid_thread_df = competitive_df[
                (competitive_df["threads"] >= thread_min)
                & (competitive_df["threads"] <= thread_max)
            ]
            if not mid_thread_df.empty:
                # Use 1.5x for tight zoom that focuses on competitive differences
                y_max = (mid_thread_df["real_time"].max() * 1.5) / scale
            else:
                # Fallback: use median-based limit
                y_max = (competitive_df["real_time"].median() * 2) / scale

            for lib in libraries:
                lib_df = ws_df[ws_df["library"] == lib].sort_values("threads")
                if not lib_df.empty and lib_df["threads"].notna().any():
                    label = lib.replace("BM_", "").replace("_", " ")
                    color = get_library_color(label)
                    ax.plot(
                        lib_df["threads"],
                        lib_df["real_time"] / scale,
                        marker="o",
                        label=label,
                        color=color,
                        linewidth=2,
                        markersize=6,
                    )

            # Only zoom y-axis, keep full x-axis range to show divergence
            ax.set_ylim(0, y_max)
            ax.set_xlabel("Threads", fontsize=12)
            ax.set_ylabel(f"Time ({display_unit})", fontsize=12)
            ax.set_title(
                f"{suite_display} - {ws_display} Elements (Y-Axis Zoomed)", fontsize=14
            )
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=10)
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.08),
                ncol=min(5, len(libraries)),
                fontsize=11,
            )

            plt.tight_layout()
            chart_path = output_dir / f"{suite}_{worksize}_zoomed_chart.png"
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close()
            charts.append(chart_path)
            print(f"Generated: {chart_path}")

    return charts


def _parse_future_benchmark(name: str):
    """Parse future benchmark name into (implementation, size)."""
    clean = name.replace("BM_", "").replace("/real_time", "")

    # Extract size
    size = None
    for sz in ["kSmallSize", "kMediumSize", "kLargeSize"]:
        if sz in clean:
            size = sz.replace("k", "").replace("Size", "")  # "Small", "Medium", "Large"
            clean = clean.replace(f"<{sz}>", "")
            break

    # Map implementation names
    impl_map = {
        "serial_tree": "serial",
        "std_tree": "std::async",
        "folly_tree": "folly::Future",
        "dispenso_tree": "dispenso::Future",
        "dispenso_tree_when_all": "dispenso::when_all",
        "dispenso_taskset_tree": "dispenso::TaskSet",
        "dispenso_taskset_tree_bulk": "dispenso::TaskSet (bulk)",
    }

    impl = impl_map.get(clean, clean.replace("_", " "))
    return impl, size


def _format_time_value(val_ns):
    """Format time value in appropriate units."""
    if val_ns >= 1e9:
        return f"{val_ns / 1e9:.1f}s"
    elif val_ns >= 1e6:
        return f"{val_ns / 1e6:.1f}ms"
    elif val_ns >= 1e3:
        return f"{val_ns / 1e3:.0f}µs"
    else:
        return f"{val_ns:.0f}ns"


# Colors for future benchmark implementations
_FUTURE_IMPL_COLORS = {
    "serial": "#95a5a6",  # Gray
    "std::async": "#e74c3c",  # Red
    "folly::Future": "#e67e22",  # Orange
    "dispenso::Future": "#3498db",  # Blue
    "dispenso::TaskSet": "#2ecc71",  # Green
    "dispenso::TaskSet (bulk)": "#1e8449",  # Dark green
    "dispenso::when_all": "#9b59b6",  # Purple
}


def _add_future_bar_value_labels(ax, all_bars, all_values, y_max):
    """Add value labels on bars for zoomed future charts."""
    for bars, values in zip(all_bars, all_values):
        for bar, val in zip(bars, values):
            if val > 0:
                label = _format_time_value(val)
                bar_center_x = bar.get_x() + bar.get_width() / 2
                if val > y_max:
                    ax.text(
                        bar_center_x,
                        y_max * 0.95,
                        label,
                        ha="center",
                        va="top",
                        fontsize=7,
                        color="white",
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        bar_center_x,
                        bar.get_height() + y_max * 0.01,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )


def _render_future_bar_chart(
    suite_df: "pd.DataFrame",
    implementations: list,
    sizes: list,
    output_dir: Path,
    chart_name: str,
    title: str,
    y_max=None,
    show_value_labels: bool = False,
) -> Path:
    """Render a grouped bar chart for future benchmark results.

    Args:
        y_max: If set, clip y-axis at this value (for zoomed charts).
        show_value_labels: If True, add value labels on bars.
    """
    import numpy as np

    num_impls = len(implementations)
    num_sizes = len(sizes)
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(num_sizes)
    width = 0.8 / num_impls

    max_val = 0
    dnf_positions = []
    all_bars = []
    all_values = []

    for i, impl in enumerate(implementations):
        impl_df = suite_df[suite_df["implementation"] == impl]
        values = []
        for size in sizes:
            size_df = impl_df[impl_df["size"] == size]
            values.append(size_df["real_time"].iloc[0] if not size_df.empty else 0)

        values = np.array(values)
        if values.max() > 0:
            max_val = max(max_val, values.max())
        color = _FUTURE_IMPL_COLORS.get(impl, "#888888")
        bar_x = x + i * width - width * num_impls / 2 + width / 2
        bars = ax.bar(bar_x, values, width, label=impl, color=color)
        all_bars.append(bars)
        all_values.append(values)

        # Track zero-value positions for "DID NOT FINISH" labels
        for j, val in enumerate(values):
            if val == 0 and not impl_df.empty:
                dnf_positions.append((bar_x[j], impl))

    # Draw "DID NOT FINISH" labels
    ref_max = y_max if y_max is not None else max_val
    for bar_x_pos, impl in dnf_positions:
        ax.text(
            bar_x_pos,
            ref_max * 0.02,
            "DID NOT\nFINISH",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            color=_FUTURE_IMPL_COLORS.get(impl, "#888888"),
            fontweight="bold",
            fontstyle="italic",
        )

    # Add value labels on the zoomed chart
    if show_value_labels and y_max is not None:
        _add_future_bar_value_labels(ax, all_bars, all_values, y_max)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=12)
    ax.set_xlabel("Tree Size", fontsize=12)
    if y_max is not None:
        ax.set_ylim(0, y_max)

    # Set y-axis label with appropriate time units
    time_unit = suite_df["time_unit"].iloc[0]
    effective_max = y_max if y_max is not None else max_val
    scale, display_unit = get_time_scale([effective_max], time_unit)
    ax.set_ylabel(f"Time ({display_unit})", fontsize=12)
    if scale != 1:
        fmt_str = ".2f" if y_max is not None and display_unit == "s" else ".1f"
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p, s=scale, f=fmt_str: f"{x / s:{f}}")
        )

    ax.set_title(title, fontsize=14)
    ax.legend(title="Implementation", loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    chart_path = output_dir / f"{chart_name}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Generated: {chart_path}")
    return chart_path


def generate_future_charts(suite_df: "pd.DataFrame", output_dir: Path):
    """Generate grouped bar chart for future benchmark.

    X-axis: Tree size (Small, Medium, Large)
    Grouped bars by implementation (serial, std::async, folly, dispenso variants)
    """
    suite_df = suite_df.copy()
    parsed = suite_df["name"].apply(_parse_future_benchmark)
    suite_df["implementation"] = parsed.apply(lambda x: x[0])
    suite_df["size"] = parsed.apply(lambda x: x[1])

    suite_df = suite_df[suite_df["size"].notna()]
    if suite_df.empty:
        return []

    # Order implementations logically
    impl_order = [
        "serial",
        "std::async",
        "folly::Future",
        "dispenso::Future",
        "dispenso::TaskSet",
        "dispenso::TaskSet (bulk)",
        "dispenso::when_all",
    ]
    implementations = [i for i in impl_order if i in suite_df["implementation"].values]
    for impl in suite_df["implementation"].unique():
        if impl not in implementations:
            implementations.append(impl)

    sizes = [s for s in ["Small", "Medium", "Large"] if s in suite_df["size"].values]

    # Main chart (full scale)
    charts = [
        _render_future_bar_chart(
            suite_df,
            implementations,
            sizes,
            output_dir,
            "future_chart",
            "Future/Async Tree Build Benchmark",
        )
    ]

    # Zoomed chart - clip y-axis to focus on competitive range
    if len(implementations) > 1:
        dispenso_impls = [
            i for i in implementations if i.startswith("dispenso") or i == "serial"
        ]
        dispenso_df = suite_df[suite_df["implementation"].isin(dispenso_impls)]
        if not dispenso_df.empty:
            y_max = dispenso_df["real_time"].max() * 1.5
        else:
            y_max = suite_df["real_time"].median() * 2

        charts.append(
            _render_future_bar_chart(
                suite_df,
                implementations,
                sizes,
                output_dir,
                "future_zoomed_chart",
                "Future/Async Tree Build (Y-Axis Zoomed)",
                y_max=y_max,
                show_value_labels=True,
            )
        )

    return charts


def generate_graph_charts(suite_df: "pd.DataFrame", output_dir: Path):
    """Generate horizontal bar chart for graph benchmark with logical grouping."""
    import numpy as np

    # Parse and clean benchmark names
    def parse_graph_name(name: str):
        clean = name.replace("BM_", "").replace("/real_time", "")
        # Simplify template names
        clean = clean.replace("<dispenso::BiPropGraph>", " (BiProp)")
        clean = clean.replace("<dispenso::Graph>", " (Graph)")
        clean = clean.replace("_", " ")
        return clean

    suite_df = suite_df.copy()
    suite_df["display_name"] = suite_df["name"].apply(parse_graph_name)

    # Define logical grouping order
    group_order = [
        # Build big tree - taskflow vs dispenso
        "taskflow build big tree",
        "build big tree (Graph)",
        "build big tree (BiProp)",
        # Dependency chain operations
        "build bi prop dependency chain",
        "build bi prop dependency group",
        "build dependency chain (Graph)",
        "build dependency chain (BiProp)",
        # Execute operations
        "execute dependency chain (Graph)",
        "execute dependency chain (BiProp)",
        # Forward propagator
        "forward propagator node (Graph)",
        "forward propagator node (BiProp)",
    ]

    # Sort by group order
    def sort_key(name):
        try:
            return group_order.index(name)
        except ValueError:
            return len(group_order)

    suite_df["sort_order"] = suite_df["display_name"].apply(sort_key)
    suite_df = suite_df.sort_values("sort_order")

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    y = np.arange(len(suite_df))
    values = suite_df["real_time"].values
    labels = suite_df["display_name"].values

    # Color by type
    colors = []
    for name in labels:
        if "taskflow" in name.lower():
            colors.append("#f39c12")  # Orange for taskflow
        elif "BiProp" in name:
            colors.append("#27ae60")  # Darker green for BiProp
        else:
            colors.append("#2ecc71")  # Green for dispenso Graph

    bars = ax.barh(y, values, color=colors)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)

    # Convert units if needed
    max_val = values.max()
    time_unit = suite_df["time_unit"].iloc[0]
    if time_unit == "ns" and max_val >= 1e6:
        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x / 1e6:.1f}"))
    elif time_unit == "ns" and max_val >= 1e3:
        ax.set_xlabel("Time (µs)", fontsize=12)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x / 1e3:.0f}"))
    else:
        ax.set_xlabel(f"Time ({time_unit})", fontsize=12)

    ax.set_title("Graph Benchmark Results", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        if time_unit == "ns" and max_val >= 1e6:
            label = f"{val / 1e6:.2f}"
        elif time_unit == "ns" and max_val >= 1e3:
            label = f"{val / 1e3:.0f}"
        else:
            label = f"{val:.0f}"
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {label}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    chart_path = output_dir / "graph_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Generated: {chart_path}")

    return [chart_path]


def _clean_allocator_name(name):
    """Clean up allocator benchmark display names."""
    clean = name.replace("BM_", "").replace("/real_time", "")
    clean = clean.replace("<kSmallSize>", "[S]")
    clean = clean.replace("<kMediumSize>", "[M]")
    clean = clean.replace("<kLargeSize>", "[L]")
    clean = clean.replace("<kSmallSize,", "[S,")
    clean = clean.replace("<kMediumSize,", "[M,")
    clean = clean.replace("<kLargeSize,", "[L,")
    clean = clean.replace(">", "]")
    clean = clean.replace("_", " ")
    return clean


def _generate_small_buffer_charts(suite_df: "pd.DataFrame", output_dir: Path) -> list:
    """Generate horizontal bar chart for small_buffer benchmark."""
    import numpy as np

    suite_df = suite_df.copy()
    suite_df["display_name"] = suite_df["name"].apply(_clean_allocator_name)

    def parse_small_buffer_name(name):
        if "[S]" in name:
            size, size_label = 0, "S"
        elif "[M]" in name:
            size, size_label = 1, "M"
        elif "[L]" in name:
            size, size_label = 2, "L"
        else:
            size, size_label = 3, "?"

        iter_match = re.search(r"/(\d+)", name)
        iterations = int(iter_match.group(1)) if iter_match else 0
        thread_match = re.search(r"threads:(\d+)", name)
        threads = int(thread_match.group(1)) if thread_match else 1
        is_small_buffer = "small buffer" in name.lower()
        return size, iterations, threads, is_small_buffer, size_label

    parsed = suite_df["display_name"].apply(parse_small_buffer_name)
    suite_df["size_order"] = parsed.apply(lambda x: x[0])
    suite_df["iterations"] = parsed.apply(lambda x: x[1])
    suite_df["threads"] = parsed.apply(lambda x: x[2])
    suite_df["is_small_buffer"] = parsed.apply(lambda x: x[3])

    suite_df = suite_df.sort_values(
        ["size_order", "iterations", "threads", "is_small_buffer"]
    )

    fig, ax = plt.subplots(figsize=(14, max(8, len(suite_df) * 0.35)))
    y = np.arange(len(suite_df))
    values = suite_df["real_time"].values.copy()
    labels = suite_df["display_name"].values

    # Detect outliers using 90th percentile and clip
    threshold = np.percentile(values, 90)
    clipped_values = np.minimum(values, threshold * 1.1)
    outlier_mask = values > threshold

    colors = [
        "#2ecc71" if is_sb else "#e74c3c" for is_sb in suite_df["is_small_buffer"]
    ]

    bars = ax.barh(y, clipped_values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)

    # Add outlier annotations
    for i, (is_outlier, orig_val) in enumerate(zip(outlier_mask, values)):
        if is_outlier:
            label = f"({_format_time_value(orig_val)})"
            ax.text(
                clipped_values[i],
                i,
                f" {label}",
                va="center",
                fontsize=7,
                style="italic",
            )

    time_unit = suite_df["time_unit"].iloc[0]
    scale, display_unit = get_time_scale(clipped_values, time_unit)
    ax.set_xlabel(f"Time ({display_unit})", fontsize=12)
    if scale != 1:
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p, s=scale: f"{x / s:.1f}")
        )

    ax.set_title("Small Buffer Allocator Benchmark", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", label="SmallBufferAllocator"),
        Patch(facecolor="#e74c3c", label="new/delete"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    chart_path = output_dir / "small_buffer_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Generated: {chart_path}")
    return [chart_path]


def _generate_pool_allocator_split_charts(
    suite_df: "pd.DataFrame", output_dir: Path
) -> list:
    """Generate split fast/slow horizontal bar charts for pool_allocator benchmark."""
    import numpy as np

    suite_df = suite_df.copy()
    suite_df["display_name"] = suite_df["name"].apply(_clean_allocator_name)

    charts = []
    threshold = suite_df["real_time"].median()

    for group_name, group_df in [
        ("fast", suite_df[suite_df["real_time"] < threshold]),
        ("slow", suite_df[suite_df["real_time"] >= threshold]),
    ]:
        if group_df.empty:
            continue

        group_df = group_df.sort_values("real_time")

        fig, ax = plt.subplots(figsize=(12, max(6, len(group_df) * 0.35)))
        y = np.arange(len(group_df))
        values = group_df["real_time"].values
        labels = group_df["display_name"].values

        colors = []
        for name in labels:
            name_lower = name.lower()
            if "malloc" in name_lower:
                colors.append("#e74c3c")
            elif "arena" in name_lower:
                colors.append("#2ecc71")
            elif "nl pool" in name_lower:
                colors.append("#3498db")
            else:
                colors.append("#9b59b6")

        ax.barh(y, values, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)

        max_val = values.max()
        time_unit = group_df["time_unit"].iloc[0]
        scale, display_unit = get_time_scale([max_val], time_unit)
        ax.set_xlabel(f"Time ({display_unit})", fontsize=12)
        if scale != 1:
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p, s=scale: f"{x / s:.1f}")
            )

        title_suffix = "Fast Operations" if group_name == "fast" else "Slow Operations"
        ax.set_title(f"Pool Allocator - {title_suffix}", fontsize=14)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        chart_path = output_dir / f"pool_allocator_{group_name}_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        charts.append(chart_path)
        print(f"Generated: {chart_path}")

    return charts


def generate_allocator_charts(suite_df: "pd.DataFrame", output_dir: Path, suite: str):
    """Generate horizontal bar charts for pool_allocator and small_buffer benchmarks."""
    if suite == "small_buffer":
        return _generate_small_buffer_charts(suite_df, output_dir)
    elif suite == "pool_allocator":
        return _generate_pool_allocator_split_charts(suite_df, output_dir)
    return []


def _parse_rw_lock_name(name: str):
    """Parse rw_lock benchmark name into (operation, mutex, threads, contention)."""
    clean = name.replace("BM_", "").replace("/real_time", "")

    mutex = None
    for m in ["NopMutex", "std::shared_mutex", "dispenso::RWLock"]:
        if m in clean:
            mutex = m
            clean = clean.replace(f"<{m}>", "")
            break

    parts = clean.split("/")
    operation = parts[0]

    if operation == "serial":
        threads = None
        contention = parts[1] if len(parts) > 1 else None
    else:
        threads = parts[1] if len(parts) > 1 else None
        contention = parts[2] if len(parts) > 2 else None

    return operation, mutex, threads, contention


def _render_rw_lock_parallel_chart(
    cont_df: "pd.DataFrame",
    cont: str,
    available_mutexes: list,
    thread_counts: list,
    colors: dict,
    output_dir: Path,
) -> Path:
    """Render a single rw_lock parallel chart for a given contention level."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 5))

    y = np.arange(len(thread_counts))
    height = 0.8 / len(available_mutexes)

    max_val = 0
    for i, mutex in enumerate(available_mutexes):
        values = []
        for threads in thread_counts:
            row = cont_df[(cont_df["mutex"] == mutex) & (cont_df["threads"] == threads)]
            values.append(row["real_time"].iloc[0] if not row.empty else 0)
        values = np.array(values)
        max_val = max(max_val, values.max())
        ax.barh(
            y + i * height - height * len(available_mutexes) / 2 + height / 2,
            values,
            height,
            label=mutex,
            color=colors.get(mutex, "#3498db"),
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"{t} threads" for t in thread_counts], fontsize=10)

    scale, display_unit = get_time_scale([max_val], "ns")
    ax.set_xlabel(f"Time ({display_unit})", fontsize=12)
    if scale != 1:
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p, s=scale: f"{x / s:.0f}")
        )

    ax.set_title(f"RW Lock - Parallel ({cont} iterations)", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / f"rw_lock_parallel_{cont}_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Generated: {chart_path}")
    return chart_path


def generate_rw_lock_charts(suite_df: "pd.DataFrame", output_dir: Path):
    """Generate horizontal grouped bar charts for rw_lock benchmark."""
    import numpy as np

    suite_df = suite_df.copy()
    parsed = suite_df["name"].apply(_parse_rw_lock_name)
    suite_df["operation"] = parsed.apply(lambda x: x[0])
    suite_df["mutex"] = parsed.apply(lambda x: x[1])
    suite_df["threads"] = parsed.apply(lambda x: x[2])
    suite_df["contention"] = parsed.apply(lambda x: x[3])

    charts = []
    mutexes = ["NopMutex", "std::shared_mutex", "dispenso::RWLock"]
    colors = {
        "NopMutex": "#95a5a6",
        "std::shared_mutex": "#e74c3c",
        "dispenso::RWLock": "#2ecc71",
    }

    # Serial chart
    serial_df = suite_df[suite_df["operation"] == "serial"]
    if not serial_df.empty:
        available_mutexes = [m for m in mutexes if m in serial_df["mutex"].values]
        contentions = sorted(
            serial_df["contention"].dropna().unique(), key=lambda x: int(x)
        )

        if available_mutexes and contentions:
            fig, ax = plt.subplots(figsize=(10, 5))

            y = np.arange(len(contentions))
            height = 0.8 / len(available_mutexes)

            for i, mutex in enumerate(available_mutexes):
                values = []
                for cont in contentions:
                    row = serial_df[
                        (serial_df["mutex"] == mutex)
                        & (serial_df["contention"] == cont)
                    ]
                    values.append(row["real_time"].iloc[0] if not row.empty else 0)
                values = np.array(values)
                ax.barh(
                    y + i * height - height * len(available_mutexes) / 2 + height / 2,
                    values,
                    height,
                    label=mutex,
                    color=colors.get(mutex, "#3498db"),
                )

            ax.set_yticks(y)
            ax.set_yticklabels([f"{c} iterations" for c in contentions], fontsize=10)
            ax.set_xlabel("Time (ms)", fontsize=12)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{x / 1e6:.1f}")
            )
            ax.set_title("RW Lock - Serial Operations", fontsize=14)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(axis="x", alpha=0.3)

            plt.tight_layout()
            chart_path = output_dir / "rw_lock_serial_chart.png"
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close()
            charts.append(chart_path)
            print(f"Generated: {chart_path}")

    # Parallel charts - one per contention level
    parallel_df = suite_df[suite_df["operation"] == "parallel"]
    if not parallel_df.empty:
        available_mutexes = [m for m in mutexes if m in parallel_df["mutex"].values]
        available_mutexes = [m for m in available_mutexes if m != "NopMutex"]
        contentions = sorted(
            parallel_df["contention"].dropna().unique(), key=lambda x: int(x)
        )
        thread_counts = sorted(
            parallel_df["threads"].dropna().unique(), key=lambda x: int(x)
        )

        for cont in contentions:
            cont_df = parallel_df[parallel_df["contention"] == cont]
            if not cont_df.empty:
                chart_path = _render_rw_lock_parallel_chart(
                    cont_df, cont, available_mutexes, thread_counts, colors, output_dir
                )
                charts.append(chart_path)

    return charts


def generate_once_function_charts(suite_df: "pd.DataFrame", output_dir: Path):
    """Generate horizontal bar charts for once_function benchmark.

    Split into move vs queue operations, grouped by size.
    """
    import numpy as np

    # Parse benchmark names
    def parse_once_function_name(name: str):
        clean = name.replace("BM_", "").replace("/real_time", "")

        # Extract operation (move or queue)
        if clean.startswith("move_"):
            operation = "move"
            clean = clean[5:]
        elif clean.startswith("queue_"):
            operation = "queue"
            clean = clean[6:]
        else:
            operation = "unknown"

        # Extract function type and size
        if "<" in clean:
            func_type, size = clean.split("<")
            size = size.rstrip(">")
        else:
            func_type = clean
            size = "unknown"

        # Clean up function type name
        func_type = func_type.replace("_", " ").title()

        return operation, func_type, size

    suite_df = suite_df.copy()
    parsed = suite_df["name"].apply(parse_once_function_name)
    suite_df["operation"] = parsed.apply(lambda x: x[0])
    suite_df["func_type"] = parsed.apply(lambda x: x[1])
    suite_df["size"] = parsed.apply(lambda x: x[2])

    charts = []
    size_order = ["kSmallSize", "kMediumSize", "kLargeSize", "kExtraLargeSize"]
    size_labels = {
        "kSmallSize": "Small",
        "kMediumSize": "Medium",
        "kLargeSize": "Large",
        "kExtraLargeSize": "Extra Large",
    }
    colors = {
        "Std Function": "#e74c3c",
        "Once Function": "#2ecc71",
        "Inline Function": "#3498db",
    }

    for op in ["move", "queue"]:
        op_df = suite_df[suite_df["operation"] == op]
        if op_df.empty:
            continue

        func_types = sorted(op_df["func_type"].unique())
        sizes = [s for s in size_order if s in op_df["size"].values]

        if not func_types or not sizes:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        y = np.arange(len(sizes))
        height = 0.8 / len(func_types)

        max_val = 0
        for i, func_type in enumerate(func_types):
            values = []
            for size in sizes:
                row = op_df[(op_df["func_type"] == func_type) & (op_df["size"] == size)]
                values.append(row["real_time"].iloc[0] if not row.empty else 0)
            values = np.array(values)
            max_val = max(max_val, values.max())
            ax.barh(
                y + i * height - height * len(func_types) / 2 + height / 2,
                values,
                height,
                label=func_type,
                color=colors.get(func_type, "#9b59b6"),
            )

        ax.set_yticks(y)
        ax.set_yticklabels([size_labels.get(s, s) for s in sizes], fontsize=11)

        # Format time axis
        if max_val >= 1e6:
            ax.set_xlabel("Time (ms)", fontsize=12)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{x / 1e6:.1f}")
            )
        elif max_val >= 1e3:
            ax.set_xlabel("Time (µs)", fontsize=12)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{x / 1e3:.1f}")
            )
        else:
            ax.set_xlabel("Time (ns)", fontsize=12)

        ax.set_title(f"Once Function - {op.title()} Operations", fontsize=14)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        chart_path = output_dir / f"once_function_{op}_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        charts.append(chart_path)
        print(f"Generated: {chart_path}")

    return charts


def generate_pool_allocator_charts(suite_df: "pd.DataFrame", output_dir: Path):
    """Generate horizontal bar charts for pool_allocator benchmark.

    Split by thread count, grouped by allocator type and size.
    """
    import numpy as np

    # Parse benchmark names
    def parse_pool_alloc_name(name: str):
        clean = name.replace("BM_", "").replace("/real_time", "")

        # Extract allocator type
        if clean.startswith("mallocfree_threaded"):
            alloc_type = "malloc/free"
            rest = clean[19:]  # after "mallocfree_threaded"
        elif clean.startswith("mallocfree"):
            alloc_type = "malloc/free"
            rest = clean[10:]  # after "mallocfree"
        elif clean.startswith("nl_pool_allocator_arena"):
            alloc_type = "NoLock Arena"
            rest = clean[23:]
        elif clean.startswith("nl_pool_allocator"):
            alloc_type = "NoLock"
            rest = clean[17:]
        elif clean.startswith("pool_allocator_arena"):
            alloc_type = "Arena"
            rest = clean[20:]
        elif clean.startswith("pool_allocator_threaded"):
            alloc_type = "PoolAllocator"
            rest = clean[23:]
        elif clean.startswith("pool_allocator"):
            alloc_type = "PoolAllocator"
            rest = clean[14:]
        else:
            alloc_type = "unknown"
            rest = clean

        # Parse <Size,Threads>/iterations or <Size>/iterations
        size = "unknown"
        threads = 1
        iterations = 0

        if "<" in rest:
            params, iters = rest.split(">")
            params = params.lstrip("<")
            if "," in params:
                size, threads = params.split(",")
                threads = int(threads)
            else:
                size = params
            if "/" in iters:
                iterations = int(iters.lstrip("/"))
        elif "/" in rest:
            iterations = int(rest.lstrip("/"))

        return alloc_type, size, threads, iterations

    suite_df = suite_df.copy()
    parsed = suite_df["name"].apply(parse_pool_alloc_name)
    suite_df["alloc_type"] = parsed.apply(lambda x: x[0])
    suite_df["size"] = parsed.apply(lambda x: x[1])
    suite_df["threads"] = parsed.apply(lambda x: x[2])
    suite_df["iterations"] = parsed.apply(lambda x: x[3])

    charts = []
    size_order = ["kSmallSize", "kMediumSize", "kLargeSize"]
    size_labels = {"kSmallSize": "S", "kMediumSize": "M", "kLargeSize": "L"}

    # Colors for allocator types
    colors = {
        "malloc/free": "#e74c3c",
        "PoolAllocator": "#2ecc71",
        "Arena": "#27ae60",
        "NoLock": "#3498db",
        "NoLock Arena": "#2980b9",
    }

    # Group by thread count
    thread_counts = sorted(suite_df["threads"].unique())

    for threads in thread_counts:
        thread_df = suite_df[suite_df["threads"] == threads]
        if thread_df.empty:
            continue

        # For single-threaded, show all allocator types
        # For multi-threaded, show only threaded variants
        if threads == 1:
            alloc_types = [
                "malloc/free",
                "PoolAllocator",
                "Arena",
                "NoLock",
                "NoLock Arena",
            ]
            title_suffix = "Single-threaded"
        else:
            alloc_types = ["malloc/free", "PoolAllocator"]
            title_suffix = f"{threads} Threads"

        alloc_types = [a for a in alloc_types if a in thread_df["alloc_type"].values]
        iterations_list = sorted(thread_df["iterations"].unique())

        if not alloc_types:
            continue

        # Create grouped labels: Size x Iterations
        group_labels = []
        for size in size_order:
            for iters in iterations_list:
                if iters >= 1000:
                    iters_label = f"{iters // 1000}K"
                else:
                    iters_label = str(iters)
                group_labels.append(f"{size_labels.get(size, size)}/{iters_label}")

        fig, ax = plt.subplots(figsize=(12, max(6, len(group_labels) * 0.4)))

        y = np.arange(len(group_labels))
        height = 0.8 / len(alloc_types)

        max_val = 0
        for i, alloc_type in enumerate(alloc_types):
            values = []
            for size in size_order:
                for iters in iterations_list:
                    row = thread_df[
                        (thread_df["alloc_type"] == alloc_type)
                        & (thread_df["size"] == size)
                        & (thread_df["iterations"] == iters)
                    ]
                    values.append(row["real_time"].iloc[0] if not row.empty else 0)
            values = np.array(values)
            max_val = max(max_val, values.max())
            ax.barh(
                y + i * height - height * len(alloc_types) / 2 + height / 2,
                values,
                height,
                label=alloc_type,
                color=colors.get(alloc_type, "#9b59b6"),
            )

        ax.set_yticks(y)
        ax.set_yticklabels(group_labels, fontsize=10)

        # Format time axis
        if max_val >= 1e9:
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{x / 1e9:.1f}")
            )
        elif max_val >= 1e6:
            ax.set_xlabel("Time (ms)", fontsize=12)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{x / 1e6:.0f}")
            )
        elif max_val >= 1e3:
            ax.set_xlabel("Time (µs)", fontsize=12)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{x / 1e3:.0f}")
            )
        else:
            ax.set_xlabel("Time (ns)", fontsize=12)

        ax.set_title(f"Pool Allocator - {title_suffix}", fontsize=14)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        chart_path = output_dir / f"pool_allocator_{threads}t_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        charts.append(chart_path)
        print(f"Generated: {chart_path}")

    return charts


def generate_simple_horizontal_chart(
    suite_df: "pd.DataFrame", output_dir: Path, suite: str
):
    """Generate a simple horizontal bar chart for a benchmark suite."""
    import numpy as np

    suite_df = suite_df.copy()
    suite_df["display_name"] = suite_df["name"].apply(
        lambda x: x.replace("BM_", "").replace("/real_time", "").replace("_", " ")
    )
    suite_df = suite_df.sort_values("real_time")

    fig, ax = plt.subplots(figsize=(12, max(6, len(suite_df) * 0.5)))

    y = np.arange(len(suite_df))
    values = suite_df["real_time"].values
    labels = suite_df["display_name"].values

    bars = ax.barh(y, values, color="#3498db")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)

    max_val = values.max()
    time_unit = suite_df["time_unit"].iloc[0]
    if time_unit == "ns" and max_val >= 1e6:
        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x / 1e6:.1f}"))
    elif time_unit == "ns" and max_val >= 1e3:
        ax.set_xlabel("Time (µs)", fontsize=12)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x / 1e3:.0f}"))
    else:
        ax.set_xlabel(f"Time ({time_unit})", fontsize=12)

    ax.set_title(f"{suite.replace('_', ' ').title()} Benchmark", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / f"{suite}_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Generated: {chart_path}")

    return [chart_path]


def _parse_timed_task_row(row):
    """Parse a timed_task benchmark row into a dict with library, config, etc.

    Returns None if the row doesn't have the expected format.
    """
    name = row["name"]
    mean_err = row.get("mean", None)
    stddev_err = row.get("stddev", None)

    if mean_err is None or stddev_err is None:
        return None

    match = re.match(r"BM_(\w+)<(.+)>", name)
    if not match:
        return None

    func_name = match.group(1)
    template_args = match.group(2).strip()

    if "mixed" in func_name:
        library = func_name.replace("_mixed", "")
        steady = template_args.strip().lower() == "true"
        config_label = f"Mixed ({'Steady' if steady else 'Normal'})"
        period_ms = None
        is_mixed = True
    else:
        library = func_name
        is_mixed = False
        parts = [p.strip() for p in template_args.split(",")]
        if len(parts) != 2:
            return None
        try:
            period_ms = int(parts[0])
        except ValueError:
            return None
        steady = parts[1].lower() == "true"
        config_label = f"{period_ms}ms ({'Steady' if steady else 'Normal'})"

    return {
        "library": library,
        "config": config_label,
        "is_mixed": is_mixed,
        "steady": steady,
        "period_ms": period_ms,
        "mean_error_us": mean_err * 1e6,
        "stddev_us": stddev_err * 1e6,
    }


def _build_timed_task_config_order(timed_df: "pd.DataFrame") -> list:
    """Build the ordered list of config labels for timed_task charts."""
    config_order = []
    # Single-period configs sorted by period
    for steady in [False, True]:
        for period in sorted(
            timed_df[~timed_df["is_mixed"]]["period_ms"].dropna().unique()
        ):
            label = f"{int(period)}ms ({'Steady' if steady else 'Normal'})"
            if label in timed_df["config"].values:
                config_order.append(label)
    # Mixed configs
    for steady in [False, True]:
        label = f"Mixed ({'Steady' if steady else 'Normal'})"
        if label in timed_df["config"].values:
            config_order.append(label)
    return [c for c in config_order if c in timed_df["config"].values]


def generate_timed_task_charts(suite_df: "pd.DataFrame", output_dir: Path):
    """Generate bar chart of scheduling jitter for timed_task benchmark."""
    import numpy as np

    rows = []
    for _, row in suite_df.iterrows():
        parsed = _parse_timed_task_row(row)
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        print("  No timed_task data with mean/stddev counters found")
        return []

    timed_df = pd.DataFrame(rows)
    config_order = _build_timed_task_config_order(timed_df)

    libraries = sorted(timed_df["library"].unique())
    lib_colors = {lib: get_library_color(lib) for lib in libraries}

    fig, ax = plt.subplots(figsize=(12, max(6, len(config_order) * 0.8)))

    y = np.arange(len(config_order))
    bar_height = 0.8 / len(libraries)

    for i, lib in enumerate(libraries):
        means = []
        stddevs = []
        for config in config_order:
            row = timed_df[
                (timed_df["library"] == lib) & (timed_df["config"] == config)
            ]
            if not row.empty:
                means.append(row["mean_error_us"].iloc[0])
                stddevs.append(row["stddev_us"].iloc[0])
            else:
                means.append(0)
                stddevs.append(0)

        offset = i * bar_height - bar_height * len(libraries) / 2 + bar_height / 2
        ax.barh(
            y + offset,
            means,
            bar_height,
            xerr=stddevs,
            label=lib,
            color=lib_colors.get(lib, "#7f8c8d"),
            capsize=3,
            error_kw={"linewidth": 1},
        )

    ax.set_yticks(y)
    ax.set_yticklabels(config_order, fontsize=10)
    ax.set_xlabel("Mean Scheduling Error (µs)", fontsize=12)
    ax.set_title("Timed Task Scheduling Jitter", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / "timed_task_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Generated: {chart_path}")

    return [chart_path]


def generate_charts(df: "pd.DataFrame", output_dir: Path):
    """Generate comparison charts from benchmark data."""
    if df.empty:
        print("No data to plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse benchmark names to extract structured data
    parsed = df["name"].apply(parse_benchmark_name)
    df = df.copy()
    df["library"] = parsed.apply(lambda x: x["library"])
    df["threads"] = parsed.apply(lambda x: x["threads"])
    df["worksize"] = parsed.apply(lambda x: x["worksize"])

    # Suites with specialized chart generation (thread scaling line charts)
    specialized_thread_scaling = [
        "simple_for",
        "summing_for",
        "trivial_compute",
        "nested_for",
    ]

    # Group by suite
    for suite, suite_df in df.groupby("suite"):
        # Special handling for concurrent_vector suite (and tcmalloc variant)
        if suite == "concurrent_vector":
            generate_concurrent_vector_charts(suite_df, output_dir)
            continue
        if suite == "concurrent_vector_tcmalloc":
            generate_concurrent_vector_charts(suite_df, output_dir, suffix="_tcmalloc")
            continue

        # Special handling for future benchmark
        if suite == "future":
            generate_future_charts(suite_df, output_dir)
            continue

        # Special handling for graph benchmark
        if suite == "graph":
            generate_graph_charts(suite_df, output_dir)
            continue

        # Special handling for pool_allocator and small_buffer
        if suite == "pool_allocator":
            generate_pool_allocator_charts(suite_df, output_dir)
            continue
        if suite == "small_buffer":
            generate_allocator_charts(suite_df, output_dir, suite)
            continue

        # Special handling for rw_lock
        if suite == "rw_lock":
            generate_rw_lock_charts(suite_df, output_dir)
            continue

        # Special handling for once_function - split into move/queue charts
        if suite == "once_function":
            generate_once_function_charts(suite_df, output_dir)
            continue

        # Timed task: bar chart of scheduling jitter (mean error + stddev)
        if suite == "timed_task":
            generate_timed_task_charts(suite_df, output_dir)
            continue

        # Special handling for simple_for, summing_for, trivial_compute
        if suite in specialized_thread_scaling:
            generate_simple_for_charts(suite_df, output_dir, suite)
            continue

        # Determine chart type based on data structure
        has_thread_data = suite_df["threads"].notna().any()
        worksizes = suite_df["worksize"].dropna().unique()

        if has_thread_data and len(worksizes) > 0:
            # Generate line charts for each worksize (no max_threads chart)
            for worksize in sorted(worksizes, key=lambda x: (str(x).isdigit(), x)):
                chart_path = generate_line_chart(suite, suite_df, output_dir, worksize)
                if chart_path:
                    print(f"Generated: {chart_path}")
        else:
            # Simple bar chart for benchmarks without thread/worksize structure
            chart_path = generate_bar_chart(suite, suite_df, output_dir)
            print(f"Generated: {chart_path}")


def _format_worksize(worksize) -> str:
    """Format a numeric worksize for display (e.g., 1000000 -> '1M')."""
    if not str(worksize).isdigit():
        return str(worksize)
    ws_int = int(worksize)
    if ws_int >= 1000000000:
        return f"{ws_int // 1000000000}B"
    elif ws_int >= 1000000:
        return f"{ws_int // 1000000}M"
    elif ws_int >= 1000:
        return f"{ws_int // 1000}K"
    return str(ws_int)


# Registry mapping suite names to their chart entries.
# Each entry is (caption, chart_filename) or None for entries that need dynamic generation.
_SUITE_CHART_REGISTRY = {
    "concurrent_vector": [
        ("**Serial/Access Operations:**", "concurrent_vector_serial_chart.png"),
        ("**Parallel Operations:**", "concurrent_vector_parallel_chart.png"),
    ],
    "concurrent_vector_tcmalloc": [
        (
            "**Serial/Access Operations (tcmalloc):**",
            "concurrent_vector_serial_tcmalloc_chart.png",
        ),
        (
            "**Parallel Operations (tcmalloc):**",
            "concurrent_vector_parallel_tcmalloc_chart.png",
        ),
    ],
    "future": [
        ("**Full comparison (including std::async):**", "future_chart.png"),
        ("**Zoomed (excluding std::async):**", "future_zoomed_chart.png"),
    ],
    "graph": [
        (None, "graph_chart.png"),
    ],
    "small_buffer": [
        (None, "small_buffer_chart.png"),
    ],
    "pool_allocator": [
        ("**Single-threaded:**", "pool_allocator_1t_chart.png"),
        ("**2 Threads:**", "pool_allocator_2t_chart.png"),
        ("**8 Threads:**", "pool_allocator_8t_chart.png"),
        ("**16 Threads:**", "pool_allocator_16t_chart.png"),
    ],
    "once_function": [
        ("**Move Operations:**", "once_function_move_chart.png"),
        ("**Queue Operations:**", "once_function_queue_chart.png"),
    ],
    "rw_lock": [
        ("**Serial Operations:**", "rw_lock_serial_chart.png"),
        ("**Parallel Operations:**", None),  # sentinel for per-contention charts
    ],
    "timed_task": [
        ("**Scheduling Jitter (Mean Error ± Stddev):**", "timed_task_chart.png"),
    ],
}

# Suites that use worksize-based thread scaling charts (with zoomed variants)
_THREAD_SCALING_SUITES = {"simple_for", "summing_for", "trivial_compute", "nested_for"}

# RW lock contention levels for parallel sub-charts
_RW_LOCK_CONTENTIONS = [2, 8, 32, 128, 512]


def _write_suite_detail_table(suite: str, suite_df: "pd.DataFrame", output_dir: Path):
    """Write a per-suite detailed results table to a separate markdown file."""
    detail_path = output_dir / f"{suite}_details.md"
    with open(detail_path, "w") as detail_f:
        detail_f.write(f"# {suite} - Detailed Results\n\n")
        detail_f.write("| Benchmark | Time | Unit | Iterations |\n")
        detail_f.write("|-----------|------|------|------------|\n")
        for _, row in suite_df.iterrows():
            detail_f.write(
                f"| {row['name']} | {row['real_time']:.2f} | "
                f"{row['time_unit']} | {row['iterations']} |\n"
            )


def _write_suite_charts_markdown(f, suite: str, suite_parsed: "pd.DataFrame"):
    """Write chart markdown for a single suite to the report file."""
    # Check for registry entry first
    if suite in _SUITE_CHART_REGISTRY:
        for caption, chart_file in _SUITE_CHART_REGISTRY[suite]:
            if suite == "rw_lock" and chart_file is None:
                # Emit per-contention parallel charts
                for cont in _RW_LOCK_CONTENTIONS:
                    f.write(f"**{cont} Iterations:**\n\n")
                    f.write(
                        f"![rw_lock parallel {cont}](rw_lock_parallel_{cont}_chart.png)\n\n"
                    )
                continue
            if caption:
                f.write(f"{caption}\n\n")
            alt = suite.replace("_", " ")
            f.write(f"![{alt}]({chart_file})\n\n")
        return

    # Thread scaling suites with zoomed charts
    if suite in _THREAD_SCALING_SUITES:
        worksizes = suite_parsed["worksize"].dropna().unique()
        numeric_worksizes = [w for w in worksizes if str(w).isdigit()]
        for worksize in sorted(numeric_worksizes, key=lambda x: int(x)):
            ws_display = _format_worksize(worksize)
            f.write(f"**{ws_display} elements:**\n\n")
            f.write(f"![{suite} {ws_display}]({suite}_{worksize}_chart.png)\n\n")
            f.write(f"**{ws_display} elements (Y-Axis Zoomed):**\n\n")
            f.write(
                f"![{suite} {ws_display} zoomed]({suite}_{worksize}_zoomed_chart.png)\n\n"
            )
        return

    # Generic fallback: worksize line charts or single bar chart
    has_thread_data = suite_parsed["threads"].notna().any()
    worksizes = suite_parsed["worksize"].dropna().unique()

    if has_thread_data and len(worksizes) > 0:
        for worksize in sorted(
            worksizes,
            key=lambda x: (str(x).isdigit(), int(x) if str(x).isdigit() else 0),
        ):
            ws_display = _format_worksize(worksize)
            chart_path = f"{suite}_{worksize}_chart.png"
            f.write(f"**{ws_display} elements:**\n\n")
            f.write(f"![{suite} {ws_display}]({chart_path})\n\n")
            if suite == "nested_pool" and str(worksize) == "1000000":
                f.write(
                    "*Note: folly::CPUThreadPoolExecutor is excluded from "
                    "the 1M chart as it fails to complete (likely due to "
                    "memory exhaustion from creating too many futures).*\n\n"
                )
    else:
        chart_path = f"{suite}_chart.png"
        f.write(f"![{suite} results]({chart_path})\n\n")


def generate_markdown_report(
    machine_info: Dict[str, Any],
    results: List[Dict],
    df: "pd.DataFrame",
    output_dir: Path,
):
    """Generate a markdown report of benchmark results."""
    report_path = output_dir / "benchmark_report.md"

    with open(report_path, "w") as f:
        f.write("# Dispenso Benchmark Results\n\n")

        # Machine info
        f.write("## Machine Information\n\n")
        f.write(f"- **Date**: {machine_info.get('timestamp', 'unknown')}\n")
        f.write(
            f"- **Platform**: {machine_info.get('platform', 'unknown')} "
            f"{machine_info.get('platform_release', '')}\n"
        )
        f.write(f"- **CPU**: {machine_info.get('cpu_model', 'unknown')}\n")
        f.write(f"- **Hardware Threads**: {machine_info.get('cpu_cores', 'unknown')}\n")
        f.write(f"- **Memory**: {machine_info.get('memory_gb', 'unknown')} GB\n")
        compiler = machine_info.get("compiler", {})
        if compiler:
            f.write(f"- **Compiler**: {compiler.get('compiler_summary', 'unknown')}\n")
        f.write("\n")

        # Results summary
        f.write("## Results Summary\n\n")

        successful = sum(1 for r in results if r.get("success"))
        f.write(f"- **Benchmarks run**: {len(results)}\n")
        f.write(f"- **Successful**: {successful}\n")
        f.write(f"- **Failed**: {len(results) - successful}\n\n")

        # Per-suite results with links to detailed tables
        if not df.empty:
            parsed = df["name"].apply(parse_benchmark_name)
            df_with_parsed = df.copy()
            df_with_parsed["threads"] = parsed.apply(lambda x: x["threads"])
            df_with_parsed["worksize"] = parsed.apply(lambda x: x["worksize"])

            for suite in df["suite"].unique():
                f.write(f"### {suite}\n\n")

                suite_df = df[df["suite"] == suite].sort_values("real_time")
                suite_parsed = df_with_parsed[df_with_parsed["suite"] == suite]

                _write_suite_detail_table(suite, suite_df, output_dir)
                _write_suite_charts_markdown(f, suite, suite_parsed)
                f.write(f"[View detailed results table]({suite}_details.md)\n\n")

        # Failures
        failures = [r for r in results if not r.get("success")]
        if failures:
            f.write("## Failures\n\n")
            for fail in failures:
                f.write(f"### {fail['name']}\n\n")
                f.write(f"```\n{fail.get('error', 'Unknown error')}\n```\n\n")

    print(f"Generated report: {report_path}")


def main():
    if not HAS_PLOTTING:
        import sys

        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Generate charts from dispenso benchmark results"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input JSON file from run_benchmarks.py",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("charts"),
        help="Output directory for charts and reports",
    )
    parser.add_argument(
        "--platform",
        "-p",
        type=str,
        default=None,
        help="Platform identifier (e.g., 'linux-threadripper', 'macos-m2', 'windows-zen4'). "
        "When set, output goes into a per-platform subdirectory under output-dir.",
    )

    args = parser.parse_args()

    # If platform specified, output into a subdirectory
    output_dir = args.output_dir
    if args.platform:
        output_dir = output_dir / args.platform

    # Load input data
    print(f"Loading results from: {args.input}")
    with open(args.input) as f:
        data = json.load(f)

    machine_info = data.get("machine_info", {})
    results = data.get("results", [])

    print(f"  CPU: {machine_info.get('cpu_model', 'unknown')}")
    print(f"  Hardware Threads: {machine_info.get('cpu_cores', 'unknown')}")
    compiler = machine_info.get("compiler", {})
    if compiler:
        print(f"  Compiler: {compiler.get('compiler_summary', 'unknown')}")
    if args.platform:
        print(f"  Platform: {args.platform}")
    print()

    # Extract benchmark data
    df = extract_benchmark_data(results)
    if df.empty:
        print("No benchmark data found in input file")
        import sys

        sys.exit(1)

    print(f"Found {len(df)} benchmark results across {df['suite'].nunique()} suites")
    print()

    # Generate charts
    generate_charts(df, output_dir)
    generate_markdown_report(machine_info, results, df, output_dir)

    print(f"\nAll charts saved to: {output_dir}")


if __name__ == "__main__":
    main()
