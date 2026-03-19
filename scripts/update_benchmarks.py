#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
One-stop-shop for updating benchmarks and documentation.

This script orchestrates the full benchmark workflow:
1. Configures and builds dispenso with benchmarks enabled
2. Runs benchmarks via run_benchmarks.py
3. Generates charts via generate_charts.py
4. Copies charts to docs/benchmarks/ with README-compatible names
5. Updates benchmark_results.md with detailed results

Multi-platform compose mode:
  Accepts pre-generated JSON results from multiple platforms and generates
  per-platform subdirectories with a unified landing page.

Usage:
    # Full run (build + run + charts) for current platform
    python update_benchmarks.py --platform linux-threadripper

    # Use existing build directory
    python update_benchmarks.py -b /path/to/build --skip-build --platform linux-threadripper

    # Run specific benchmarks only
    python update_benchmarks.py -B "concurrent_vector|pipeline"

    # Compose mode: generate docs from multiple platform results
    python update_benchmarks.py --compose \\
        linux-threadripper:results/linux.json \\
        macos-m2:results/macos.json \\
        windows-zen4:results/windows.json

Requirements:
    pip install matplotlib pandas
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
SOURCE_DIR = SCRIPT_DIR.parent
DOCS_DIR = SOURCE_DIR / "docs" / "benchmarks"
DEFAULT_BUILD_DIR = Path("/tmp/dispenso-benchmark-build")

# Chart name mapping: generated name -> README expected name
# The README.md references specific filenames that we need to maintain
README_CHART_MAPPING = {
    # future benchmark generates future_zoomed_chart.png, README expects par_tree_build.png
    "future_zoomed_chart.png": "par_tree_build.png",
    # simple_for 1M zoomed is used for README parallel for section (shows competitive cases)
    "simple_for_1000000_zoomed_chart.png": "simple_for.png",
    # pipeline chart
    "pipeline_chart.png": "pipeline.png",
    # concurrent_vector parallel chart maps to the README name
    "concurrent_vector_parallel_chart.png": "concurrent_vector.png",
    # nested_for charts map to README expected names
    "nested_for_10_chart.png": "nested_for_small.png",
    "nested_for_500_chart.png": "nested_for_medium.png",
    "nested_for_3000_chart.png": "nested_for_large.png",
}


def run_command(cmd: list, description: str, cwd: Path = None) -> bool:
    """Run a command and return success status."""
    print(f"  Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"  Error: {description} failed with code {result.returncode}")
        return False
    return True


def configure_build(build_dir: Path, source_dir: Path, jobs: int) -> bool:
    """Configure cmake build with benchmarks enabled."""
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake_cmd = [
        "cmake",
        str(source_dir),
        "-DDISPENSO_BUILD_BENCHMARKS=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_STANDARD=20",  # C++20 for folly and taskflow 3.x support
        "-DBENCHMARK_WITHOUT_FOLLY=OFF",
    ]

    print("  CMake configuration:")
    print("    - C++20 (for folly and taskflow support)")
    print("    - Release mode (-O3)")
    print("    - Folly enabled")
    print()

    return run_command(cmake_cmd, "CMake configure", cwd=build_dir)


def build_benchmarks(build_dir: Path, jobs: int) -> bool:
    """Build the benchmarks."""
    make_cmd = ["make", f"-j{jobs}"]
    return run_command(make_cmd, "Build", cwd=build_dir)


def copy_all_charts(charts_dir: Path, docs_dir: Path) -> int:
    """Copy all charts and markdown files from charts_dir to docs_dir.

    Returns count of files copied.
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    # Copy all PNG files
    for src_path in charts_dir.glob("*.png"):
        dst_path = docs_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        count += 1

    # Copy all markdown detail files
    for src_path in charts_dir.glob("*_details.md"):
        dst_path = docs_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        count += 1

    # Copy benchmark report as benchmark_results.md
    report_src = charts_dir / "benchmark_report.md"
    if report_src.exists():
        shutil.copy2(report_src, docs_dir / "benchmark_results.md")
        count += 1

    return count


def copy_readme_charts(charts_dir: Path, docs_dir: Path) -> list:
    """Copy charts with README-compatible names.

    Returns list of (src, dst) pairs that were copied.
    """
    copied = []

    for src_name, dst_name in README_CHART_MAPPING.items():
        src_path = charts_dir / src_name
        dst_path = docs_dir / dst_name
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            copied.append((src_name, dst_name))
            print(f"  Copied: {src_name} -> {dst_name}")
        else:
            print(f"  Warning: Source chart not found: {src_name}")

    return copied


def generate_landing_page(docs_dir: Path, platforms: list) -> Path:
    """Generate a landing page linking to all per-platform benchmark results.

    Args:
        docs_dir: The docs/benchmarks/ directory.
        platforms: List of dicts with keys: name, cpu, threads, memory, timestamp.

    Returns:
        Path to the generated landing page.
    """
    landing_path = docs_dir / "benchmark_results.md"

    with open(landing_path, "w") as f:
        f.write("# Dispenso Benchmark Results\n\n")
        f.write("Benchmark results across multiple platforms.\n\n")
        f.write("## Platforms\n\n")
        f.write("| Platform | CPU | Threads | Memory | Date |\n")
        f.write("|----------|-----|---------|--------|------|\n")

        for p in platforms:
            report_link = f"[{p['name']}]({p['name']}/benchmark_results.md)"
            f.write(
                f"| {report_link} | {p.get('cpu', '?')} | "
                f"{p.get('threads', '?')} | {p.get('memory', '?')} GB | "
                f"{p.get('timestamp', '?')} |\n"
            )

        f.write("\n---\n\n")
        f.write("Each platform directory contains:\n")
        f.write("- `benchmark_results.md` — Full benchmark report with charts\n")
        f.write("- `*_details.md` — Per-suite detailed result tables\n")
        f.write("- `*.png` — Individual chart images\n")

    print(f"Generated landing page: {landing_path}")
    return landing_path


def compose_platforms(platform_specs: list, docs_dir: Path) -> bool:
    """Compose multi-platform benchmark results.

    For each platform, runs generate_charts.py with --platform, then copies
    results to docs/benchmarks/<platform>/. Finally generates a landing page.

    Args:
        platform_specs: List of "platform_name:json_path" strings.
        docs_dir: Target docs/benchmarks/ directory.

    Returns:
        True on success.
    """
    platforms = []

    for spec in platform_specs:
        if ":" not in spec:
            print(
                f"Error: Invalid platform spec '{spec}'. Expected 'name:path/to/results.json'"
            )
            return False

        name, json_path_str = spec.split(":", 1)
        json_path = Path(json_path_str).resolve()

        if not json_path.exists():
            print(f"Error: Results file not found: {json_path}")
            return False

        # Read machine info from JSON
        with open(json_path) as f:
            data = json.load(f)
        machine_info = data.get("machine_info", {})

        platforms.append(
            {
                "name": name,
                "json_path": json_path,
                "cpu": machine_info.get("cpu_model", "unknown"),
                "threads": machine_info.get("cpu_cores", "?"),
                "memory": machine_info.get("memory_gb", "?"),
                "timestamp": machine_info.get("timestamp", "unknown"),
            }
        )

    print(f"Composing results for {len(platforms)} platform(s):")
    for p in platforms:
        print(f"  {p['name']}: {p['json_path']} ({p['cpu']}, {p['threads']} threads)")
    print()

    # Generate charts for each platform
    for p in platforms:
        print(f"Generating charts for {p['name']}...")
        print("-" * 40)

        charts_dir = docs_dir / p["name"]
        chart_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "generate_charts.py"),
            "-i",
            str(p["json_path"]),
            "-o",
            str(charts_dir),
        ]

        result = subprocess.run(chart_cmd)
        if result.returncode != 0:
            print(f"Error: Chart generation failed for {p['name']}")
            return False
        print()

    # Copy README-compatible charts from the first platform (primary)
    primary = platforms[0]
    primary_charts = docs_dir / primary["name"]
    print(f"Copying README charts from primary platform ({primary['name']})...")
    print("-" * 40)
    copied = copy_readme_charts(primary_charts, docs_dir)
    print(f"  Created {len(copied)} README-compatible chart names")
    print()

    # Generate landing page
    print("Generating landing page...")
    print("-" * 40)
    generate_landing_page(docs_dir, platforms)
    print()

    return True


def _step_build(build_dir: Path, source_dir: Path, jobs: int, skip: bool) -> bool:
    """Handle the build steps (configure + build). Returns True on success."""
    if skip:
        print("Step 1-2: Skipping build (--skip-build)")
        if not (build_dir / "bin").exists():
            print(f"Error: Build directory does not contain bin/: {build_dir}")
            return False
        print()
        return True

    print("Step 1: Configuring cmake...")
    print("-" * 40)
    if not configure_build(build_dir, source_dir, jobs):
        return False
    print()

    print("Step 2: Building benchmarks...")
    print("-" * 40)
    if not build_benchmarks(build_dir, jobs):
        return False
    print()
    return True


def _step_run(
    json_path: Path,
    build_dir: Path,
    benchmarks: str,
    min_time: float,
    skip: bool,
    step_num: int,
) -> bool:
    """Handle the benchmark run step. Returns True on success."""
    output_dir = json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if skip:
        print(f"Step {step_num}: Skipping benchmark run (--skip-run)")
        if not json_path.exists():
            print(f"Error: Results file not found: {json_path}")
            return False
        print()
        return True

    print(f"Step {step_num}: Running benchmarks...")
    print("-" * 40)

    run_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_benchmarks.py"),
        "-b",
        str(build_dir),
        "-o",
        str(json_path),
    ]
    if benchmarks:
        run_cmd.extend(["-B", benchmarks])
    if min_time:
        run_cmd.extend(["--min-time", str(min_time)])

    result = subprocess.run(run_cmd)
    if result.returncode != 0:
        print("Error: Benchmark run failed")
        return False
    print()
    return True


def _step_generate_charts(
    json_path: Path,
    charts_dir: Path,
    platform_id: str,
    skip: bool,
    step_num: int,
) -> bool:
    """Handle the chart generation step. Returns True on success."""
    if skip:
        print(f"Step {step_num}: Skipping chart generation (--skip-charts)")
        if not charts_dir.exists():
            print(f"Error: Charts directory not found: {charts_dir}")
            return False
        print()
        return True

    print(f"Step {step_num}: Generating charts...")
    print("-" * 40)

    chart_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "generate_charts.py"),
        "-i",
        str(json_path),
        "-o",
        str(charts_dir),
    ]
    if platform_id:
        chart_cmd.extend(["-p", platform_id])

    result = subprocess.run(chart_cmd)
    if result.returncode != 0:
        print("Error: Chart generation failed")
        return False
    print()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Update dispenso benchmarks and documentation"
    )
    parser.add_argument(
        "--build-dir",
        "-b",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help=f"Build directory (default: {DEFAULT_BUILD_DIR})",
    )
    parser.add_argument(
        "--source-dir",
        "-s",
        type=Path,
        default=SOURCE_DIR,
        help=f"Dispenso source directory (default: {SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Temporary output directory for results (default: build-dir/benchmark_output)",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=os.cpu_count() or 8,
        help=f"Number of parallel build jobs (default: {os.cpu_count() or 8})",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directory before building",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip cmake configure and build (use existing build)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running benchmarks (use existing results.json)",
    )
    parser.add_argument(
        "--skip-charts",
        action="store_true",
        help="Skip chart generation (use existing charts)",
    )
    parser.add_argument(
        "--benchmarks",
        "-B",
        type=str,
        default=None,
        help="Regex pattern to filter which benchmark executables to run",
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=None,
        help="Minimum time per benchmark in seconds",
    )
    parser.add_argument(
        "--platform",
        "-p",
        type=str,
        default=None,
        help="Platform identifier (e.g., 'linux-threadripper'). "
        "Charts go into docs/benchmarks/<platform>/ subdirectory.",
    )
    parser.add_argument(
        "--compose",
        nargs="+",
        metavar="PLATFORM:JSON_PATH",
        help="Compose mode: generate docs from multiple platform results. "
        "Each argument is 'platform-name:/path/to/results.json'. "
        "Example: --compose linux-threadripper:linux.json macos-m2:macos.json",
    )

    args = parser.parse_args()

    # Compose mode: just generate per-platform charts + landing page
    if args.compose:
        print("=" * 60)
        print("Dispenso Multi-Platform Benchmark Compose")
        print("=" * 60)
        print(f"Docs directory: {DOCS_DIR}")
        print()

        if not compose_platforms(args.compose, DOCS_DIR):
            sys.exit(1)

        print("=" * 60)
        print("Compose Complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Review the charts in docs/benchmarks/")
        print("  2. Verify README.md displays correctly")
        print("  3. Run: sl status")
        print("  4. Commit the updated charts")
        print()
        return

    # Normal mode: build + run + generate charts for a single platform
    build_dir = args.build_dir.resolve()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir or (build_dir / "benchmark_output")
    json_path = output_dir / "benchmark_results.json"
    charts_dir = output_dir / "charts"

    # When platform is set, docs go into a subdirectory
    docs_dir = DOCS_DIR
    if args.platform:
        docs_dir = DOCS_DIR / args.platform

    print("=" * 60)
    print("Dispenso Benchmark Update")
    print("=" * 60)
    print(f"Source directory: {source_dir}")
    print(f"Build directory:  {build_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Docs directory:   {docs_dir}")
    if args.platform:
        print(f"Platform:         {args.platform}")
    print(f"Parallel jobs:    {args.jobs}")
    print()

    # Step 0: Clean if requested
    if args.clean and build_dir.exists():
        print("Step 0: Cleaning build directory...")
        print("-" * 40)
        shutil.rmtree(build_dir)
        print(f"  Removed: {build_dir}")
        print()

    # Steps 1-2: Configure and build
    if not _step_build(build_dir, source_dir, args.jobs, args.skip_build):
        sys.exit(1)
    step_offset = 0 if args.skip_build else 2

    # Step 3: Run benchmarks
    if not _step_run(
        json_path,
        build_dir,
        args.benchmarks,
        args.min_time,
        args.skip_run,
        3 + step_offset,
    ):
        sys.exit(1)

    # Step 4: Generate charts
    if not _step_generate_charts(
        json_path,
        charts_dir,
        args.platform,
        args.skip_charts,
        4 + step_offset,
    ):
        sys.exit(1)

    # When --platform is used, charts are already in a platform subdir
    src_charts_dir = charts_dir
    if args.platform:
        src_charts_dir = charts_dir / args.platform

    # Step 5: Copy all charts to docs/benchmarks[/<platform>]/
    print(f"Step {5 + step_offset}: Copying all charts to {docs_dir}/...")
    print("-" * 40)

    file_count = copy_all_charts(src_charts_dir, docs_dir)
    print(f"  Copied {file_count} files (charts + markdown)")
    print()

    # Step 6: Copy README-specific charts with mapping
    print(f"Step {6 + step_offset}: Copying README charts with mapped names...")
    print("-" * 40)

    # README charts go to the top-level docs/benchmarks/ regardless of platform
    copied = copy_readme_charts(src_charts_dir, DOCS_DIR)
    print(f"  Created {len(copied)} README-compatible chart names")
    print()

    # Summary
    print("=" * 60)
    print("Update Complete!")
    print("=" * 60)
    print()
    print(f"Charts copied to {docs_dir}/:")
    for src, dst in copied:
        print(f"  {dst}")
    print()
    print("Next steps:")
    print("  1. Review the charts in docs/benchmarks/")
    print("  2. Verify README.md displays correctly")
    print("  3. Run: sl status")
    print("  4. Commit the updated charts")
    print()


if __name__ == "__main__":
    main()
