#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark runner and chart generator for dispenso.

This script runs the dispenso benchmarks, collects results, and generates
comparison charts. Results include machine information for reproducibility.

Usage:
    python run_benchmarks.py [--output-dir OUTPUT] [--benchmarks PATTERN]

Requirements:
    pip install matplotlib pandas
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    import pandas as pd

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/pandas not available. Charts will not be generated.")
    print("Install with: pip install matplotlib pandas")


def get_machine_info() -> Dict[str, Any]:
    """Gather machine information for benchmark context."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    # Try to get more detailed CPU info
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            # Extract model name
            match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
            if match:
                info["cpu_model"] = match.group(1).strip()
            # Count cores
            info["cpu_cores"] = cpuinfo.count("processor\t:")
            # Get cache info
            match = re.search(r"cache size\s*:\s*(.+)", cpuinfo)
            if match:
                info["cache_size"] = match.group(1).strip()
        except Exception as e:
            info["cpu_info_error"] = str(e)

        # Memory info
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
            match = re.search(r"MemTotal:\s*(\d+)", meminfo)
            if match:
                info["memory_kb"] = int(match.group(1))
                info["memory_gb"] = round(int(match.group(1)) / 1024 / 1024, 1)
        except Exception:
            pass

    elif platform.system() == "Darwin":  # macOS
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                info["cpu_model"] = result.stdout.strip()

            result = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True
            )
            if result.returncode == 0:
                info["cpu_cores"] = int(result.stdout.strip())

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            )
            if result.returncode == 0:
                mem_bytes = int(result.stdout.strip())
                info["memory_gb"] = round(mem_bytes / 1024 / 1024 / 1024, 1)
        except Exception as e:
            info["cpu_info_error"] = str(e)

    elif platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            info["cpu_model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            winreg.CloseKey(key)
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["wmic", "computersystem", "get", "totalphysicalmemory"],
                capture_output=True,
                text=True,
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                mem_bytes = int(lines[1].strip())
                info["memory_gb"] = round(mem_bytes / 1024 / 1024 / 1024, 1)
        except Exception:
            pass

        info["cpu_cores"] = os.cpu_count()

    return info


def find_benchmarks(build_dir: Path, pattern: Optional[str] = None) -> List[Path]:
    """Find benchmark executables in the build directory."""
    benchmarks = []

    # Common locations
    search_paths = [
        build_dir / "bin",
        build_dir / "benchmarks",
        build_dir,
    ]

    for search_path in search_paths:
        if not search_path.exists():
            continue

        for f in search_path.iterdir():
            if not f.is_file():
                continue
            name = f.name
            if "benchmark" in name.lower():
                if pattern is None or re.search(pattern, name):
                    benchmarks.append(f)

    return sorted(benchmarks)


def run_benchmark(benchmark_path: Path, extra_args: List[str] = None) -> Dict[str, Any]:
    """Run a single benchmark and return results."""
    args = [str(benchmark_path), "--benchmark_format=json"]
    if extra_args:
        args.extend(extra_args)

    print(f"Running: {benchmark_path.name}...")

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per benchmark
        )

        if result.returncode != 0:
            return {
                "name": benchmark_path.name,
                "success": False,
                "error": result.stderr,
            }

        # Parse JSON output
        # google benchmark outputs JSON to stdout
        try:
            data = json.loads(result.stdout)
            return {
                "name": benchmark_path.name,
                "success": True,
                "data": data,
            }
        except json.JSONDecodeError:
            # Maybe it's not a google benchmark, try to parse stdout
            return {
                "name": benchmark_path.name,
                "success": True,
                "raw_output": result.stdout,
            }

    except subprocess.TimeoutExpired:
        return {
            "name": benchmark_path.name,
            "success": False,
            "error": "Timeout after 600 seconds",
        }
    except Exception as e:
        return {
            "name": benchmark_path.name,
            "success": False,
            "error": str(e),
        }


def extract_benchmark_data(results: List[Dict]) -> pd.DataFrame:
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
            }

            # Extract any custom counters
            for key, value in bm.items():
                if key not in row and isinstance(value, (int, float)):
                    row[key] = value

            rows.append(row)

    return pd.DataFrame(rows)


def generate_charts(df: pd.DataFrame, output_dir: Path):
    """Generate comparison charts from benchmark data."""
    if df.empty:
        print("No data to plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by suite
    for suite, suite_df in df.groupby("suite"):
        fig, ax = plt.subplots(figsize=(12, 6))

        # Bar chart of real_time
        suite_df_sorted = suite_df.sort_values("real_time")
        bars = ax.barh(suite_df_sorted["name"], suite_df_sorted["real_time"])

        ax.set_xlabel(f"Time ({suite_df_sorted['time_unit'].iloc[0]})")
        ax.set_title(f"{suite} Benchmark Results")
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, suite_df_sorted["real_time"]):
            ax.text(
                val,
                bar.get_y() + bar.get_height() / 2,
                f" {val:.2f}",
                va="center",
                fontsize=8,
            )

        plt.tight_layout()
        chart_path = output_dir / f"{suite}_chart.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Generated: {chart_path}")

    # Generate summary chart if multiple suites
    if df["suite"].nunique() > 1:
        # This would need custom logic based on what comparisons make sense
        pass


def generate_markdown_report(
    machine_info: Dict[str, Any],
    results: List[Dict],
    df: pd.DataFrame,
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
        f.write(f"- **Cores**: {machine_info.get('cpu_cores', 'unknown')}\n")
        f.write(f"- **Memory**: {machine_info.get('memory_gb', 'unknown')} GB\n")
        f.write("\n")

        # Results summary
        f.write("## Results Summary\n\n")

        successful = sum(1 for r in results if r.get("success"))
        f.write(f"- **Benchmarks run**: {len(results)}\n")
        f.write(f"- **Successful**: {successful}\n")
        f.write(f"- **Failed**: {len(results) - successful}\n\n")

        # Per-suite results
        if not df.empty:
            for suite in df["suite"].unique():
                f.write(f"### {suite}\n\n")

                suite_df = df[df["suite"] == suite].sort_values("real_time")
                f.write("| Benchmark | Time | Unit | Iterations |\n")
                f.write("|-----------|------|------|------------|\n")

                for _, row in suite_df.iterrows():
                    f.write(
                        f"| {row['name']} | {row['real_time']:.2f} | "
                        f"{row['time_unit']} | {row['iterations']} |\n"
                    )

                f.write("\n")

                # Link to chart if it exists
                chart_path = f"{suite}_chart.png"
                f.write(f"![{suite} results]({chart_path})\n\n")

        # Failures
        failures = [r for r in results if not r.get("success")]
        if failures:
            f.write("## Failures\n\n")
            for fail in failures:
                f.write(f"### {fail['name']}\n\n")
                f.write(f"```\n{fail.get('error', 'Unknown error')}\n```\n\n")

    print(f"Generated report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run dispenso benchmarks")
    parser.add_argument(
        "--build-dir",
        "-b",
        type=Path,
        default=Path("build"),
        help="Build directory containing benchmark executables",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for results and charts",
    )
    parser.add_argument(
        "--benchmarks",
        "-B",
        type=str,
        default=None,
        help="Regex pattern to filter benchmarks",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only output JSON, skip chart generation",
    )

    args = parser.parse_args()

    # Gather machine info
    print("Gathering machine information...")
    machine_info = get_machine_info()
    print(f"  CPU: {machine_info.get('cpu_model', 'unknown')}")
    print(f"  Cores: {machine_info.get('cpu_cores', 'unknown')}")
    print(f"  Memory: {machine_info.get('memory_gb', 'unknown')} GB")
    print()

    # Find benchmarks
    benchmarks = find_benchmarks(args.build_dir, args.benchmarks)
    if not benchmarks:
        print(f"No benchmarks found in {args.build_dir}")
        print("Make sure you've built with -DDISPENSO_BUILD_BENCHMARKS=ON")
        sys.exit(1)

    print(f"Found {len(benchmarks)} benchmark(s):")
    for b in benchmarks:
        print(f"  - {b.name}")
    print()

    # Run benchmarks
    results = []
    for benchmark in benchmarks:
        result = run_benchmark(benchmark)
        results.append(result)
        if result["success"]:
            print(f"  ✓ {benchmark.name}")
        else:
            print(f"  ✗ {benchmark.name}: {result.get('error', 'unknown error')[:50]}")

    # Save raw results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "machine_info": machine_info,
        "results": results,
    }

    json_path = args.output_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nSaved raw results to: {json_path}")

    # Generate charts and report
    if not args.json_only and HAS_PLOTTING:
        df = extract_benchmark_data(results)
        if not df.empty:
            generate_charts(df, args.output_dir)
            generate_markdown_report(machine_info, results, df, args.output_dir)
        else:
            print(
                "No benchmark data to plot (benchmarks may not use google benchmark format)"
            )


if __name__ == "__main__":
    main()
