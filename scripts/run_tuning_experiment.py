#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Cross-platform tuning experiment for dispenso thread pool constants.

Builds and runs a representative set of benchmarks across different
tuning configurations, producing a JSON report for comparison. Run on
each target platform (Linux, macOS, Windows) to find optimal constants.

Usage:
    python3 run_tuning_experiment.py                    # all configs
    python3 run_tuning_experiment.py -c wake_factor     # single knob
    python3 run_tuning_experiment.py --list              # list configs

Output: JSON file with per-config benchmark results for comparison.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR.parent

# Representative benchmarks that exercise different scheduling patterns.
# Thread count filters use regex alternation — unmatched counts are silently
# skipped, so the same suite works on machines from 4 cores to 128+ cores.
# Each entry: (benchmark_executable, filter_pattern, description)
BENCHMARK_SUITE = [
    (
        "locality_benchmark",
        "BM_dispenso_static/(4|8|16|64|128)/4000000/",
        "locality (kStatic, 4M)",
    ),
    (
        "simple_for_benchmark",
        "BM_dispenso/(4|8|16|64|128)/1000000/",
        "simple_for (kAuto, 1M)",
    ),
    ("tree_benchmark", "BM_dispenso.*(kSmallSize|kMediumSize)>", "tree building"),
    (
        "simple_pool_benchmark",
        "BM_dispenso_bulk/(4|8|16|64|128)/1000/",
        "pool bulk (1K tasks)",
    ),
    (
        "nested_pool_benchmark",
        "BM_dispenso_bulk/(4|8|16|64|128)/10000/",
        "nested pool bulk (10K)",
    ),
    (
        "idle_pool_benchmark",
        "BM_dispenso_very_idle/(4|8|16|64|128)",
        "idle pool (CPU usage)",
    ),
    (
        "cascading_parallel_for_benchmark",
        "BM_dispenso.*(4|8|16|32|64|128)/100000/",
        "cascading (100K)",
    ),
    (
        "nested_for_benchmark",
        "BM_dispenso/(4|8|16|64|128)/500/",
        "nested_for (500 iters)",
    ),
    ("graph_scene_benchmark", "BM_scene_graph", "graph scene"),
]


def define_configurations():
    """Define tuning configurations to test.

    Each config is a dict with:
      name: human-readable name
      category: which knob is being tuned
      defines: list of -D flags to pass to cmake CXX_FLAGS
    """
    configs = []

    # Baseline: current defaults (no overrides)
    configs.append({"name": "baseline", "category": "baseline", "defines": []})

    # --- Wake branch factor ---
    for bf in [2, 4, 8, 16]:
        configs.append(
            {
                "name": f"wake_bf_{bf}",
                "category": "wake_factor",
                "defines": [f"-DDISPENSO_TUNE_WAKE_BRANCH_FACTOR={bf}"],
            }
        )

    # --- Max spin timeout ---
    for us in [32, 64, 128, 256, 512]:
        configs.append(
            {
                "name": f"spin_max_{us}us",
                "category": "spin_timeout",
                "defines": [f"-DDISPENSO_TUNE_MAX_SPIN_US={us}"],
            }
        )

    # --- Min spin timeout ---
    for us in [1, 2, 4, 8]:
        configs.append(
            {
                "name": f"spin_min_{us}us",
                "category": "spin_timeout",
                "defines": [f"-DDISPENSO_TUNE_MIN_SPIN_US={us}"],
            }
        )

    # --- Spin check interval ---
    for interval in [16, 32, 64, 128, 256]:
        configs.append(
            {
                "name": f"spin_check_{interval}",
                "category": "spin_interval",
                "defines": [f"-DDISPENSO_TUNE_SPIN_CHECK_INTERVAL={interval}"],
            }
        )

    # --- Wake group size ---
    for gs in [8, 16, 32, 64]:
        configs.append(
            {
                "name": f"wake_group_{gs}",
                "category": "wake_group",
                "defines": [f"-DDISPENSO_TUNE_WAKE_GROUP_SIZE={gs}"],
            }
        )

    # --- Waiter subgroup size (threads sharing one futex) ---
    # Min 2: distributeBudget declares peerSlots[kWaiterSubgroupSize - 1].
    for ss in [2, 4, 8]:
        configs.append(
            {
                "name": f"waiter_sub_{ss}",
                "category": "waiter_subgroup",
                "defines": [f"-DDISPENSO_TUNE_WAITER_SUBGROUP_SIZE={ss}"],
            }
        )

    # --- Wake group + steal-ring sharing matched pairs ---
    # The most useful tuning sweep on a new platform: matches group size
    # to ring sharing (the configuration the production defaults use).
    # Mirrors the v1_g16/v2_g8s/v3_g8p/v4_g4p variants from the original
    # Linux sweep. Run with and without promote-seed for each group size.
    for g in [4, 8, 16, 32]:
        configs.append(
            {
                "name": f"wake_g{g}_sharing",
                "category": "wake_combo",
                "defines": [
                    f"-DDISPENSO_TUNE_WAKE_GROUP_SIZE={g}",
                    f"-DDISPENSO_TUNE_STEAL_RING_SHARING={g}",
                    "-DDISPENSO_TUNE_PROMOTE_SEED=0",
                ],
            }
        )
        configs.append(
            {
                "name": f"wake_g{g}_sharing_promote",
                "category": "wake_combo",
                "defines": [
                    f"-DDISPENSO_TUNE_WAKE_GROUP_SIZE={g}",
                    f"-DDISPENSO_TUNE_STEAL_RING_SHARING={g}",
                    "-DDISPENSO_TUNE_PROMOTE_SEED=1",
                ],
            }
        )

    # --- Promote-seed on/off (at default group size) ---
    for v in [0, 1]:
        configs.append(
            {
                "name": f"promote_seed_{v}",
                "category": "promote_seed",
                "defines": [f"-DDISPENSO_TUNE_PROMOTE_SEED={v}"],
            }
        )

    # --- Wake-all threshold (mainly relevant on macOS/Windows) ---
    # K above which `bumpAndWakeN` switches from looping wake-one to a
    # single wake-all. Linux's exact-K syscall makes this irrelevant
    # there; on macOS/Windows the optimal K depends on the per-syscall
    # cost ratio (see benchmarks/wake_cost_bench).
    for k in [2, 3, 4, 6, 8, 12, 16]:
        configs.append(
            {
                "name": f"wake_all_thresh_{k}",
                "category": "wake_all_threshold",
                "defines": [f"-DDISPENSO_TUNE_WAKE_ALL_THRESHOLD={k}"],
            }
        )

    return configs


def build_config(source_dir, build_dir, config, jobs, extra_cmake_args=None):
    """Build dispenso benchmarks with a specific tuning configuration."""
    defines = " ".join(config["defines"])
    cxx_flags = f"-DCMAKE_CXX_FLAGS={defines}" if defines else ""

    cmake_cmd = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DDISPENSO_BUILD_BENCHMARKS=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if cxx_flags:
        cmake_cmd.append(cxx_flags)
    if extra_cmake_args:
        cmake_cmd.extend(extra_cmake_args)

    result = subprocess.run(cmake_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Configure FAILED: {result.stderr[:200]}")
        return False

    build_cmd = [
        "cmake",
        "--build",
        str(build_dir),
        f"-j{jobs}",
        "--config",
        "Release",
    ]
    result = subprocess.run(build_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Build FAILED: {result.stderr[:200]}")
        return False

    return True


def run_benchmark(build_dir, bench_name, filter_pattern, repetitions=3):
    """Run a single benchmark and return parsed results."""
    # Search for the benchmark executable
    for search in [build_dir / "bin", build_dir / "benchmarks", build_dir]:
        exe = search / bench_name
        if exe.exists():
            break
    else:
        return {"error": f"Benchmark {bench_name} not found"}

    args = [
        str(exe),
        "--benchmark_format=json",
        f"--benchmark_filter={filter_pattern}",
        f"--benchmark_repetitions={repetitions}",
    ]

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            return {"error": result.stderr[:200]}
        data = json.loads(result.stdout)
        # Extract median values
        medians = {}
        for bm in data.get("benchmarks", []):
            if "_median" in bm["name"]:
                t = bm["real_time"]
                unit = bm.get("time_unit", "ns")
                if unit == "ns":
                    t /= 1000
                elif unit == "ms":
                    t *= 1000
                name = bm["name"].replace("/real_time_median", "")
                medians[name] = t
        return {"medians": medians}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except json.JSONDecodeError:
        return {"error": "JSON parse failed"}
    except Exception as e:
        return {"error": str(e)}


def run_config(build_dir, repetitions=3):
    """Run all benchmarks for a configuration."""
    results = {}
    for bench_name, filter_pattern, description in BENCHMARK_SUITE:
        print(f"    {description}...", end="", flush=True)
        result = run_benchmark(build_dir, bench_name, filter_pattern, repetitions)
        if "error" in result:
            print(f" FAIL ({result['error'][:60]})")
        else:
            count = len(result.get("medians", {}))
            print(f" OK ({count} tests)")
        results[bench_name] = result
    return results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run dispenso cross-platform tuning experiments",
    )
    parser.add_argument(
        "--source-dir",
        "-s",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Dispenso source directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(f"tuning_results_{platform.system().lower()}.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        default=None,
        help="Only run configs in this category (e.g., wake_factor, spin_timeout)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available configurations and exit",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=os.cpu_count() or 8,
        help="Parallel build jobs",
    )
    parser.add_argument(
        "--repetitions",
        "-r",
        type=int,
        default=3,
        help="Benchmark repetitions per config",
    )
    parser.add_argument(
        "--cmake-args",
        action="append",
        default=None,
        help="Extra cmake arguments (e.g., --cmake-args='-DDISPENSO_DEPS_DIR=...')",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directory between configs (slower but avoids stale artifacts)",
    )
    return parser.parse_args()


def list_configurations(configs):
    """Print available configurations grouped by category."""
    categories = {}
    for c in configs:
        categories.setdefault(c["category"], []).append(c)
    for cat, cfgs in sorted(categories.items()):
        print(f"\n{cat}:")
        for c in cfgs:
            defines = " ".join(c["defines"]) if c["defines"] else "(defaults)"
            print(f"  {c['name']:30s} {defines}")


def run_experiment(args, configs):
    """Build and benchmark each configuration, returning the collected results."""
    all_results = {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "timestamp": datetime.now().isoformat(),
        "configs": [],
    }

    base_build_dir = Path(tempfile.gettempdir()) / "dispenso-tuning"

    for i, config in enumerate(configs):
        print(f"[{i + 1}/{len(configs)}] {config['name']}")
        defines_str = " ".join(config["defines"]) if config["defines"] else "(defaults)"
        print(f"  Defines: {defines_str}")

        build_dir = base_build_dir / config["name"]

        if args.clean and build_dir.exists():
            import shutil

            shutil.rmtree(build_dir)

        print("  Building...", flush=True)
        if not build_config(
            args.source_dir.resolve(), build_dir, config, args.jobs, args.cmake_args
        ):
            all_results["configs"].append({"config": config, "error": "build failed"})
            continue

        print("  Running benchmarks:")
        results = run_config(build_dir, args.repetitions)

        all_results["configs"].append({"config": config, "results": results})
        print()

    return all_results


def compute_deltas(results, baseline):
    """Compute percentage deltas between a config's results and the baseline."""
    deltas = []
    for bench_name, bench_result in results.items():
        base_result = baseline.get(bench_name, {})
        base_medians = base_result.get("medians", {})
        curr_medians = bench_result.get("medians", {})
        for key in set(base_medians) & set(curr_medians):
            if base_medians[key] > 0:
                delta = (
                    (curr_medians[key] - base_medians[key]) / base_medians[key] * 100
                )
                deltas.append(delta)
    return deltas


def print_summary(all_results):
    """Print a comparison table of all configs against the baseline."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    baseline = None
    for entry in all_results["configs"]:
        if entry["config"]["name"] == "baseline":
            baseline = entry.get("results", {})
            break

    if not baseline:
        print("No baseline results found")
        return

    for entry in all_results["configs"]:
        config = entry["config"]
        results = entry.get("results")
        if not results or "error" in entry:
            print(f"\n{config['name']}: FAILED")
            continue

        deltas = compute_deltas(results, baseline)
        if deltas:
            avg = sum(deltas) / len(deltas)
            improved = sum(1 for d in deltas if d < -5)
            regressed = sum(1 for d in deltas if d > 5)
            sign = "+" if avg > 0 else ""
            print(
                f"  {config['name']:30s} avg {sign}{avg:5.1f}%  "
                f"({improved} improved, {regressed} regressed, {len(deltas)} total)"
            )


def main():
    args = parse_args()
    configs = define_configurations()

    if args.list:
        list_configurations(configs)
        return

    if args.category:
        configs = [c for c in configs if c["category"] in (args.category, "baseline")]
        if not configs:
            print(f"No configs found for category '{args.category}'")
            sys.exit(1)

    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Source:   {args.source_dir}")
    print(f"Configs:  {len(configs)}")
    print(f"Reps:     {args.repetitions}")
    print()

    all_results = run_experiment(args, configs)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {args.output}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
