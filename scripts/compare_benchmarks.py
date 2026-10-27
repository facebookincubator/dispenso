#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Compare dispenso benchmark results across multiple runs.

Shows side-by-side timing and percentage change for each benchmark test.
Useful for A/B testing optimization changes.

Usage:
    python scripts/compare_benchmarks.py baseline.json after.json
    python scripts/compare_benchmarks.py baseline.json v2.json v3.json --filter concurrent_vector
    python scripts/compare_benchmarks.py baseline.json after.json --prefix BM_dispenso
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# A test is identified by (executable, benchmark name). Benchmark names alone
# are not unique: e.g. BM_omp/1/100000/real_time exists in both
# cascading_parallel_for_benchmark and locality_benchmark.
TestKey = Tuple[str, str]


def format_time(ns: float) -> str:
    """Format nanoseconds into a human-readable string."""
    if ns < 1000:
        return f"{ns:.0f}ns"
    elif ns < 1e6:
        return f"{ns / 1e3:.1f}us"
    elif ns < 1e9:
        return f"{ns / 1e6:.1f}ms"
    else:
        return f"{ns / 1e9:.2f}s"


def _pct_change(val: float, base_val: float) -> Optional[float]:
    """Return percentage change from base_val to val, or None if base is zero."""
    if base_val == 0:
        return None
    return ((val - base_val) / base_val) * 100


_TIME_UNIT_MULTIPLIERS = {"us": 1000, "ms": 1e6, "s": 1e9}


def _normalize_to_ns(time_val: float, unit: str) -> float:
    """Convert a time value to nanoseconds based on its unit string."""
    return time_val * _TIME_UNIT_MULTIPLIERS.get(unit, 1)


def normalize_exe_name(name: str) -> str:
    """Reduce a recorded executable name to a platform-independent identity.

    Drops any directory (either separator) and a trailing ".exe", so Windows
    and POSIX runs of the same benchmark compare against each other.
    """
    base = re.split(r"[\\/]", name)[-1]
    if base.lower().endswith(".exe"):
        base = base[: -len(".exe")]
    return base


def format_test(key: TestKey) -> str:
    """Render a test key for display."""
    return f"{key[0]}: {key[1]}"


def load_benchmarks(
    path: Path,
    file_filter: Optional[str] = None,
) -> Tuple[Dict[TestKey, float], Dict[TestKey, str]]:
    """Load benchmark results.

    Returns ({(executable, test_name): time_ns}, {(executable, test_name):
    error_message}). Rows Google Benchmark marks with error_occurred carry a
    real_time of 0, which is not a measurement; they are left out of the
    timings and reported as errors instead, unless the same test also has a
    valid (e.g. median) result.

    When --benchmark_repetitions is used, Google Benchmark outputs both
    individual iteration rows (run_type=iteration) and aggregate rows
    (run_type=aggregate) with names like "BM_foo_median", "BM_foo_mean", etc.
    We prefer the median aggregate when available, falling back to the
    individual iteration row.
    """
    with open(path) as f:
        data = json.load(f)

    iteration_results: Dict[TestKey, float] = {}
    median_results: Dict[TestKey, float] = {}
    errors: Dict[TestKey, str] = {}
    for entry in data.get("results", []):
        if not entry.get("success"):
            continue
        exe_name = normalize_exe_name(entry["name"])
        if file_filter and file_filter.lower() not in exe_name.lower():
            continue
        for bm in entry.get("data", {}).get("benchmarks", []):
            name = bm["name"]
            if bm.get("error_occurred"):
                run_name = bm.get("run_name", name)
                errors.setdefault(
                    (exe_name, run_name), bm.get("error_message") or "error"
                )
                continue
            time_ns = _normalize_to_ns(bm["real_time"], bm.get("time_unit", "ns"))

            run_type = bm.get("run_type")
            if run_type == "iteration":
                key = (exe_name, name)
                if key not in iteration_results:
                    iteration_results[key] = time_ns
            elif run_type == "aggregate" and name.endswith("_median"):
                base_name = name[: -len("_median")]
                median_results[(exe_name, base_name)] = time_ns

    # Prefer median aggregates over raw iteration values
    results = dict(iteration_results)
    results.update(median_results)
    errors = {k: v for k, v in errors.items() if k not in results}
    return results, errors


def get_compiler_info(path: Path) -> str:
    """Extract compiler summary from benchmark file."""
    with open(path) as f:
        data = json.load(f)
    mi = data.get("machine_info", {})
    compiler = mi.get("compiler", {})
    return compiler.get("compiler_summary", "unknown")


def _format_comparison_cell(
    val: Optional[float], base_val: Optional[float], errored: bool = False
) -> str:
    """Format one comparison column cell (time + percentage change)."""
    if errored:
        return f"  {'ERROR':>10} {'':>7}"
    if val is None:
        return f"  {'---':>10} {'':>7}"
    cell = f"  {format_time(val):>10}"
    if base_val is None:
        return cell + f" {'N/A':>7}"
    pct = _pct_change(val, base_val)
    if pct is None:
        return cell + f" {'N/A':>7}"
    marker = " <<<" if abs(pct) > 15 else ""
    return cell + f" {pct:>+6.1f}%{marker}"


def _print_summary(
    datasets: List[Dict[TestKey, float]],
    labels: List[str],
    tests: List[TestKey],
):
    """Print summary of significant changes (>10%) vs baseline."""
    base_label = labels[0]
    for i, label in enumerate(labels[1:], 1):
        improvements = []
        regressions = []
        for test in tests:
            base_val = datasets[0].get(test)
            val = datasets[i].get(test)
            if base_val is None or val is None:
                continue
            pct = _pct_change(val, base_val)
            if pct is None:
                continue
            if pct < -10:
                improvements.append((test, pct))
            elif pct > 10:
                regressions.append((test, pct))

        print(f"[{label}] vs [{base_label}] significant changes (>10%):")
        if improvements:
            for test, pct in sorted(improvements, key=lambda x: x[1]):
                print(f"  {pct:>+7.1f}%  {format_test(test)}")
        if regressions:
            for test, pct in sorted(regressions, key=lambda x: -x[1]):
                print(f"  {pct:>+7.1f}%  {format_test(test)}")
        if not improvements and not regressions:
            print("  (none)")
        print()


def _print_errors(
    error_sets: List[Dict[TestKey, str]],
    labels: List[str],
    tests: List[TestKey],
):
    """List tests that reported errors instead of a timing, per run."""
    shown = set(tests)
    for errors, label in zip(error_sets, labels):
        rows = sorted(k for k in errors if k in shown)
        if not rows:
            continue
        print(f"[{label}] errored (excluded from comparison):")
        for key in rows:
            print(f"  {format_test(key)}: {errors[key]}")
        print()


def compare(
    files: List[Path],
    labels: List[str],
    file_filter: Optional[str] = None,
    prefix_filter: Optional[str] = None,
):
    """Compare benchmark results across files."""
    loaded = [load_benchmarks(path, file_filter) for path in files]
    datasets = [results for results, _ in loaded]
    error_sets = [errors for _, errors in loaded]

    # Print headers with compiler info
    print("Runs:")
    for i, (path, label) in enumerate(zip(files, labels)):
        compiler = get_compiler_info(path)
        count = len(datasets[i])
        errored = len(error_sets[i])
        extra = f", {errored} errored" if errored else ""
        print(f"  [{label}] {path.name} ({compiler}, {count} tests{extra})")
    print()

    # Collect and filter test names
    all_tests = set()
    for ds in datasets + error_sets:
        all_tests.update(ds.keys())
    if prefix_filter:
        all_tests = {t for t in all_tests if t[1].startswith(prefix_filter)}

    if not all_tests:
        print("No matching tests found.")
        return

    tests = sorted(all_tests)
    name_width = min(max(len(t[1]) for t in tests), 55)

    # Print header
    base_label = labels[0]
    header = f"{'Test':<{name_width}}  {base_label:>10}"
    for label in labels[1:]:
        header += f"  {label:>10} {'chg%':>7}"
    print(header)
    print("-" * len(header))

    # Print each test row, grouped under the executable that produced it
    current_exe = None
    for test in tests:
        exe, test_name = test
        if exe != current_exe:
            print(f"[{exe}]")
            current_exe = exe
        name = (
            test_name
            if len(test_name) <= name_width
            else test_name[: name_width - 2] + ".."
        )
        base_val = datasets[0].get(test)

        line = f"{name:<{name_width}}"
        if test in error_sets[0]:
            line += f"  {'ERROR':>10}"
        elif base_val is not None:
            line += f"  {format_time(base_val):>10}"
        else:
            line += f"  {'---':>10}"
        for i in range(1, len(labels)):
            line += _format_comparison_cell(
                datasets[i].get(test), base_val, test in error_sets[i]
            )
        print(line)

    print()
    _print_errors(error_sets, labels, tests)
    _print_summary(datasets, labels, tests)


def main():
    parser = argparse.ArgumentParser(
        description="Compare dispenso benchmark results across runs"
    )
    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="Benchmark JSON files (first is baseline, rest are compared against it)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated labels for each file (default: base,v2,v3,...)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter to benchmark files matching this substring (e.g., 'concurrent_vector')",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filter to test names starting with this prefix (e.g., 'BM_dispenso')",
    )

    args = parser.parse_args()

    for f in args.files:
        if not f.exists():
            print(f"Error: {f} not found")
            sys.exit(1)

    if args.labels:
        labels = args.labels.split(",")
        if len(labels) != len(args.files):
            print(f"Error: {len(labels)} labels for {len(args.files)} files")
            sys.exit(1)
    else:
        labels = ["base"] + [f"v{i + 1}" for i in range(len(args.files) - 1)]

    compare(args.files, labels, args.filter, args.prefix)


if __name__ == "__main__":
    main()
