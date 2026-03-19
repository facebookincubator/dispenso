#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark runner for dispenso.

This script discovers and runs the dispenso benchmarks, collecting results
into a JSON file that includes machine information for reproducibility.

Usage:
    python run_benchmarks.py                          # run only (existing build)
    python run_benchmarks.py --build                  # build + run all
    python run_benchmarks.py --build -B pipeline      # build + run matching
    python run_benchmarks.py --build --cmake-args="-DCMAKE_CXX_COMPILER=g++-13"

The output JSON can then be processed by generate_charts.py to create
visualizations and reports.

Requirements:
    None (matplotlib/pandas only needed for generate_charts.py)
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR.parent
DEFAULT_BUILD_DIR = Path(tempfile.gettempdir()) / "dispenso-build"


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


def generate_platform_id(machine_info: Dict[str, Any]) -> str:
    """Generate a human-readable platform identifier from machine info.

    Examples: linux-threadripper-96c, macos-m4-12c, windows-zen4-24c
    """
    system = machine_info.get("platform", "unknown").lower()
    if system == "darwin":
        system = "macos"

    cpu_model = machine_info.get("cpu_model", "")
    cores = machine_info.get("cpu_cores", os.cpu_count() or 0)
    model_lower = cpu_model.lower()

    # Extract CPU family identifier
    cpu_id = "unknown"
    if system == "macos":
        # Apple Silicon: "Apple M4", "Apple M2 Pro", etc.
        match = re.search(r"apple\s+(m\d+(?:\s+\w+)?)", model_lower)
        if match:
            cpu_id = match.group(1).replace(" ", "-")
    elif "threadripper" in model_lower:
        cpu_id = "threadripper"
    elif "epyc" in model_lower:
        cpu_id = "epyc"
    elif "ryzen" in model_lower:
        match = re.search(r"ryzen\s+(\d+)\s+(\d{4})", model_lower)
        if match:
            cpu_id = f"ryzen{match.group(1)}-{match.group(2)}"
        else:
            cpu_id = "ryzen"
    elif "xeon" in model_lower:
        cpu_id = "xeon"
    elif re.search(r"core.*i\d", model_lower):
        match = re.search(r"i(\d+)-(\d+)", model_lower)
        if match:
            cpu_id = f"i{match.group(1)}-{match.group(2)}"
        else:
            cpu_id = "core"
    elif "zen" in model_lower:
        match = re.search(r"zen\s*(\d+)", model_lower)
        if match:
            cpu_id = f"zen{match.group(1)}"

    return f"{system}-{cpu_id}-{cores}c"


def _parse_cmake_cache(cache_path: Path, var_map: list) -> Dict[str, str]:
    """Parse CMakeCache.txt for specified variables.

    Args:
        cache_path: Path to CMakeCache.txt
        var_map: List of (cmake_var, label) tuples

    Returns:
        Dict mapping labels to their values.
    """
    info: Dict[str, str] = {}
    try:
        cache = cache_path.read_text()
        for var, label in var_map:
            match = re.search(rf"^{re.escape(var)}:\w+=(.+)", cache, re.MULTILINE)
            if match:
                val = match.group(1).strip()
                if val:
                    info[label] = val
    except Exception:
        pass
    return info


def _parse_compiler_cmake(cmake_files_dir: Path, var_map: list) -> Dict[str, str]:
    """Parse CMakeCXXCompiler.cmake for compiler ID and version.

    Args:
        cmake_files_dir: Path to CMakeFiles/ directory
        var_map: List of (cmake_var, label) tuples to look for

    Returns:
        Dict mapping labels to their values.
    """
    info: Dict[str, str] = {}
    if not cmake_files_dir.is_dir():
        return info
    for child in cmake_files_dir.iterdir():
        compiler_cmake = child / "CMakeCXXCompiler.cmake"
        if compiler_cmake.exists():
            try:
                text = compiler_cmake.read_text()
                for var, label in var_map:
                    match = re.search(rf'set\({re.escape(var)}\s+"([^"]+)"\)', text)
                    if match:
                        info[label] = match.group(1)
            except Exception:
                pass
            break
    return info


def _build_compiler_summary(info: Dict[str, str]) -> str:
    """Assemble a human-readable compiler summary string."""
    parts = []
    if "compiler_id" in info:
        parts.append(info["compiler_id"])
    if "compiler_version" in info:
        parts.append(info["compiler_version"])
    if "build_type" in info:
        parts.append(info["build_type"])
    if "cxx_standard" in info:
        parts.append(f"C++{info['cxx_standard']}")
    return " ".join(parts) if parts else ""


def get_compiler_info(build_dir: Path) -> Dict[str, str]:
    """Detect compiler and build settings from the CMake build directory."""
    # Locate the cmake build root (handle build_dir pointing at bin/)
    cmake_root = None
    for candidate in [build_dir, build_dir.parent]:
        if (candidate / "CMakeCache.txt").exists():
            cmake_root = candidate
            break

    if cmake_root is None:
        return {}

    # Parse CMakeCache.txt for build settings & compiler path
    cache_vars = [
        ("CMAKE_CXX_COMPILER", "compiler_path"),
        ("CMAKE_CXX_COMPILER_ID", "compiler_id"),
        ("CMAKE_CXX_COMPILER_VERSION", "compiler_version"),
        ("CMAKE_BUILD_TYPE", "build_type"),
        ("CMAKE_CXX_STANDARD", "cxx_standard"),
        ("CMAKE_CXX_FLAGS", "cxx_flags"),
    ]
    info = _parse_cmake_cache(cmake_root / "CMakeCache.txt", cache_vars)

    # Fall back to CMakeCXXCompiler.cmake for compiler ID & version
    if "compiler_id" not in info or "compiler_version" not in info:
        id_version_vars = [
            ("CMAKE_CXX_COMPILER_ID", "compiler_id"),
            ("CMAKE_CXX_COMPILER_VERSION", "compiler_version"),
        ]
        # Only fill missing values
        fallback = _parse_compiler_cmake(cmake_root / "CMakeFiles", id_version_vars)
        for label, val in fallback.items():
            if label not in info:
                info[label] = val

    summary = _build_compiler_summary(info)
    if summary:
        info["compiler_summary"] = summary

    return info


def _discover_benchmark_targets(build_dir: Path, pattern: str) -> List[str]:
    """Discover benchmark targets matching a regex pattern.

    Uses cmake --build --target help to list available targets, then filters
    for benchmark targets matching the pattern.  Returns an empty list if
    target discovery fails (caller should fall back to building all).
    """
    try:
        result = subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "help"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return []

        targets = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if "benchmark" not in line.lower():
                continue
            # cmake --target help format: "... target_name" (Makefiles)
            # or just "target_name" (Ninja)
            match = re.search(r"(?:\.\.\.\s+)?(\S*benchmark\S*)", line, re.IGNORECASE)
            if match:
                target = match.group(1)
                if re.search(pattern, target):
                    targets.append(target)
        return targets
    except Exception:
        return []


def configure_and_build(
    source_dir: Path,
    build_dir: Path,
    jobs: int,
    clean: bool = False,
    benchmark_filter: Optional[str] = None,
    extra_cmake_args: Optional[List[str]] = None,
) -> bool:
    """Configure and build dispenso benchmarks.

    Args:
        source_dir: Path to dispenso source root.
        build_dir: Path to cmake build directory.
        jobs: Number of parallel build jobs.
        clean: If True, wipe build_dir before configuring.
        benchmark_filter: If set, attempt to build only matching benchmark targets.
        extra_cmake_args: Additional cmake flags (e.g., -DCMAKE_CXX_COMPILER=g++-13).
            Pass -G "generator" here to override CMake's default generator.

    Returns:
        True on success, False on failure.
    """
    if clean and build_dir.exists():
        print(f"Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    # Configure
    cmake_cmd = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DDISPENSO_BUILD_BENCHMARKS=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_STANDARD=20",
    ]

    if extra_cmake_args:
        cmake_cmd.extend(extra_cmake_args)

    print("Configuring cmake...")
    print(f"  Source: {source_dir}")
    print(f"  Build:  {build_dir}")
    print(f"  Flags:  {' '.join(cmake_cmd[2:])}")
    sys.stdout.flush()

    result = subprocess.run(cmake_cmd)
    if result.returncode != 0:
        print("Error: cmake configure failed")
        return False

    # Build
    if benchmark_filter:
        targets = _discover_benchmark_targets(build_dir, benchmark_filter)
        if targets:
            print(f"Building {len(targets)} matching target(s): {', '.join(targets)}")
            sys.stdout.flush()
            for target in targets:
                target_cmd = [
                    "cmake",
                    "--build",
                    str(build_dir),
                    "--target",
                    target,
                    f"-j{jobs}",
                    "--config",
                    "Release",
                ]
                result = subprocess.run(target_cmd)
                if result.returncode != 0:
                    print(f"Error: build failed for target {target}")
                    return False
            return True
        else:
            print("No matching targets discovered, building all...")

    print(f"Building with {jobs} parallel job(s)...")
    sys.stdout.flush()
    build_cmd = [
        "cmake",
        "--build",
        str(build_dir),
        f"-j{jobs}",
        "--config",
        "Release",
    ]
    result = subprocess.run(build_cmd)
    if result.returncode != 0:
        print("Error: build failed")
        return False

    return True


def find_benchmarks(build_dir: Path, pattern: Optional[str] = None) -> List[Path]:
    """Find benchmark executables in the build directory."""
    benchmarks = []

    # Search common locations and multi-config subdirectories (Release, Debug, etc.)
    _config_subdirs = ["Release", "RelWithDebInfo", "Debug", "MinSizeRel"]
    search_paths = []
    for base in [build_dir / "bin", build_dir / "benchmarks", build_dir]:
        search_paths.append(base)
        for cfg in _config_subdirs:
            search_paths.append(base / cfg)

    for search_path in search_paths:
        if not search_path.exists():
            continue

        for f in search_path.iterdir():
            if not f.is_file():
                continue
            name = f.name
            # Only consider executable files (on Windows, must end with .exe)
            if platform.system() == "Windows" and not name.endswith(".exe"):
                continue
            if "benchmark" in name.lower():
                if pattern is None or re.search(pattern, name):
                    benchmarks.append(f)

    # Remove duplicates, preferring Release over Debug
    seen = {}
    for bench in benchmarks:
        key = bench.name
        if key not in seen:
            seen[key] = bench
        else:
            # Prefer Release build
            if "Release" in str(bench) and "Debug" in str(seen[key]):
                seen[key] = bench

    return sorted(seen.values())


def run_benchmark(
    benchmark_path: Path,
    extra_args: Optional[List[str]] = None,
    filter_pattern: Optional[str] = None,
    env_override: Optional[Dict[str, str]] = None,
    name_suffix: str = "",
) -> Dict[str, Any]:
    """Run a single benchmark and return results."""
    args = [str(benchmark_path), "--benchmark_format=json"]
    if extra_args:
        args.extend(extra_args)

    # Apply filter if provided
    if filter_pattern:
        args.append(f"--benchmark_filter={filter_pattern}")

    # Special handling for nested_for_benchmark - exclude slow async cases
    if benchmark_path.stem == "nested_for_benchmark" and not filter_pattern:
        # Exclude async benchmarks (extremely slow, not competitive)
        # Only include serial, omp, tbb, and dispenso tests
        args.append("--benchmark_filter=BM_(serial|omp|tbb|dispenso)")

    result_name = benchmark_path.name + name_suffix

    # Set up environment
    env = None
    if env_override:
        env = os.environ.copy()
        env.update(env_override)

    print(f"Running: {benchmark_path.name}...")

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout per benchmark
            env=env,
        )

        if result.returncode != 0:
            error_msg = (
                result.stderr or result.stdout or f"exit code {result.returncode}"
            )
            return {
                "name": benchmark_path.name,
                "success": False,
                "error": error_msg,
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
            "error": "Timeout after 1800 seconds",
        }
    except Exception as e:
        return {
            "name": benchmark_path.name,
            "success": False,
            "error": str(e),
        }


def _build_extra_benchmark_args(args) -> List[str]:
    """Build the extra benchmark arguments list from parsed CLI args."""
    extra_args = []
    if args.min_time is not None:
        extra_args.append(f"--benchmark_min_time={args.min_time}s")
    if args.repetitions is not None:
        extra_args.append(f"--benchmark_repetitions={args.repetitions}")
    return extra_args


def _build_windows_env_override(
    cmake_args: Optional[List[str]],
) -> Optional[Dict[str, str]]:
    """Build PATH environment override for Windows DLL discovery.

    On Windows, adds bin/ dirs from CMAKE_PREFIX_PATH so vcpkg DLLs
    (TBB, etc.) are found at benchmark runtime.
    """
    if platform.system() != "Windows" or not cmake_args:
        return None
    extra_bin_dirs = []
    for arg in cmake_args:
        if arg.startswith("-DCMAKE_PREFIX_PATH="):
            prefix = arg.split("=", 1)[1].strip("\"'")
            for p in prefix.split(";"):
                extra_bin_dirs.append(str(Path(p) / "bin"))
    if not extra_bin_dirs:
        return None
    return {
        "PATH": os.pathsep.join(extra_bin_dirs)
        + os.pathsep
        + os.environ.get("PATH", "")
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run dispenso benchmarks and output results as JSON"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Configure and build benchmarks before running",
    )
    parser.add_argument(
        "--source-dir",
        "-s",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Source directory (dispenso root)",
    )
    parser.add_argument(
        "--build-dir",
        "-b",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="Build directory containing benchmark executables",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("benchmark_results.json"),
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--benchmarks",
        "-B",
        type=str,
        default=None,
        help="Regex pattern to filter which benchmark executables to run",
    )
    parser.add_argument(
        "--filter",
        "-f",
        type=str,
        default=None,
        help="Filter pattern passed to --benchmark_filter for individual tests",
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=None,
        help="Minimum time per benchmark in seconds (passed to --benchmark_min_time)",
    )
    parser.add_argument(
        "--tcmalloc",
        type=str,
        default=None,
        help="Path to libtcmalloc.so for LD_PRELOAD tests (e.g., /usr/lib64/libtcmalloc.so.4)",
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
        help="Wipe the build directory before configuring (requires --build)",
    )
    parser.add_argument(
        "--cmake-args",
        action="append",
        default=None,
        metavar="ARG",
        help="Extra cmake argument, can be repeated "
        '(e.g., --cmake-args="-DCMAKE_CXX_COMPILER=g++-13")',
    )
    parser.add_argument(
        "--repetitions",
        "-r",
        type=int,
        default=None,
        help="Number of repetitions per benchmark (passed to --benchmark_repetitions). "
        "Produces mean/median/stddev/cv aggregate rows in the output.",
    )
    parser.add_argument(
        "--platform",
        "-p",
        type=str,
        default=None,
        help="Platform identifier (e.g., 'linux-threadripper-96c'). "
        "Auto-detected from machine info if not specified.",
    )

    args = parser.parse_args()

    # Gather machine info
    print("Gathering machine information...")
    machine_info = get_machine_info()
    print(f"  CPU: {machine_info.get('cpu_model', 'unknown')}")
    print(f"  Hardware Threads: {machine_info.get('cpu_cores', 'unknown')}")
    print(f"  Memory: {machine_info.get('memory_gb', 'unknown')} GB")

    # Platform identifier (auto-detect or user-specified)
    platform_id = args.platform or generate_platform_id(machine_info)
    machine_info["platform_id"] = platform_id
    print(f"  Platform: {platform_id}")

    # Build if requested
    if args.build:
        print()
        if not configure_and_build(
            source_dir=args.source_dir.resolve(),
            build_dir=args.build_dir.resolve(),
            jobs=args.jobs,
            clean=args.clean,
            benchmark_filter=args.benchmarks,
            extra_cmake_args=args.cmake_args,
        ):
            sys.exit(1)

    # Gather compiler info from the build directory
    compiler_info = get_compiler_info(args.build_dir)
    if compiler_info:
        machine_info["compiler"] = compiler_info
        summary = compiler_info.get("compiler_summary", "unknown")
        print(f"  Compiler: {summary}")
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
    extra_args = _build_extra_benchmark_args(args)
    env_override = _build_windows_env_override(args.cmake_args)

    results = []
    for benchmark in benchmarks:
        result = run_benchmark(
            benchmark,
            extra_args if extra_args else None,
            args.filter,
            env_override=env_override,
        )
        results.append(result)
        if result["success"]:
            print(f"  OK {benchmark.name}")
        else:
            print(
                f"  FAIL {benchmark.name}: {result.get('error', 'unknown error')[:200]}"
            )

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "machine_info": machine_info,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nSaved results to: {args.output}")

    # Print summary
    successful = sum(1 for r in results if r.get("success"))
    print(f"\nSummary: {successful}/{len(results)} benchmarks succeeded")

    if successful < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
