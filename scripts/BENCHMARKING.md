# Benchmark Generation Guide

This guide covers building dispenso benchmarks and generating performance
charts for the 1.5.0 release across Linux, Mac, and Windows.

## Prerequisites

```bash
pip install matplotlib pandas
```

## Quick Start (Build + Run)

The simplest way to build and run benchmarks is with `run_benchmarks.py --build`:

```bash
# Build and run all benchmarks
python3 scripts/run_benchmarks.py --build \
  -o results/my-platform.json

# Build and run only pipeline benchmark
python3 scripts/run_benchmarks.py --build -B pipeline \
  -o results/my-platform.json

# Use a custom compiler
python3 scripts/run_benchmarks.py --build \
  --cmake-args="-DCMAKE_CXX_COMPILER=g++-13" \
  -o results/my-platform.json

# Clean rebuild with custom build directory
python3 scripts/run_benchmarks.py --build --clean \
  -b /tmp/dispenso-bench \
  -o results/my-platform.json
```

The script auto-detects your platform (e.g., `linux-threadripper-96c`,
`macos-m4-12c`) and includes it in the output. Override with `--platform`:

```bash
python3 scripts/run_benchmarks.py --build \
  --platform my-custom-label \
  -o results/my-custom-label.json
```

### Build Flags

| Flag | Description |
|------|-------------|
| `--build` | Enable cmake configure + build before running |
| `--build-dir` / `-b` | Build directory (default: `/tmp/dispenso-build`) |
| `--source-dir` / `-s` | Dispenso source root (default: auto-detected) |
| `--jobs` / `-j` | Parallel build jobs (default: all cores) |
| `--clean` | Wipe build directory before configuring |
| `--cmake-args` | Extra cmake flags (repeatable) |

## Per-Platform Manual Build

If you prefer to configure and build manually, or need to pass
platform-specific options not covered by `--cmake-args`:

### Linux

```bash
mkdir /tmp/dispenso-bench && cd /tmp/dispenso-bench
cmake <path-to-dispenso> \
  -DDISPENSO_BUILD_TESTS=OFF \
  -DDISPENSO_BUILD_BENCHMARKS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/g++ \
  -DFETCHCONTENT_SOURCE_DIR_TASKFLOW=~/dispenso_deps/taskflow-3.11.0
cmake --build . -j$(nproc)
```

### Mac

```bash
mkdir /tmp/dispenso-bench && cd /tmp/dispenso-bench
cmake <path-to-dispenso> \
  -DDISPENSO_BUILD_TESTS=OFF \
  -DDISPENSO_BUILD_BENCHMARKS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DFETCHCONTENT_SOURCE_DIR_TASKFLOW=~/dispenso_deps/taskflow-3.11.0
cmake --build . -j$(sysctl -n hw.ncpu)
```

If using Homebrew GCC instead of Xcode clang, add:
```
  -DCMAKE_CXX_COMPILER=g++-13
```

### Windows

```bat
mkdir C:\tmp\dispenso-bench && cd C:\tmp\dispenso-bench
cmake <path-to-dispenso> ^
  -DDISPENSO_BUILD_TESTS=OFF ^
  -DDISPENSO_BUILD_BENCHMARKS=ON ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CXX_STANDARD=20 ^
  -DFETCHCONTENT_SOURCE_DIR_TASKFLOW=C:\dispenso_deps\taskflow-3.11.0
cmake --build . --config Release -j
```

Then run benchmarks against the existing build:

```bash
python3 scripts/run_benchmarks.py \
  -b /tmp/dispenso-bench \
  -o results/<platform-name>.json
```

## Step 1: Run Benchmarks

After building (either via `--build` or manually), run all benchmarks and
capture results as JSON:

```bash
python3 scripts/run_benchmarks.py \
  -b /tmp/dispenso-bench \
  -o results/<platform-name>.json
```

Where `<platform-name>` is a descriptive identifier like `linux-threadripper-96c`,
`macos-m4-12c`, `windows-zen4-24c`, etc. (auto-detected if omitted).

### Run Options

- `-B <regex>` — filter which benchmark executables to run
  (e.g. `-B "pipeline|simple_pool"`)
- `-f <pattern>` — filter individual test cases within benchmarks
  (passed to `--benchmark_filter`)
- `--min-time <seconds>` — minimum time per benchmark
- `--tcmalloc <path>` — use tcmalloc via LD_PRELOAD
- `-p <name>` — override auto-detected platform identifier

## Step 2: Generate Charts

Generate per-platform charts from the JSON results:

```bash
python3 scripts/generate_charts.py \
  -i results/<platform-name>.json \
  -o docs/benchmarks/ \
  --platform <platform-name>
```

Charts are written to `docs/benchmarks/<platform-name>/`.

## Step 3: Compose Multi-Platform Landing Page

After collecting JSON from all platforms, generate a unified landing page:

```bash
python3 scripts/update_benchmarks.py --compose \
  linux-threadripper:results/linux.json \
  macos-m2:results/macos.json \
  windows-zen4:results/windows.json
```

This creates per-platform subdirectories under `docs/benchmarks/` and a
unified `benchmark_results.md` landing page.

## All-in-One (Single Platform)

`update_benchmarks.py` can build, run, chart, and update docs in one command:

```bash
python3 scripts/update_benchmarks.py \
  --platform linux-threadripper

# Or with an existing build:
python3 scripts/update_benchmarks.py \
  -b /tmp/dispenso-bench \
  --skip-build \
  --platform linux-threadripper
```

## Notes

- Build directories should be in `/tmp` (or equivalent) to avoid polluting
  the source tree
- Taskflow must be available locally (FetchContent fallback requires internet)
- The benchmark runner automatically collects machine info (CPU, cores, memory)
  and includes it in the JSON output
- Platform identifiers are auto-detected from machine info when not specified
- Chart names are mapped to README-expected filenames via `README_CHART_MAPPING`
  in `update_benchmarks.py`
