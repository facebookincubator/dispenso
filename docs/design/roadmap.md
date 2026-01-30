# Dispenso Roadmap

This document tracks planned features and improvements for the dispenso library.

## In Progress

| Feature | Status | Notes |
|---------|--------|-------|
| vcpkg package | In progress | External PR to microsoft/vcpkg |
| Conan package | In progress | External PR to conan-center-index |
| ConcurrentHashMap | In progress | High-value concurrent container |

## Planned

### High Priority

| Feature | Description | Doc |
|---------|-------------|-----|
| Parallel sorting | `dispenso::sort` and MSD radix hybrid | [parallel_algorithms.md](parallel_algorithms.md) |
| Parallel algorithms (Phase 1) | for_each, transform, fill, reduce | [parallel_algorithms.md](parallel_algorithms.md) |
| C++20 concepts | Better error messages with concept constraints | [cpp20_concepts.md](cpp20_concepts.md) |
| Benchmark automation | Script to run benchmarks and generate charts | See benchmarks/ |

### Medium Priority

| Feature | Description | Doc |
|---------|-------------|-----|
| Parallel algorithms (Phase 2-3) | Search, count, copy, replace | [parallel_algorithms.md](parallel_algorithms.md) |
| Compiler Explorer examples | Godbolt links in README | - |
| Barrier/Semaphore | C++20-style synchronization for C++14/17 | - |
| ConcurrentQueue | Public API for blocking MPMC queue | - |

### Lower Priority

| Feature | Description | Doc |
|---------|-------------|-----|
| Parallel algorithms (Phase 4-5) | Sorting, scan, unique | [parallel_algorithms.md](parallel_algorithms.md) |
| Coroutine integration | Coroutine-based task scheduling | [coroutines.md](coroutines.md) |
| Single-header amalgamation | Full library in one header | - |

## Completed

| Feature | Version | Notes |
|---------|---------|-------|
| `dispenso.h` convenience header | 1.5.0 | Includes all public headers |
| `util.h` public utilities | 1.5.0 | Exposes internal utilities |
| OpenMP migration guide | 1.4.x | docs/migrating_from_openmp.md |
| TBB migration guide | 1.4.x | docs/migrating_from_tbb.md |
| awesome-cpp listing | - | Listed in fffaraz/awesome-cpp |
| awesome-modern-cpp listing | - | Listed in rigtorp/awesome-modern-cpp |

## External Submissions

| Target | Status | Notes |
|--------|--------|-------|
| awesome-cpp | Listed | fffaraz/awesome-cpp |
| awesome-modern-cpp | Listed | rigtorp/awesome-modern-cpp |
| awesome-high-performance-computing | Listed | Already present in dstansby/awesome-high-performance-computing |
| awesome-scientific-computing | Not applicable | Focus is numerical methods, not parallelism libraries |
| awesome-hpc | Not applicable | Focus is cluster infrastructure, not app-level parallelism |
| vcpkg | In progress | - |
| Conan | In progress | - |

## Ideas / Backlog

These are ideas that may be pursued based on community feedback:

- Lock-free stack
- Scalable allocator (like TBB's)
- Range-based API wrappers (explicit opt-in)
- SIMD-optimized algorithms
- Integration examples (game engines, scientific computing)
- Discord/Slack community channel
