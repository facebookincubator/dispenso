<!-- Part of the dispenso roadmap; index at ../roadmap.md -->

# Roadmap: parallel algorithms

`parallel_for`, sorting, and std-style parallel algorithms.

Design: [parallel_algorithms](../proposals/parallel_algorithms.md).

## Planned

| Feature | Description | Priority |
|---------|-------------|----------|
| Parallel sorting | `dispenso::sort` and MSD radix hybrid | High |
| Parallel algorithms (Phase 1) | for_each, transform, fill, reduce | High |
| Parallel algorithms (Phase 2-3) | Search, count, copy, replace | Medium |
| Parallel algorithms (Phase 4-5) | Sorting, scan, unique | Lower |

## Backlog

- SIMD-optimized algorithms
- Range-based API wrappers (explicit opt-in)
