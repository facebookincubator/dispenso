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

### Code TODOs

Open `TODO` comments in the source, each named by file and function. See
[the README](../README.md#conventions) for how these are kept in step.

- **C++20 ranges overloads of `for_each` and `for_each_n`** (`dispenso/for_each.h`). Part of the range-based wrappers item above.
- **Nested `for_each_n` with `maxThreads`** (`dispenso/for_each.h`). Called from inside another parallel loop, `for_each_n` runs serially. Consider allowing a small fan-out when the caller sets `options.maxThreads`.
