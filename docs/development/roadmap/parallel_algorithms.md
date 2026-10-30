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
- **`for_each_n` on small inputs.** At 1K elements (`for_each_benchmark`, Linux EPYC), `for_each_n` takes 0.8 us on one thread and gets slower as threads are added: 6 us at 16 threads, 62 us at 32, 230 us at 166. Unlike `parallel_for`, it has no minimum number of items per chunk, so a small input is always split across every thread. Consider the same option, or a default threshold.
- Range-based API wrappers (explicit opt-in)

### Code TODOs

Open `TODO` comments in the source, each named by file and function. See
[the README](../README.md#conventions) for how these are kept in step.

- **C++20 ranges overloads of `for_each` and `for_each_n`** (`dispenso/for_each.h`). Part of the range-based wrappers item above.
- **Nested `for_each_n` with `maxThreads`** (`dispenso/for_each.h`). Called from inside another parallel loop, `for_each_n` runs serially. Consider allowing a small fan-out when the caller sets `options.maxThreads`.
