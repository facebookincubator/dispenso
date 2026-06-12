# Frequently Asked Questions {#faq}

## When should I use dispenso vs alternatives?

**Use dispenso when:**
- You need nested parallelism without thread explosion
- You want sanitizer-clean (ASAN/TSAN) concurrent code
- You want explicit control over thread pools
- You need compute-bound futures with `then()` and `when_all()`

**Use something else when:**
- **I/O-heavy workloads** — dispenso is designed for compute-bound tasks. Thread pool threads that block on I/O waste scheduling capacity. For async I/O, consider [Folly](https://github.com/facebook/folly).
- **GPU task graphs** — TaskFlow offers CUDA graph mappings. Dispenso does not currently provide GPU graph scheduling.
- **You need `#pragma omp parallel for reduction(+:sum)` syntax** — dispenso's reduction requires per-thread state accumulation with manual combining (see [Getting Started](getting_started.md#your-first-parallel-loop)). A first-class reduction API is planned.

In kernel context switch scenarios (networking, disk I/O, TLB misses),
`dispenso::Future` can be used with `dispenso::NewThreadInvoker` for
per-task threading similar to `std::async`.

## What happens if a lambda throws inside parallel_for?

Exceptions thrown inside `parallel_for` or `TaskSet` lambdas are caught and
stored. When the calling thread calls `wait()` (or the TaskSet destructor
runs), the first captured exception is rethrown. Other in-flight tasks
continue to run — there is no immediate cancellation on exception.

For cooperative early termination without exceptions, use
`ConcurrentTaskSet` with cancellation.

Builds with `-fno-exceptions` are fully supported. In that mode, dispenso
uses `abort()` instead of throwing.

## How do I choose a chunking strategy for parallel_for?

| Strategy | Best for | How it works |
|----------|----------|--------------|
| `kStatic` (default) | Uniform per-element work | Divides range into equal chunks up front. Lowest overhead. |
| `kAdaptive` | Non-uniform work (e.g., Mandelbrot) | Stripe-based with work stealing. Prefers same-L3 cache victims. |
| `kAuto` | General use | Currently selects `kStatic`. May change in future versions. |

Use `ParForOptions::minItemsPerChunk` to set a lower bound on chunk size,
preventing over-parallelization for cheap per-element work.

## How does dispenso compare to TBB performance?

Dispenso tends to be faster for:
- Small and medium parallel loops (lower scheduling overhead)
- Nested parallel loops (work-stealing avoids oversubscription)
- Cascading independent loops (TBB has no equivalent)
- Graph construction and execution

TBB tends to be faster for:
- Large non-uniform workloads at very high thread counts (more mature work-stealing heuristics)
- First-class parallel reductions

See the [Interactive Benchmark Dashboard](https://facebookincubator.github.io/dispenso/benchmarks/)
for detailed cross-platform comparisons, or the
[TBB Migration Guide](migrating_from_tbb.md) for API mappings.

## Does dispenso work on my platform?

| Platform | Threading | CPU affinity | Topology detection | Wake mechanism |
|----------|-----------|--------------|-------------------|----------------|
| Linux x86_64/ARM64 | Full | Full | Full (sysfs) | futex |
| macOS ARM64/x86_64 | Full | No (OS limitation) | Full | os_sync_wait / ulock |
| Windows x86_64/ARM64 | Full | Full | Full | WaitOnAddress |
| FreeBSD | Full | Full (cpuset) | Planned | Planned |
| Android ARM64 | Full | Partial | Partial | futex |

## What C++ standard does dispenso require?

C++14 minimum. C++17 enables `std::shared_mutex` benchmarks and some
convenience features. C++20 adds concept constraints for better template
error messages — no behavioral changes.

## Can I use dispenso with CMake's FetchContent?

Yes:

```cmake
include(FetchContent)
FetchContent_Declare(
  dispenso
  GIT_REPOSITORY https://github.com/facebookincubator/dispenso.git
  GIT_TAG        v1.5.1  # or main for latest
)
FetchContent_MakeAvailable(dispenso)
target_link_libraries(my_target dispenso)
```

## How do I migrate from OpenMP / TBB?

- **[Migrating from TBB](migrating_from_tbb.md)** — API mappings, thread pool differences, common porting patterns
- **[Migrating from OpenMP](migrating_from_openmp.md)** — replacing `#pragma omp` with dispenso equivalents, handling reductions and nested parallelism
