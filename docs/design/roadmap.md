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
| Compiler Explorer examples | Godbolt links in README for try-it-now experience | - |

### Medium Priority

| Feature | Description | Doc |
|---------|-------------|-----|
| Scalable allocator | Thread-caching allocator to eliminate malloc contention in concurrent growth (ConcurrentVector `parallel` is 3-5x faster with tcmalloc/jemalloc vs glibc) | - |
| Parallel algorithms (Phase 2-3) | Search, count, copy, replace | [parallel_algorithms.md](parallel_algorithms.md) |
| Barrier/Semaphore | C++20-style synchronization for C++14/17 | - |
| ConcurrentQueue | Public API for blocking MPMC queue | - |

### Lower Priority

| Feature | Description | Doc |
|---------|-------------|-----|
| Parallel algorithms (Phase 4-5) | Sorting, scan, unique | [parallel_algorithms.md](parallel_algorithms.md) |
| Coroutine integration | Coroutine-based task scheduling | [coroutines.md](coroutines.md) |
| Single-header amalgamation | Full library in one header | - |

## ConcurrentVector Optimization Notes

### Optimizations Applied (Default Traits)

Three categories of optimization have been applied to ConcurrentVector:

1. **Inline asm `bsr` for `detail::log2`** on x86 GCC/Clang, plus 32-bit
   overloads and `unsigned long` disambiguation for macOS. Prevents Clang from
   decomposing `63 - __builtin_clzll` back into `bsrq + xorq` when inlined
   into arithmetic.

2. **Platform-adaptive `bucketAndSubIndexForIndex`**: branching fast path
   (early return for `index < firstBucketLen_`) on MSVC and ARM where branch
   predictors handle the sequential pattern well; branchless cmov path on
   Clang/GCC x86 where cmovs avoid misprediction penalties.

3. **Non-atomic buffer pointer cache (`cachedPtrs_[]`)** on non-ARM platforms.
   Packs 8 pointers per cache line (vs 1 per line for `AlignedAtomic
   buffers_[]`), dramatically improving `operator[]` and iterator read paths.
   Disabled on ARM where cache-line invalidation on every write exceeds the
   read benefit. Cache stores are ordered before the release store to
   `buffers_[]`, so any acquire on `buffers_[]` guarantees cache visibility.

### Impact on Alternative Traits

| Trait | Values | Optimization Interaction |
|-------|--------|--------------------------|
| `kPreferBuffersInline` | `false` | Cache is *more* valuable — bypasses the extra indirection through heap-allocated `buffers_[]` pointer |
| `kIteratorPreferSpeed` | `false` (compact iterator) | Benefits *disproportionately* — compact iterator calls `operator[]` (and thus `cachedBuffer` + `bucketAndSubIndexForIndex`) on every dereference, vs speed iterator which only calls on bucket transitions |
| `kReallocStrategy` | `kHalfBufferAhead`, `kFullBufferAhead` | No interaction — earlier allocation just means cache is populated earlier |

**Conclusion:** All optimizations apply uniformly across trait combinations.
The current defaults (`kPreferBuffersInline=true`, `kIteratorPreferSpeed=true`,
`kReallocStrategy=kAsNeeded`) remain the best general-purpose configuration.
The compact iterator (`kIteratorPreferSpeed=false`) benefits the most from the
`cachedPtrs_` and `log2` optimizations in relative terms, since it hits the
indexed access path on every element access.

### Future Work

- **Server ARM (Graviton) benchmarking**: The `DISPENSO_HAS_CACHED_PTRS`
  guard currently disables the cache on all ARM (`__aarch64__`). If server ARM
  shows different cache pressure characteristics than mobile ARM, a more
  targeted guard could re-enable it selectively.
- **Scalable allocator** (see Medium Priority above): Concurrent growth
  benchmarks show 3-5x improvement with tcmalloc/jemalloc vs glibc malloc,
  suggesting a thread-caching allocator would benefit all trait combinations.

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

## Subcomponent Roadmaps

| Component | Doc |
|-----------|-----|
| dispenso::fast_math | [fast_math_roadmap.md](fast_math_roadmap.md) |

## Ideas / Backlog

These are ideas that may be pursued based on community feedback:

- CUDA graph mappings (TaskFlow has this; worth exploring for dispenso's Graph)
- Lock-free stack
- Range-based API wrappers (explicit opt-in)
- SIMD-optimized algorithms
- Integration examples (game engines, scientific computing)
- Discord/Slack community channel
- Windows thread pool wake strategy tuning:
  - Current approach uses unconditional `WakeByAddressAll` in `wakeN()`. The primary benefit is **parallel wake-up**: all threads begin their OS-level wake simultaneously, rather than serially. Thread wake-up latency (scheduler, context switch, cache warm-up) dominates — the `WakeByAddress` syscall itself is fast (<1 us)
  - Secondary benefit: the scheduling thread makes one syscall instead of N, freeing it to enqueue work or start its own computation sooner. This matters because the scheduling thread cannot worksteal until the queue is saturated, so every microsecond spent in wake syscalls is a microsecond of dead time where neither the caller nor the (still-sleeping) pool threads are making progress
  - The increased spin constants (kBackoffYield=100, kBackoffSleep=120 vs Linux's 50/55) are complementary: keeping threads in spin phase longer avoids the expensive wake-up latency entirely. Sub-microsecond spin cost vs 10s-of-microsecond wake-up cost
  - Benchmarks showed WakeAll outperforms the macOS-style heuristic (`n >= sleeping/2` → wakeAll, else wake individually), but thresholds N/2, N/3, N/4 were tested — not smaller fractions like N/16 that might limit thundering herd while still avoiding serial wake overhead
  - The spin constant and WakeAll changes were not benchmarked independently
  - Investigate: (1) whether a lower threshold (e.g., `n < sleeping/16`) balances thundering herd vs parallel wake benefit, (2) whether the spin constants should be tuned independently, (3) steady-state single-task patterns where WakeAll wakes unnecessary threads
  - Requires Windows benchmarking access to validate
- NUMA and topology awareness (phased):
  - Windows processor group support for >64 threads (less critical as newer Windows versions handle this automatically)
  - Topology query API: expose NUMA node count, core-to-node mapping, and inter-node distances (Linux: `/sys/devices/system/node/` or `libnuma`; Windows: `GetLogicalProcessorInformationEx`)
  - Per-NUMA-node thread pools: opt-in pool construction affinitized to a specific node, composable with existing TaskSet/Future APIs
  - NUMA-aware allocator: STL-compatible allocator for node-local allocation (`mbind`/`numa_alloc_onnode` on Linux, `VirtualAllocExNuma` on Windows), paired with first-touch initialization guidance
