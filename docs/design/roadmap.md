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
| Scalable allocator | Custom allocator for containers like ConcurrentVector. System allocators show opposing strengths: tcmalloc is best for ConcurrentVector's geometrically-growing variable-size buffers under contention, while jemalloc is best for fixed-size small allocations (SBA's pattern) but worst for ConcurrentVector. A purpose-built allocator can optimize for dispenso's specific patterns rather than relying on any single system allocator. | - |
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

## Investigation Items

### SmallBufferAllocator vs System Malloc Performance

**Context.** Benchmark comparison across glibc, tcmalloc, and jemalloc on a
192-thread Threadripper PRO 7995WX reveals that SmallBufferAllocator (SBA) does
not always outperform modern system allocators:

| Scenario | glibc vs SBA | tcmalloc vs SBA | jemalloc vs SBA |
|----------|-------------|-----------------|-----------------|
| Small/1t | glibc 2x faster | tcmalloc 4x faster | jemalloc 1.6x faster |
| Medium-Large/1t | SBA 4-7x faster than glibc | tcmalloc 1-3x faster | ~tied |
| Small/16t | glibc 2x faster | tcmalloc 2.4x slower | jemalloc 2x faster |
| Medium-Large/16t | SBA 3-53x faster than glibc | tcmalloc 2-22x slower | jemalloc 1.5-1.6x faster |

**Key findings:**
- jemalloc beats SBA in every scenario (0.5-0.7x), including under contention
- tcmalloc is faster single-threaded but collapses under 16-thread contention
  (22x slower than SBA at 32K allocs)
- SBA's main value is vs glibc under medium/large contended workloads
- Windows MSVC CRT malloc shows similar patterns to jemalloc (fast under contention)

**TODO:**
- Investigate whether SBA should detect jemalloc/mimalloc at build time and
  defer to the system allocator for small allocations
- Consider thread-local free lists (similar to jemalloc's tcache) to reduce
  atomic contention in SBA's hot path
- Test with mimalloc (not available on current test machine)
- Note: jemalloc's advantage over SBA does not extend to ConcurrentVector's
  parallel growth benchmarks, where jemalloc performs worst and tcmalloc best.
  This suggests the optimal allocator strategy differs by allocation pattern
  (fixed-size recycling vs geometrically-growing buffers), reinforcing the
  case for a custom dispenso allocator rather than deferring to any single
  system malloc.

### PoolAllocator Thread-Local Optimization

**Context.** The locked `PoolAllocator` is 3-4x slower on Windows (64-core Xeon)
vs Linux (192-thread Threadripper), while the no-lock variant (`nl_pool_allocator`)
shows only 1.1-1.2x difference. This points to atomic operations being the
bottleneck, not the allocation logic itself.

Single-threaded raw numbers (8192 allocs):
- `nl_pool_allocator`: Linux 30-34K ns, Windows 34-37K ns (1.1x)
- `pool_allocator` (locked): Linux 35-42K ns, Windows 152K ns (3.6-4.1x)

**Proposed approach:** Thread-local free lists that batch allocations from the
shared pool. Each thread maintains a small local free list (e.g. 64-256 entries).
Allocate from local list first (no atomics). When empty, grab a batch from the
shared pool (one atomic op for N allocations). When local list exceeds threshold,
return a batch to the shared pool.

This is the same pattern that makes jemalloc and tcmalloc fast for general
allocation — amortizing synchronization cost across many allocations.

### Nested parallel_for Benchmark Optimization Artifact

**Context.** `nested_for_benchmark` serial benchmarks show near-zero times on
Windows (0.00x and 0.01x ratio vs Linux), indicating the compiler is optimizing
away the computation. The benchmark loop body needs `benchmark::DoNotOptimize`
or equivalent to prevent dead code elimination.

**TODO:** Add `benchmark::DoNotOptimize` to the serial benchmark loop in
`nested_for_benchmark.cpp` to ensure the computation is not elided.

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

### Fork-Join Scheduling & Thread Groups (post-1.5)

Full design document: [fork_join_scheduling.md](fork_join_scheduling.md)

**Summary.** Add per-thread bounded MPMC rings (Vyukov algorithm), thread
groups with per-group EpochWaiters, and cache-topology-aware group assignment
to enable fork-join-style scheduling for `parallel_for`.  Closes the ~5x IPC
gap vs OMP on locality-sensitive patterns by ensuring deterministic
thread-to-chunk affinity across iterations.  All existing semantics (TaskSet,
Futures, Pipelines, deadlock prevention) are preserved.
