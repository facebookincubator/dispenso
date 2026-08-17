# Dispenso Roadmap

This document tracks planned features and improvements for the dispenso library.

## In Progress

| Feature | Status | Notes |
|---------|--------|-------|
| ConcurrentHashMap | In progress | High-value concurrent container |

Packaging and listing work is tracked under
[External Submissions](#external-submissions), not here.

## Planned

### High Priority

| Feature | Description | Doc |
|---------|-------------|-----|
| Parallel sorting | `dispenso::sort` and MSD radix hybrid | [parallel_algorithms.md](proposals/parallel_algorithms.md) |
| Parallel algorithms (Phase 1) | for_each, transform, fill, reduce | [parallel_algorithms.md](proposals/parallel_algorithms.md) |
| C++20 concepts | Better error messages with concept constraints | [cpp20_concepts.md](proposals/cpp20_concepts.md) |
| Benchmark automation | Script to run benchmarks and generate charts | See benchmarks/ |
| Compiler Explorer examples | Godbolt links in README for try-it-now experience | - |

### Medium Priority

| Feature | Description | Doc |
|---------|-------------|-----|
| Scalable allocator | Custom allocator for containers like ConcurrentVector. System allocators show opposing strengths: tcmalloc is best for ConcurrentVector's geometrically-growing variable-size buffers under contention, while jemalloc is best for fixed-size small allocations (SBA's pattern) but worst for ConcurrentVector. A purpose-built allocator can optimize for dispenso's specific patterns rather than relying on any single system allocator. | - |
| Parallel algorithms (Phase 2-3) | Search, count, copy, replace | [parallel_algorithms.md](proposals/parallel_algorithms.md) |
| Barrier/Semaphore | C++20-style synchronization for C++14/17 | - |
| ConcurrentQueue | Public API for blocking MPMC queue | - |

### Lower Priority

| Feature | Description | Doc |
|---------|-------------|-----|
| Parallel algorithms (Phase 4-5) | Sorting, scan, unique | [parallel_algorithms.md](proposals/parallel_algorithms.md) |
| Coroutine integration | Coroutine-based task scheduling | [coroutines.md](proposals/coroutines.md) |
| Single-header amalgamation | Full library in one header | - |

## ConcurrentVector

Applied optimizations and their interaction with traits are documented in
[architecture/concurrent_vector.md](architecture/concurrent_vector.md).

- **Server ARM (Graviton) benchmarking**: The `DISPENSO_HAS_CACHED_PTRS`
  guard currently disables the cache on all ARM (`__aarch64__`). If server ARM
  shows different cache pressure characteristics than mobile ARM, a more
  targeted guard could re-enable it selectively.
- **Scalable allocator** (see Medium Priority above): Concurrent growth
  benchmarks show 3-5x improvement with tcmalloc/jemalloc vs glibc malloc,
  suggesting a thread-caching allocator would benefit all trait combinations.

## Completed

[CHANGELOG.md](../../CHANGELOG.md) is the full record of shipped work. This
table lists only items that were carried on this roadmap, so their entries do
not dangle once done.

| Feature | Version | Notes |
|---------|---------|-------|
| `dispenso.h` convenience header | 1.5.0 | Includes all public headers |
| `util.h` public utilities | 1.5.0 | Exposes internal utilities |
| OpenMP migration guide | 1.4.x | docs/migrating_from_openmp.md |
| TBB migration guide | 1.4.x | docs/migrating_from_tbb.md |
| Fork-join scheduling and thread groups | 1.6.0 | Locality rings, steal rings, wake cascade, CpuSet topology. See [three_tier_scheduling.md](architecture/three_tier_scheduling.md) and [wake_cascade.md](architecture/wake_cascade.md) |

## External Submissions

| Target | Status | Notes |
|--------|--------|-------|
| awesome-cpp | Listed | fffaraz/awesome-cpp |
| awesome-modern-cpp | Listed | rigtorp/awesome-modern-cpp |
| awesome-high-performance-computing | Listed | Already present in dstansby/awesome-high-performance-computing |
| awesome-scientific-computing | Not applicable | Focus is numerical methods, not parallelism libraries |
| awesome-hpc | Not applicable | Focus is cluster infrastructure, not app-level parallelism |
| vcpkg | In progress | PR to microsoft/vcpkg |
| Conan | In progress | PR to conan-center-index |

## Subcomponent Roadmaps

| Component | Doc |
|-----------|-----|
| dispenso::fast_math | [fast_math_roadmap.md](roadmap/fast_math.md) |

## Investigation Items

Open questions and measurements that are not yet commitments. Each is a
standalone note; results here inform roadmap items rather than being them.

| Investigation |
|---------------|
| [SmallBufferAllocator vs System Malloc Performance](investigations/smallbufferallocator_vs_system_malloc.md) |
| [PoolAllocator Thread-Local Optimization](investigations/poolallocator_thread_local.md) |
| [Nested parallel_for Benchmark Optimization Artifact](investigations/nested_parallel_for_benchmark_artifact.md) |
| [Decouple sleep mask from group concept](investigations/decouple_sleep_mask_from_groups.md) |
| [Idle-Pool Wake Latency vs CPU Cost (mostly-idle then burst)](investigations/idle_pool_wake_latency.md) |
| [Recursive fork-join vs TBB task_group (deep / heavy trees)](investigations/recursive_forkjoin_vs_tbb_task_group.md) |
| [Topology-hierarchy-aware scheduling (beyond L3-as-NUMA-proxy)](investigations/topology_hierarchy_aware_scheduling.md) |
| [Experiment: externalize per-group locality; simplify thread-pool tiers](investigations/externalize_per_group_locality.md) |

## Ideas / Backlog

These are ideas that may be pursued based on community feedback:

- Steal-ring round-robin for non-sleeping placed scheduling: `scheduleImplPlaced` currently only pushes to steal rings when it can claim a sleeping thread. When no threads are sleeping, the task falls through to the central queue. A round-robin steal-ring path for pool-worker callers could improve locality by keeping work near the scheduling thread. Requires benchmarking to confirm benefit over the central queue path.
- CUDA graph mappings (TaskFlow has this; worth exploring for dispenso's Graph)
- Lock-free stack
- Speculative `Future` latency: the `future_benchmark` speculative suite runs ~1.5-3x behind `folly::Future` (while dispenso is ~2.2x faster on the kv-cache workload), so the gap is speculative-execution-specific — worth profiling folly's speculative fast path.
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
  - **NUMA-aware thread grouping**: `buildThreadGroups()` currently groups by L3 cache topology only. On multi-socket systems where NUMA node boundaries don't align with L3 boundaries (e.g., Intel Sub-NUMA Clustering), groups may span NUMA nodes. Add an option to respect NUMA boundaries in addition to L3 boundaries, so that memory allocated by threads in a group stays node-local.
  - Windows processor group support for >64 threads (less critical as newer Windows versions handle this automatically)
  - Topology query API: expose NUMA node count, core-to-node mapping, and inter-node distances (Linux: `/sys/devices/system/node/` or `libnuma`; Windows: `GetLogicalProcessorInformationEx`)
  - Per-NUMA-node thread pools: opt-in pool construction affinitized to a specific node, composable with existing TaskSet/Future APIs
  - NUMA-aware allocator: STL-compatible allocator for node-local allocation (`mbind`/`numa_alloc_onnode` on Linux, `VirtualAllocExNuma` on Windows), paired with first-touch initialization guidance
  - **>1024-CPU support in `CpuSet`**: `CpuSet` currently represents CPU IDs in `[0, 1024)` (Linux/FreeBSD `cpu_set_t` / `CPU_SETSIZE`; a 1024-bit array on Windows/macOS); IDs at or beyond that are ignored gracefully (no UB), so CPUs `>= 1024` just aren't bound or grouped. Raise the limit when hardware approaches it: dynamic `cpu_set_t` via `CPU_ALLOC` + the `*_S` ops on Linux, a wider bitset elsewhere. `CpuSet`s are allocated only at startup, so the cost is low (e.g. a `thread_local` scratch set avoids per-call allocation).
- Bounded `PoolWakeState` reclamation:
  - **Problem.** `ThreadPool::resize()` retires the old `PoolWakeState` into a grow-only graveyard that is freed only at pool destruction (see `thread_pool.h`). This is intentional and correct: `schedule()` reads `wakeState_` lock-free and dereferences it (e.g. `totalSleeping()`), so freeing a retired generation during operation is a use-after-free against a concurrent reader (TSAN-confirmed; an earlier bounded variant that freed old generations was reverted for exactly this). Cost is one `PoolWakeState` (~O(numThreads)) retained per `resize()`; only a long-lived process doing hundreds of thousands of resizes accumulates meaningful memory, so it is bounded in practice.
  - **Goal.** Reclaim retired generations safely without taxing the `schedule()` hot path — no per-call reference count and no hardware fence on the reader side.
  - **Approach (asymmetric-fence / hazard-pointer reclamation).** Readers (`schedule()`) publish the `wakeState_` pointer they are about to use into a per-thread hazard slot via a *relaxed* store bracketed by compiler-only fences — nearly free, no atomic RMW, no contention (this is a published pointer, not a reference count). `resize()` (rare, may be slow) publishes the new generation, issues a process-wide "heavy" barrier, then scans all hazard slots and frees a retired generation only once no slot still references it. The heavy barrier supplies the ordering asymmetrically so readers pay nothing: `sys_membarrier(MEMBARRIER_CMD_PRIVATE_EXPEDITED)` on Linux, `FlushProcessWriteBuffers()` on Windows, and an `mprotect`-toggle (forces a TLB-shootdown IPI → per-core barrier) fallback for macOS and other POSIX. After the barrier, a reader either had already published the old pointer (the scan sees it and waits) or runs its load after the barrier and therefore observes the new generation — so a retired object can never be freed while a reader still holds it. (folly packages this as `asymmetric_thread_fence_{light,heavy}`; dispenso would self-implement to avoid the folly dependency.)

## 2.0 (API-breaking changes)

The following changes require a major version bump due to API breakage:

| Feature | Description |
|---------|-------------|
| Remove poll mode | Remove `threadLoopPoll`, the `setSignalingWake(false)` path, and the `DISPENSO_WAKEUP_ENABLE` / `DISPENSO_POLL_PERIOD_US` defines. Signaling wake (futex/WaitOnAddress/ulock) is always-on and well-tested. Poll mode is a legacy fallback that adds code complexity without benefit on any supported platform. |
| C++17 minimum | Bump minimum standard from C++14 to C++17. Enables `std::optional` (replacing `OpResult`), `if constexpr`, structured bindings, `[[nodiscard]]`, and `std::string_view`. Simplifies template metaprogramming throughout. |
| C++20 consideration | Evaluate C++20 as minimum for a later 2.x release. Enables `std::atomic<T>::wait/notify` (potential EpochWaiter simplification), concepts (better error messages), coroutine integration, and `std::jthread`. |
