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
| dispenso::fast_math | [fast_math_roadmap.md](roadmap/fast_math.md) |

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

**Multi-threaded contention (Linux EPYC-Genoa 166c, `pool_allocator_threaded`,
kSmallSize/8192).** The locked pool scales *negatively*: vs `malloc`/`free` it is
~6x faster at 2 threads, but ~7x slower at 8 threads and ~20x slower at 16
threads. glibc/tcmalloc's per-thread caches absorb concurrency the shared central
lock cannot — the thread-local-free-list fix below is what closes this, so the
multi-threaded case is the strongest motivation for it.

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

### Decouple sleep mask from group concept

**Context.** The wake-cascade design (see [wake_cascade.md](architecture/wake_cascade.md))
bundles three things into the "group" abstraction:
1. **Steal-ring locality** — threads in a group share a steal ring
2. **EpochWaiter / futex address** — one futex per group; `bumpAndWakeAll`
   wakes every parked thread in the group
3. **Sleep mask** — a 64-bit atomic per group where bit *i* tracks thread
   *i*'s sleep state, used by the bitmask cascade to find wake targets

The 1.6 default of `kDefaultWakeGroupSize=8` works well across Linux, macOS,
and Windows. The cross-platform tuning sweep did not surface a workload
that wanted these three to be sized differently — but the coupling is
implicit, not measured.

**Why investigate.** A few hypothetical cases where the coupling could hurt:

- **Steal-ring contention vs wake bucket contention** scale with different
  factors. Steal-ring contention grows with active producers + consumers
  on the ring; wake bucket contention grows with concurrent
  `WakeByAddress*` callers on the same address. A workload heavy on one
  but not the other might benefit from independent sharding.
- **Per-platform asymmetry.** On Windows where wake calls are 7-10 μs each,
  a smaller wake-mask shard (fewer potential targets per `wakeN`) might
  reduce spurious wake amplification, while keeping a larger steal-ring
  shard for locality. Today both are tied to `kDefaultWakeGroupSize`.
- **>64 thread groups.** The 64-bit sleep mask caps group size at 64.
  Not a limit anyone has hit — current G=8 leaves 56 bits unused on every
  platform — but a different sleep-tracking representation would lift it
  if a future design wanted very large groups.

**Proposed experiments (in order of cost):**

1. **Add separate `kSleepMaskShardSize`** that defaults to `groupSize_` (preserves
   current behavior). Sweep it independently in `run_tuning_experiment.py`.
   Expected to be a no-op based on current data; the value is ruling out
   the hypothesis that they want to be different.
2. **Wider sleep-mask representation** (e.g. an array of 64-bit words per
   group) only if (1) shows a workload preferring shard size > 64.
3. **Independent steal-ring sharing** sweep — already partially exposed
   via `kStealRingSharing` but historically tied to `kDefaultWakeGroupSize`
   in defaults. Decouple in defaults, sweep, see if any platform wants
   them different.

**Cost.** Mostly mechanical refactor — ~200 lines across `thread_pool_wake.h/cpp`
and `thread_pool.cpp`. The risk is tuning regression: more knobs means
more parameter space to validate per platform.

**When to revisit.** If a user reports a workload with > 64 contending
threads on a single steal-ring, or if a future architecture extension
(e.g. NUMA-node-aware wake) wants different sharding for wakes vs steal.
Otherwise, the current coupling is the simplest design that's correct.

### Idle-Pool Wake Latency vs CPU Cost (mostly-idle then burst)

**Objective — dual.** This suite targets *both* low burst latency (wall time)
**and** low CPU spent staying responsive. A keep-alive strategy that only
minimized wall time by spinning would just trade the problem for idle CPU/power
(mobile battery, shared hosts), so future work must be judged on both axes. The
Plotly dashboard now plots `cpu_time` as a dotted companion to wall time on the
idle_pool charts to make the tradeoff visible.

**Context.** The `idle_pool_benchmark` `mostly_idle` scenario — a pool that sits
idle and then receives periodic bursts of work — is dispenso's weakest suite
versus TBB, consistently ~0.24-0.26x (TBB ~4x faster by geomean) across
platforms, with far larger worst-case gaps:

| Platform | Case | dispenso | TBB | Ratio |
|----------|------|----------|-----|-------|
| Apple M4 Pro (12c) | `mostly_idle/12/1e6` | 3.76 s | 30 ms | ~125x |
| Pixel 9 Pro XL / Tensor G4 (8c) | `mostly_idle/8/1e6` | 2.3 s | 166 ms | ~14x |

**Why.** dispenso aggressively parks idle workers (futex / `WaitOnAddress` /
ulock) to yield the CPU when there is no work; waking them for each burst pays
the full wake-from-park latency (scheduler + context switch + cache warm-up),
which dominates when the burst itself is small. TBB keeps workers alive/spinning
adaptively so a burst finds them ready, and its SPMC-deque self-stealing tends
to re-run a repeating burst on the same core with a warm cache. dispenso has no
special-casing for the mostly-idle-then-burst pattern.

**Related: sustained static loops vs OpenMP.** The same park/wake cost — plus
dispenso's runtime chunk/steal-ring setup vs OpenMP's compile-time static
scheduling — leaves dispenso ~15-20% behind OpenMP on *large, sustained*
`parallel_for` (simple_for / summing_for / locality at large sizes, ~0.81-0.86x;
far worse at small sizes where OpenMP's `OMP_WAIT_POLICY=active` team is already
hot). dispenso still beats TBB ~2x on the same locality cases, so this is
specific to OpenMP's persistent hot team. The removed `AwakeRef`/`keepAwake` API
targeted part of this; a bounded keep-alive hint (below) would help here too.

**Possible approaches:**
- Adaptive park delay / hysteresis: keep recently-active workers spinning longer
  before parking when the pool has seen recent bursty activity, trading a little
  idle CPU for burst latency. Composes with the adaptive spin backoff and
  wake-cascade tuning (see [wake_cascade.md](architecture/wake_cascade.md)).
- Locality-preserving burst placement: bias re-scheduling of a repeating burst
  back onto the cores that last ran it (warm cache), analogous to TBB's
  self-steal.
- An opt-in pool hint/mode for latency-sensitive intermittent workloads that
  favors keep-alive over CPU yield.

**Cost / tradeoff.** Any keep-alive strategy burns CPU while "idle", so it must
be opt-in or bounded (e.g. decay after N idle periods) to avoid regressing the
genuinely-idle case and hurting battery/thermals on mobile. Validate with
`idle_pool_benchmark` on frequency-pinned runs (the retail-Pixel numbers above
are unpinned and carry extra variance).

### Recursive fork-join vs TBB task_group (deep / heavy trees)

**Context.** With fork-join scheduling complete (per-thread + steal rings),
dispenso is on par with `tbb::task_group` on shallow traversal — basic tree
~1.03x, kdtree ~1.01x, light-per-node work ~1.13x — but trails on deep and
heavy-per-node recursion: `tree_work` heavy-per-node ~0.71x and 4-ary deep
fork-join ~0.69x vs TBB (EPYC-Genoa 166c). It beats folly ~5.7x throughout, so
the gap is TBB-specific and is the clearest remaining fork-join weakness.

**Why.** TBB's continuation stealing keeps a deep recursion's working set on the
stealing core and avoids re-scheduling overhead at every node; dispenso
re-schedules each `parallel_invoke` / recursive TaskSet child through the pool.

**Possible approaches:**
- Continuation-style stealing for `parallel_invoke` / recursive TaskSet so a
  stolen child resumes its parent on the stealing thread (warm working set).
- A recursion-depth / subtree-size cutoff that runs deep sub-trees inline past a
  threshold — the fork-join analogue of `parallel_for`'s `minItemsPerChunk`
  guardrail, which already wins the trivial-work cases.
- Validate against `tbb::task_group` on the `tree_work` (heavy) and `4ary` suites.

### Topology-hierarchy-aware scheduling (beyond L3-as-NUMA-proxy)

**Context.** `kAdaptive` prefers same-L3 steal victims and `buildThreadGroups`
groups by L2/L3, using L3 as a lightweight NUMA proxy. Holds on AMD
(CCX ≈ L3 ≈ domain) but breaks where L3 ≠ memory domain: monolithic-L3 Intel SNC,
and virtualized hosts. Would give the "NUMA and topology awareness" backlog item
below a shared substrate.

**Sourcing (tested on the EPYC-Genoa dev VM).**
- **OS topology (Linux sysfs / FreeBSD sysctl) is the portable primary** — the
  only source that works on x86 *and* ARM and carries NUMA memory domains
  (firmware ACPI SRAT/SLIT; ARM ACPI PPTT). `CpuSet` already reads NUMA domains;
  the gap is that scheduling only consumes the L2/L3 levels.
- **x86 CPUID** (`0x1F`/`0x0B`, `0x8000001D`/`0x04`, AMD `0x8000001E`) is a
  bare-metal enrichment only, and needs a sanity check. On this VM it is fully
  flattened — no leaf `0x1F`, a synthetic 256-way L3, x2APIC IDs a flat `0..165`
  with no die/socket bits, `0x8000001E` reporting one node — so it adds nothing
  over sysfs, and is meaningless under vCPU migration anyway. `/proc/cpuinfo` is
  a weaker rendering of the same data, not a distinct source.
- **Core-to-core latency probe** (atomic cacheline ping-pong across pinned
  vCPUs) is the *only* method that saw through this VM: a clean two-tier split
  (~40 ns near vs ~200 ns far) that CPUID and sysfs both reported as flat. Worth
  an **opt-in, coarse, stability-gated** discovery mode — but numbers are ~10x
  inflated on a shared VM and drift with vCPU migration, so trust it only when
  placement is stable, and use it for CPU/cache grouping, not level labeling.
- CPU **model-string → SKU-layout** lookup is a last-resort bare-metal heuristic;
  the VM reports a generic "EPYC-Genoa" with no SKU, and a real SKU still would
  not reveal the vCPU→pCPU mapping.

**Actionability caveat.** In a NUMA-flattened guest you can act on CPU/cache
grouping (vCPU affinity) but **cannot** place memory in a hidden domain (only
node 0 exists to `mbind`); realistic VM payoff is thread grouping, not memory
locality. Proper validation needs bare metal (multi-CCX AMD NPS4 / Intel SNC).

**Goal.** A real hierarchy (socket ⊃ NUMA/SNC domain ⊃ L3 ⊃ L2/SMP) in
`CpuSet`, exposed as ordered levels + nearest-common-level/distance, consumed by
`buildThreadGroups` and kAdaptive victim ranking
(same-L2 > L3 > domain > socket > remote); prefer NUMA domains over L3, degrade
to the finest level the OS differentiates.

### Experiment: externalize per-group locality; simplify thread-pool tiers

**Context.** The pool has three tiers — per-thread rings, per-group steal rings,
central queue. `kAdaptive` already shows algorithms can own their locality
(stripes + own victim policy). Hypothesis: letting known-layout algorithms
(`parallel_for`/`reduce`/…) manage locality externally — **in addition to**, not
replacing, the steal rings — could give better-tailored behavior and trim
per-iteration `threadLoop` work (also helping the idle-pool CPU objective above).

**Keep.** Per-thread rings stay — the preferred path for bulk known-layout
scheduling of `parallel_for`/`reduce`/etc.

**Key risk.** Steal rings are the *universal* cross-algorithm work-stealing
fallback. If each algorithm only manages its own structure, a thread committed to
algorithm A's work cannot be stolen for algorithm B — risking load imbalance and,
for nested / `wait`-blocking patterns, potential **deadlock**. So steal rings
most likely remain as the safety net and external management layers on top,
rather than replacing them.

**Scope.** (1) Profile `threadLoopImpl` to quantify each tier's per-iteration
cost, so any simplification is justified by measured idle-path savings.
(2) Prototype an external placed-locality hook for `parallel_for`/`reduce`
alongside steal rings. (3) Prove cross-algorithm steal and deadlock-freedom hold.
Supersedes the "steal-ring round-robin for placed scheduling" backlog idea;
interacts with "Decouple sleep mask from group concept".

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

### Fork-Join Scheduling & Thread Groups (post-1.5) --- COMPLETE

Design documents:
- [three_tier_scheduling.md](architecture/three_tier_scheduling.md) — three-tier queue architecture
- [wake_cascade.md](architecture/wake_cascade.md) — leader-team parallel wake cascade

**Implemented.** Per-thread locality rings, per-group steal rings, leader-team
parallel wake cascade, CpuSet topology detection, adaptive spin backoff,
and tunable pool-recursive load factor. Locality-sensitive parallel_for
improved 74-86%. Pipeline sub-6ms. Graph scene CTS improved 18%. Dispenso
compositing benchmark 26-58x faster than Taskflow.

### 2.0 (API-breaking changes)

The following changes require a major version bump due to API breakage:

| Feature | Description |
|---------|-------------|
| Remove poll mode | Remove `threadLoopPoll`, the `setSignalingWake(false)` path, and the `DISPENSO_WAKEUP_ENABLE` / `DISPENSO_POLL_PERIOD_US` defines. Signaling wake (futex/WaitOnAddress/ulock) is always-on and well-tested. Poll mode is a legacy fallback that adds code complexity without benefit on any supported platform. |
| C++17 minimum | Bump minimum standard from C++14 to C++17. Enables `std::optional` (replacing `OpResult`), `if constexpr`, structured bindings, `[[nodiscard]]`, and `std::string_view`. Simplifies template metaprogramming throughout. |
| C++20 consideration | Evaluate C++20 as minimum for a later 2.x release. Enables `std::atomic<T>::wait/notify` (potential EpochWaiter simplification), concepts (better error messages), coroutine integration, and `std::jthread`. |
