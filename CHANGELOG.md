1.6.2 (August 19, 2026)

### Bug fixes
* `DISPENSO_USE_SYSTEM_CONCURRENTQUEUE=ON` builds again. 1.6.1 required `find_package(concurrentqueue 1.0.5 CONFIG REQUIRED)`, which no installed concurrentqueue can satisfy: upstream's `CMakeLists.txt` reads `project(concurrentqueue VERSION 1.0.0)` at every release including v1.0.5, so the config version file it installs always reports `1.0.0` -- a value matching no released tag. A correct 1.0.5 installation was rejected, so the vcpkg port and any source build against the upstream CMake package failed to configure. The version argument has been removed; the ABI concern that motivated it is real but cannot be expressed through this package, and the code now says so.

### Build system
* Added a CI job covering `DISPENSO_USE_SYSTEM_CONCURRENTQUEUE=ON`. It is the configuration the vcpkg and conan ports ship and nothing exercised it, so a break reached a release before anyone noticed.

1.6.1 (August 18, 2026)

### Bug fixes
* Fixed an access violation at process exit in Windows shared-library builds. `NewThreadInvoker` registered its thread drain with `atexit()`, which in a shared build belongs to the dispenso DLL and therefore runs at `DLL_PROCESS_DETACH` -- after `ExitProcess` has already terminated every other thread, so the drain never saw the threads it exists to join. The drain is now registered from a static in the calling module, whose destructors run during ordinary exit processing while those threads are still alive and joinable.

### Build system
* Added `DISPENSO_WERROR` (default ON) to control whether warnings are fatal. Warnings themselves are unconditional; this only governs `-Werror`, so distributors building against compilers we never see can stop a new warning from failing their build. It could not be worked around before: the flags come from `target_compile_options` and land after `CMAKE_CXX_FLAGS`.
* The bundled moodycamel headers now install to `<prefix>/include/moodycamel` rather than `<prefix>/include/dispenso/third-party/moodycamel`. `dispenso/thread_pool.h` and `dispenso/resource_pool.h` include `<moodycamel/concurrentqueue.h>`, which previously resolved only through a second include directory carried on the CMake target -- so an installed dispenso could not be consumed with a plain `-I<prefix>/include`. The bundled and system layouts are now identical.
* The bundled moodycamel directory is added as a SYSTEM include, so our warning flags no longer diagnose third-party code.
* `DISPENSO_USE_SYSTEM_CONCURRENTQUEUE=ON` now requires concurrentqueue 1.0.5 or newer instead of accepting any installed version. `moodycamel::ConcurrentQueue` is a by-value member of `ThreadPool`, so its layout is part of dispenso's ABI.

### Dependencies
* Updated the bundled moodycamel concurrentqueue to v1.0.5, from a 2022 snapshot. The copy is now byte-identical to upstream, with no local patches.

### Documentation
* The published documentation site is now built with the same pinned Doxygen version that CI validates, and fails on Doxygen warnings. Previously it was built by an action whose Doxygen version was independent of the pin, so a version in the broken 1.9.2-1.9.8 range could publish unresolved cross-page links silently.
* Corrected the README's statement of the bundled moodycamel licenses, which omitted the Boost option and overstated the scope of the Zlib terms.

1.6.0 (August 13, 2026)

### New features
* **`ChaseLevDeque`** — lock-free single-producer multi-consumer work-stealing deque with dynamic resizing. Classic data structure for work-stealing schedulers.
* **`MpmcRingBuffer`** — bounded multi-producer multi-consumer ring buffer with power-of-two capacity and CAS-based push/pop.
* **`CpuSet`** — portable CPU affinity and NUMA topology facility. Supports thread-to-core binding, L2/L3 cache group detection, and cache-aware thread group building. Full support on Linux, Windows, and FreeBSD; topology-only on macOS.
* **`parallel_invoke`** — fork-join invocation of heterogeneous tasks. Schedules N-1 tasks to the pool and runs the last inline. Composes naturally with recursive divide-and-conquer.
* **`kAdaptive` parallel_for** — new chunking strategy inspired by Callisto-RTS (Harris/Kaestle, USENIX ATC 2015). The iteration space is partitioned into P contiguous stripes (one per worker), each consumed front-to-back via per-stripe atomic cursors. When a worker's stripe is exhausted, it steals from peers, preferring same-L3 victims for cache locality. Bitmasks prevent probing exhausted stripes. Competitive with TBB on SpMM benchmarks (3–12% faster at 8–32 threads, within noise at 64–192 threads on 1M-row workloads).
* **`when_any` combinator** — returns a future that completes when any input future is ready, with the index of the first completed future.
* **`DistributedRWLock`** — sharded reader-writer lock that spreads reader traffic across per-shard state to avoid a single contended cache line, with an OS-level writer drain rather than a spin. Defaults to 16 shards.
* **`granularity` option for `parallel_for`** — bounds the smallest unit of work a chunking strategy will hand to a worker, so loops with expensive per-iteration bodies can stop the scheduler from subdividing past the point where the split costs more than the work.
* **FreeBSD support** — native thread-pool wait/wake via the `_umtx_op` syscall, plus `CpuSet` CPU-affinity, NUMA-domain, and L2/L3 cache-topology queries built on `cpuset_getaffinity` and the `kern.sched.topology_spec` sysctl. (thanks bimokh!)

### Thread pool rework
* **Per-thread rings** — each worker thread gets a dedicated SPMC ring buffer (16 slots). `schedule()` distributes work round-robin across rings, eliminating central queue contention at high thread counts. Foundation for fork-join scheduling — threads check their own ring before the central queue.
* **Steal-ring scheduling** — shared steal rings (one per `kStealRingSharing` threads) provide a secondary work-distribution tier between the per-thread ring and the central queue, enabling same-group work stealing without central queue CAS contention.
* **Wake cascade** — replaced `wakeN()` bitmask scanning with a promote-seed cascade pattern (O(log N) wake latency). One thread is woken and propagates wakes through the group via pre-staged lambdas in per-thread rings. Pattern C cascade is within ±5% of the old scheme on all benchmarks while being 6x faster on the mandelbrot workload (reverting to old wakeN was +615% on mandelbrot).
* **Lean-spin warmup** — idle threads first check only their own ring (no central queue, no CAS contention) for `kSpinCheckInterval` iterations before engaging full work-finding machinery. Minimizes cache-line traffic from idle threads during sustained parallel_for bursts.
* **Fixed-spin termination** — replaced adaptive time-based spin backoff with a simple fixed iteration count (`kDefaultSpinLimit`). Eliminates `getTime()` calls from the idle path. Windows defaults to 200 iterations; Linux/macOS to 400. Platform constants are tunable via `-DDISPENSO_TUNE_FIXED_SPIN_ITERS` and `-DDISPENSO_TUNE_SPIN_CHECK_INTERVAL`.
* **Separate poll-mode / wake-mode timeouts** — poll-mode timeout (200µs Linux, 1ms Windows) controls how often threads check for work when wake signaling is disabled. Wake-mode backstop (100ms) bounds worst-case latency from rare races. Previously these shared a single confusing constant.
* **Unified thread loop** — `threadLoopWake` and `threadLoopPoll` merged into a single `threadLoopImpl<kUseWakeSleep>` template, eliminating code duplication.
* **`PoolWakeState` moved to `detail/`** — wake infrastructure is now properly namespaced under `dispenso::detail`.

### Performance improvements
* `OnceFunction` inline SBO storage: small functors (≤56 bytes) stored inline in a 64-byte cache-line-sized object, eliminating pool allocation for the common case. Bulk scheduling 19–49% faster (allocation elimination); futures tree 4–12% faster; graph scene 2–11% faster. Medium/large functor moves are 29–58% slower as expected (64B vs 16B copy), but these are rare in practice. (166-thread EPYC)
* `TaskCost` routing: dual-path dispatch for lightweight vs heavyweight tasks, reducing scheduling overhead for small tasks.
* Windows spin tuning: `spin_fixed_200` was ~6% faster (geomean across full tuning set) than the adaptive baseline on a 48-thread Xeon Platinum 8259CL.

### Bug fixes
* Fixed `try_pop_into` race in `MpmcRingBuffer`: read slot value before CAS in the last-element case to prevent use-after-overwrite.
* Fixed `NewThreadInvoker` rare Windows hang on process exit: pinned the module and bounded the shutdown drain.
* Fixed a lost wakeup that could strand a scheduled task indefinitely: `centralQueueNonEmpty_` is a cheap hint that lets spinning workers skip the central queue, but a worker clearing it after a failed dequeue could overwrite a concurrent producer's store, leaving that task queued while every worker believed the queue was empty. The sleep-timeout backstop could not recover it, because the wakeup path consults the same hint. A worker that wakes on the timeout rather than a signal now re-checks the queue and repairs the hint, bounding a strand to one sleep period.
* Fixed `scheduleBulkImpl` inline boundary oscillation: prevented enqueue/inline mode switching on every iteration.
* Fixed `-Wconversion` warning on narrow `parallel_for` index types.
* Fixed Doxygen warnings in OSS CI, and stopped extracting `NewThreadInvoker`'s private thread-tracking internals, which were reported as undocumented compounds.
* Fixed `util_test` warning suppression for Clang and GCC. The GCC arm is now restricted to GCC 13 and newer, which is where `-Wself-move` was introduced; on GCC 11 and 12 the unknown option made `#pragma GCC diagnostic` itself a warning and `-Werror=pragmas` turned that into a build failure.
* Fixed narrowing conversions in `EpochWaiter::waitFor`.
* Fixed OSS cross-platform build breaks on 32-bit targets and MSVC x86/x64, and an MSVC C3493 capture error.
* Fixed a `TaskSetTest` `TEST`/`TEST_P` suite-name collision.
* Replaced `rdtscp` with `lfence; rdtsc` in timing code — some simulators don't support `rdtscp`.
* Restored `ThreadWaiter` to make `NewThreadInvoker` safe at process exit.
* Added `static_assert` for trivially-copyable types in `ConcurrentObjectArena` copy constructor.
* Fixed `RWLock` livelock under oversubscription: pure spin loops in `lock_shared()`, `setWriteBit()`, and `lock_upgrade()` now yield after 256 iterations, preventing permanent writer starvation when thread count exceeds physical cores.

### fast_math (experimental)

**Still experimental.** The API is unstable and will change in future releases; it
remains gated behind the `DISPENSO_BUILD_FAST_MATH` CMake option. Do not depend on
signatures, header layout, or accuracy traits staying put.

* **SIMD backends now require FMA.** SIMD detection is centralized in one place, and a
  backend is only selected when the target provides fused multiply-add. Targets without
  FMA fall back to the scalar path rather than silently selecting a backend whose
  polynomials were tuned assuming fused arithmetic. This can change which backend is
  chosen for an existing build.
* Added functions: `pow`, `hypot`, `expm1`, `log1p`, `tanh`, `erf`, `sincos`, `sinpi`,
  `cospi`, `sincospi`.
* Added `rsqrt_approx`, `rsqrt`, `rcp_approx`, `rcp` with configurable accuracy.
* Improved accuracy using Sollya-generated minimax polynomials for `exp`, `exp10`,
  `asin`, `atan`, `acos`, `expm1`, `log1p`, and `tanh`. `cbrt` gained an FMA correction
  step, denormal safety, and a native SIMD `int_div_by_3`.
* Unified scalar and SIMD dispatch for `sin`/`cos`.
* Added SIMD building blocks: `shuffle`, `maskBits`, `maskLoad`/`maskStore`, `extract`,
  `testBit`, `kLanes`, `load`/`store`, `any_true`.
* Added `polyEval`, `hornerEval`, and `estrinEval` polynomial-evaluation abstractions.
* Added CUDA compilation support and exhaustive GPU correctness tests. **CUDA support is
  correctness-tested only — it has had no performance testing whatsoever.** Treat the GPU
  path as unmeasured for speed.
* Added unified SIMD test infrastructure covering accuracy and special values for both
  single- and two-argument functions, plus a bivariate ULP evaluation harness using
  Halton sampling.
* Fixed `log(-0.0f)` and `exp` overflow bounds handling.
* Truncated `exp10`'s Cody-Waite constant so the low-order term carries meaning.
* Ordered `expm1`'s bounds checks ahead of its `n == 0` shortcut, so out-of-range inputs
  are clamped before the shortcut can return an unclamped result.
* Documented that `convert_to_int`'s result for non-finite input is unspecified and must
  not be branched on.
* Relaxed test bounds where the hardware justifies it: one extra ULP for MSVC, and looser
  `rcp_approx`/`rsqrt_approx` bounds on ARM64.

### Benchmarks
* Added `mandelbrot_benchmark` — escape-time workload for testing dynamic load balancing with non-uniform per-pixel work.
* Added `spmm_benchmark` — sparse matrix-dense matrix multiply with power-law row distribution, realistic density (<1%), and 64 RHS columns.
* Added `mandelbrot_instrument` — standalone scheduler quality analysis tool (chunk distribution, affinity hit rate). Moved to `tools/`.
* Added `wake_cost_bench` — platform syscall cost measurement for wake tuning. Moved to `tools/`.
* Added benchmark runner infrastructure for Windows/Buck (`facebook/run_benchmarks_buck.py`), including resume support for interrupted runs.
* Added an Android (adb) benchmark runner that cross-compiles, pushes to a device, and emits the same result schema as the other runners.
* Added Windows tuning sweep results and data.
* Replaced the matplotlib chart pipeline (`generate_charts.py`, `update_benchmarks.py`) with an interactive Plotly dashboard (`scripts/generate_plotly_benchmarks.py`, output at `docs/benchmarks/index.html`).
* Results now record what produced them: compiler and C++ standard, CPU SKU or generation, and the versions of the comparison libraries (TBB, Taskflow, OpenMP, google-benchmark). The dashboard shows these per platform, so a number can be traced to the toolchain and libraries it was measured against.
* Per-benchmark timeouts scale with core count instead of using a flat value, so thread-scaling suites are not truncated on many-core machines.
* Trimmed the `mandelbrot` thread sweep and capped `rw_lock`'s write-heavy cases, cutting total runtime substantially without losing the shape of either curve.
* Refreshed the published results for all four platforms: Linux x64 (166-core EPYC Genoa), Windows x64 (60-core Xeon Gold), macOS ARM64 (M4 Pro), and Android ARM64 (Pixel 9 Pro XL).

### Documentation
* Added examples for `CpuSet`, `ChaseLevDeque`, `parallel_invoke`, `SmallVector`, `MpmcRingBuffer`, `SPSCRingBuffer`, `when_any`, and `kAdaptive` chunking.
* Documented the new public APIs in the Getting Started guide (containers, `when_any`, and adaptive chunking) and the README feature list.
* Added all missing public headers to `dispenso.h` umbrella include.
* Added C++20 concept constraints for `parallel_for` and graph SFINAE.
* Added a FAQ page and trimmed the README to improve the onboarding path.
* Standardized `#pragma once` placement across all headers.
* Documented the C++ standard choice in [docs/building.md](docs/building.md#c-standard): C++14 remains fully supported and is the CMake default, while C++17 replaces the bundled compatibility shims with standard facilities and C++20 turns the `DISPENSO_REQUIRES` constraints into real concepts.

### Build system and infrastructure
* Added CMake build for `tools/` directory (development utilities).
* Added three missing benchmark targets to Buck benchmark runner.
* Bumped GitHub Actions checkout to v5 (Node 20 deprecation).
* Removed dead `run_bench.bat` and duplicate `benchmarks/run_benchmarks.py`.
* Publish the benchmark dashboard to GitHub Pages, with the workflow permissions the deploy requires.
* Bumped CodeQL to v4.
* Gated the `for_each` list/set benchmarks behind `DISPENSO_BENCH_ALL_CONTAINERS`.
* The documentation build now requires Doxygen 1.11.0 or newer, verified in CI. Releases 1.9.2 through 1.9.8 fail to resolve markdown links to pages carrying an explicit `{#label}` anchor. The CI job installs a pinned Doxygen, checked against a pinned SHA-256, rather than whatever the runner image happens to ship.

1.5.1 (March 28, 2026)

### Bug fixes
* Fixed `__ulock_wait`/`__ulock_wake` usage on macOS versions prior to 10.12 and on PowerPC where these APIs are unavailable. The ulock path is now guarded behind a runtime version check with `pthread_cond` fallback.
* Fixed ARM64 Windows build failure: `notifier_common.h` incorrectly defined `_ARM_` (32-bit ARM) instead of `_ARM64_` on ARM64 Windows, causing `winnt.h` to reference missing 32-bit ARM intrinsics.
* Fixed `platform.h` version macros not being updated for 1.5.0 release (were stuck at 1.4.1)
* Removed vestigial `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS` from CMakeLists.txt. All public APIs now use proper `DISPENSO_DLL_ACCESS` annotations; the blanket export is no longer needed and is prohibited by vcpkg's maintainer guide.

### Build system
* Added `DISPENSO_USE_SYSTEM_CONCURRENTQUEUE` CMake option to use system-installed `moodycamel::concurrentqueue` instead of bundled copy (default OFF), for vcpkg compatibility
* Export C++ standard requirement via `target_compile_features` so downstream consumers compile with at least the same standard dispenso was built with
* Respect `BUILD_SHARED_LIBS` for `DISPENSO_SHARED_LIB` default, allowing vcpkg to control static/shared linkage

### Infrastructure
* Added package manager release automation script (`scripts/update_package_managers.py`) with post-write checksum verification, platform-aware testing, and PR body templates following each repo's CONTRIBUTING.md
* Added CodeQL security analysis workflow scoped to main branch
* Added package manager badges (vcpkg, Conan, Homebrew, MacPorts) to README
* Added release checklist documentation

1.5.0 (March 22, 2026)

### New features
* Added `SmallVector` container with configurable inline storage, reducing heap allocations for small collections
* Added `SPSCRingBuffer` lock-free single-producer single-consumer ring buffer with power-of-two optimization
* Added `scheduleBulk(count, generator)` API to ThreadPool, TaskSet, and ConcurrentTaskSet for efficient bulk task submission with reduced atomic contention
* Added random-access iterator specialization for `for_each_n`, with iterator category dispatch for optimal chunk boundary computation
* Added Mac futex-based wakeup using `os_sync_wait_on_address` (macOS 14.4+) with `__ulock_wait` fallback
* Added C++20 concept constraints for better error messages when template requirements aren't met
* Added experimental `fast_math` sublibrary with SIMD-accelerated math functions including `log2`, `exp2`, `exp`, `exp10`, `cbrt`, `sin`, `cos`, `asin`, and `atan2` with configurable accuracy/performance trade-offs and multiple SIMD backends (SSE4.1, AVX2, AVX512, NEON, Highway). **API unstable** — gated by `DISPENSO_BUILD_FAST_MATH` CMake option
* Added benchmark runner and chart generation scripts with multi-platform support
* Added interactive Plotly.js benchmark dashboard generator

### Performance improvements
* ThreadPool atomics simplification: replaced 3 per-task tracking atomics with `numSleeping_` + batched `workRemaining_` decrements, reducing per-task atomic operations from 5-6 to ~1 (+24% geometric mean across 568 benchmark tests)
* ThreadPool wakeup heuristic: reduced futex calls from ~1M to ~11K by only waking when capacity cannot cover queued work; `mostly_idle` benchmark 2.6x faster
* Cache-line alignment for `poolLoadFactor_` and `numThreads_` to reduce false sharing (L1 cache miss rate 16.33% → 6.92% on `schedule()` hot path)
* Graph executor optimizations: `SmallVector` for node dependents, pre-reserve capacity, inline continuation (build_big_tree 1.95x faster, build_dep_chain 2.14x faster)
* Serial pipeline SPSC optimization: dedicated executor with ring buffers for fully-serial pipelines (~33% faster)
* Inline continuation for serial pipeline stages (scheduling overhead reduced ~3x)
* Bulk wakeup with threshold-based `wakeN()`/`wakeAll()` selection for efficient bulk scheduling
* `for_each_n` converted to `scheduleBulk`: 2.1x faster at 32 threads, 1.6x at 64 threads for 100M elements
* `parallel_for` kAuto bulk scheduling: trivial_compute 52ms → 19ms at 192 threads (matching TBB)
* ConcurrentVector: non-atomic buffer pointer cache for read-hot paths (disabled on ARM), inline asm `bsr` for `detail::log2` on x86, and platform-adaptive `bucketAndSubIndexForIndex` fast path
* `OnceFunction` devirtualized: replaced vtable-based dispatch with direct function pointer, eliminating indirect call overhead
* `TaskSet`/`ConcurrentTaskSet` noWait path: replaced `shared_ptr<Atomic>` with pool-allocated single-atomic chunk index, reducing allocation overhead

### Infrastructure
* Benchmark runner (`run_benchmarks.py`) with JSON output and machine info collection
* Chart generator (`generate_charts.py`) with specialized visualizations per benchmark suite
* Multi-platform benchmark composition (`update_benchmarks.py --compose`) for unified documentation
* Prefer system-installed GoogleTest, Taskflow, and TBB in CMake with FetchContent fallback
* Added oneTBB compatibility via `tbb_compat.h` wrapper for `task_scheduler_init`
* Added BUCK targets for `idle_pool_benchmark`, `nested_pool_benchmark`, `for_each_benchmark`, and `locality_benchmark`

### Documentation
* Added examples directory with compilable example programs for each feature
* Added Getting Started guide (`docs/getting_started.md`) with inline code snippets from examples
* Added OpenMP migration guide (`docs/migrating_from_openmp.md`)
* Improved README clarity, discoverability, and feature descriptions

### CI and build improvements
* Comprehensive CI matrix: 11 jobs covering 3 architectures (x64, x86, ARM64), 3 OSes, 3 compilers (GCC, Clang, MSVC), C++14/20, TSan/ASan, code coverage, and Doxygen builds
* Added codecov.yml for enforcing 92% code coverage threshold

### Bug fixes
* Fixed ABI mismatch between exception and no-exception builds: `TaskSetBase` and `FutureImplResultMember` had conditionally compiled members that shifted struct layout depending on `-fno-exceptions`, causing crashes when translation units disagreed. Exception-related data members are now always present in the layout (with zero runtime cost when exceptions are disabled). **Note:** this changes the ABI for builds that previously used `-fno-exceptions`; recompile all code against 1.5 headers if mixing exception modes.
* Fixed `ConcurrentTaskSet` parent stack overflow when tasks recursively schedule to the same task set: self-recursive inlining via `tryExecuteNext()` repeatedly pushed the same `TaskSetBase*` onto the thread-local parent stack (depth limit 64), causing an abort under heavy inlining. Fix skips redundant push/pop when the TaskSet is already the current parent.
* Fixed pipeline `kLimited` scheduler `wait()` losing late-arriving items: the LIMITED path only drained the local queue without waiting for in-flight items, so items enqueued by a previous stage's CTS task after the drain could be permanently orphaned. Fix replaces the drain-only loop with an `outstanding_`-based spin that ensures all items complete, with `tryExecuteNext()` to keep the calling thread productive.
* Fixed `parallel_for` with `kAuto` chunking incorrectly falling back to static chunking when `maxThreads` was left at default
* Fixed `NoOpIter` missing iterator trait typedefs for C++20 compliance
* Fixed `NoOpIter::operator*()` / `operator[]` static local data race
* Fixed SmallBufferAllocator unsigned underflow where `allocSmallBuffer<1/2/3>()` returned nullptr instead of a 4-byte block
* Fixed `cpuRelax()` being a no-op on MSVC (missing `_mm_pause()` / `__yield()` intrinsics)
* Fixed x86 Windows build issues
* Fixed Doxygen documentation warnings
* Fixed pipeline exception safety: exceptions thrown in pipeline stage functors are now caught and propagated to the caller via `ConcurrentTaskSet`. Added RAII guards for stage resource cleanup, `OnceFunction::cleanupNotRun()` for proper deallocation of unexecuted tasks, and a deadlock fix in the `kLimited` scheduler's resource spin loop when exceptions leave no threads to release resources.
* Fixed MSVC lambda capture for constexpr variable
* Fixed `idle_pool_benchmark` fairness (loop bound, static scheduling, and pool placement)
* Fixed `nested_for_benchmark` incorrect loop bound, static scheduling, and pool placement

### Test improvements
* Added comprehensive tests for thread_pool spin-poll with sleep mode
* Added comprehensive task_set edge case tests
* Added comprehensive tests for concurrent_object_arena
* Added edge case tests for pool_allocator
* Added timing tests for getTime() function
* Added tests for Graph/Subgraph accessors and BiPropNode edge cases
* Added `SmallVector` test suite (43 tests)
* Added `SPSCRingBuffer` test suite (47 tests)
* Improved overall test coverage from ~89% to 96.3% (dispenso source only, excluding stdlib and third-party)

1.4.1 (January 5, 2026)

### Bug fixes and build improvements
* Fixed clock frequency calculation for mac-arm platforms
* Addressed potential race condition at TimedTaskScheduler construction
* Adjusted build platforms for better compatibility

1.4 (January 2, 2025)

### Efficiency improvements, bug and warning fixes
* Added some benchmarks and comparison with TaskFlow (thanks andre-nguyen!)
* Fixed compilation when compiling with DISPENSO_DEBUG (thanks EscapeZero!)
* Improved efficiency on Linux for infrequent thread pool usage.  Reduces polling overhead by 10x by switching to event-based wakeup instead of spin polling.
* Fix C++20 compilation issues (thanks aavbsouza!)
* Fix several build warnings (thanks SeaOtocinclus!)
* Add conda package badge, disable gtest install (thanks JeongSeok Lee!)
* Solved rare post-main shutdown issues with NewThreadInvoker
* Fixed test issues for 32-bit builds
* Fixed broken test logic for test thread IDs
* Fixed various build warnings

1.3 (April 25, 2024)

### Bug fixes, portability enhancements, and small functionality enhancements

* Fixed several generic warnings (thanks michel-slm!)
* cpuRelax added for PowerPC and ARM (thanks barracuda156!)
* Added missing header (thanks ryandesign!)
* Try to detect and add libatomic when required (thanks for discussions barracuda156!)
* Enable small buffers from small buffer allocators to go down to 4 bytes (thanks for discussion David Caruso!).  This is handy for 32-bit builds where pointers are typically 4 bytes
* Ensure that NOMINMAX is propagated for CMake Windows builds (thanks SeaOtocinclus!)
* Fix some cases using std::make_shared for types requiring large alignment, which is a bug prior to C++17 (thanks for help finding these SeaOtocinclus!)
* Set up CI on GitHub Actions, including builds for Mac and Windows in addition to Linux (thanks SeaOtocinclus!)
* Add an envinronment variable `DISPENSO_MAX_THREADS_PER_POOL` to limit max number of threads available to any thread pool.  In the spirit of `OMP_NUM_THREADS`.  (thanks Yong-Chull Jang!)
* Slight change of behavior w.r.t. use of `maxThreads` option in `ForEachOptions` and `ParForOptions` to limit concurrency the same way in both blocking and non-blocking `for_each` and `parallel_for` (thanks Arnie Yuan!)
* Various fixes to enable CMake builds on various 32-bit platforms (thanks for discussions barracuda156!)
* Updates to README

Known Issues:
* Large subset of dispenso tests are known to fail on 32-bit PPC Mac.  If you have access to such a machine and are willing to help debug, it would be appreciated!
* NewThreadInvoker can have a program shutdown race on Windows platforms if the threads launched by it are not finished running by end of main()

1.2 (December 27, 2023)

### Bug fixes and functionality enhancements

* Several small bug fixes, especially around 32-bit builds and at-exit shutdown corner cases, and TSAN finding benign races and/or causing timeout due to pathological lock-free behaviors in newer versions of TSAN
* Improve accuracy of `dispenso::getTime`
* Add C++-20-like `Latch` functionality
* Add mechanism for portable thread priorities
* Add a timed task/periodically scheduled task feature.  Average and standard deviation of the accuracy of `dispenso::TimedTaskScheduler` are both much better than `folly::FunctionScheduler` (from 2x to 10x+ depending on settings and platform)
* Enhancements to `parallel_for`
  * Add an option that allows to automatically reduce the number of threads working on a range if the work is too cheap to justify parallelization.  This can result in 3000x+ speedups for very lightweight loops
  * Resuse per-thread state containers across parallel for calls (these must block in-between, or be thread-safe types)
  * `parallel_for` functors may now be called with an input range directly instead of requiring a ChunkedRange.  This is as simple as providing a functor/lambda that takes the additional argument, just as was previously done with `ChunkedRange`.  `ChunkedRange`s still work, and this is fully backward compatible
* `ThreadPool`s have a new option for full spin polling.  This is generally best avoided, and I'd argue never to use this for the default Global thread pool, but can be useful for a subset of threads in systems that require real-time responsivity (especially, can be combined with the thread priority feature also found in this release)
* Task graph execution (thanks Roman Fedotov!).  Building and running dispenso task graphs is typically 25% faster than the (already excellent) `TaskFlow` library in our benchmarks.  Additionally, we have a partial update feature that can enable much faster (e.g. 50x faster) execution in cases where only a small percentage of task inputs are updated (think of per-frame partial scene updates in a game)

1.1 (October 1, 2022)

### Performance and functionality enhancements

* CMake changes to allow install of targets and CMake dispenso target exports (thanks jeffamstutz!)
* Addition of typical container type definitions for ConcurrentVector (thanks Michael Jung!)
* Large performance improvements for Futures and CompletionEvents on MacOs.  Resulted in order-of-magnitude speedups for those use cases on MacOs.
* Addition of new benchmark for performance with infrequent use of `parallel_for`, `for_latency_benchmark`
* Fixes to ensure `parallel_for` works with thread pools with zero threads (thanks kevinbchen!).  Further work has been done to ensure that thread pools with zero threads simply always run code inline.
* By default, the global thread pool uses one fewer thread than the machine has hardware threads.  This behavior was introduced because dispenso very often runs on the calling thread as well as pool threads, and so one fewer thread in the pool can lead to better performance.
* Update googletest version to 1.12.1 (thanks porumbes!)
* Add a utility in dispenso to get a thread ID, `threadId`.  These 64-bit IDs are unique per thread, and will not be recyled.  These values grow from zero, ensuring the caller can assume they are small if number of threads also is small (e.g. you won't have an ID of `0xdeadbeef` if you only run hundreds or thousands of threads in the lifetime of the process).
* Add a utility, `getTime`, to get time quickly.  This provides the double-precision time in seconds since the first call to `getTime` after process start.
* Use a new scheduling mechanism in the thread pool when in Windows.  This resulted in up to a 13x improvement in latency between putting items in the pool and having those items run.  This scheduling is optional, but turned off for Linux and MacOs since scheduling was already fast on those platforms.
* Optimizations to enable faster scheduling in thread pools.  This resulted in a range of 5% to 45% speedup across multiple benchmarks including `future_benchmark` and `pipeline_benchmark`.
* Fixed a performance bug in work stealing logic; now dispenso outperforms TBB in the `pipeline_benchmark`
* Added a task set cancellation feature, with a relatively simple mechanism for submitted work to check if it's owning task set has been cancelled.  When creating a task set, you can optionally opt into parent cancellation propagation as well.  While this propagation is fairly efficient, it did create a noticeable impact on performance in some cases, and thus it was decided to allow this behavior, but not penalize performance for those who don't need the behavior.

1.0 (November 24, 2021)

### dispenso initial release
