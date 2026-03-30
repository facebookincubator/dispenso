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
* Added experimental `fast_math` sublibrary with SIMD-accelerated math functions including `log2`, `exp2`, `exp`, `exp10`, `cbrt`, `sin`, `cos`, `sincos`, `asin`, and `atan2` with configurable accuracy/performance trade-offs and multiple SIMD backends (SSE4.1, AVX2, AVX512, NEON, Highway). **API unstable** — gated by `DISPENSO_BUILD_FAST_MATH` CMake option
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
