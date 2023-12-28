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
