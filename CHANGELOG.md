1.1 (October 1, 2022)

### performance and functionality enhancements

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
