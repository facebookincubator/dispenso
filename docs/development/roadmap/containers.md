<!-- Part of the dispenso roadmap; index at ../roadmap.md -->

# Roadmap: containers

Concurrent and small-buffer containers.

Current design: [concurrent_vector](../architecture/concurrent_vector.md).

## Planned

| Feature | Description | Status |
|---------|-------------|--------|
| ConcurrentHashMap | High-value concurrent container | In progress |
| ConcurrentQueue | Public API for blocking MPMC queue | Medium |

## ConcurrentVector follow-on

- **Server ARM (Graviton) benchmarking**: The `DISPENSO_HAS_CACHED_PTRS`
  guard currently disables the cache on all ARM (`__aarch64__`). If server ARM
  shows different cache pressure characteristics than mobile ARM, a more
  targeted guard could re-enable it selectively.
- **Scalable allocator** (see [allocators](allocators.md)): concurrent growth
  benchmarks show 3-5x improvement with tcmalloc/jemalloc vs glibc malloc,
  suggesting a thread-caching allocator would benefit all trait combinations.
- **Serial `push_back`.** Without `reserve`, appending 1M elements one at a
  time takes 8.4 ms on Linux (EPYC) against 6.8 ms for
  `tbb::concurrent_vector`, 2.4 ms for `std::vector` and 0.9 ms for
  `std::deque`; on Windows and macOS dispenso and TBB are level, and on Android
  dispenso is faster. With `reserve` dispenso drops to 3.0 ms on Linux. Linking
  tcmalloc speeds `ConcurrentVector` up but slows other allocations inside
  dispenso, so the fix is a custom allocator for its buffers rather than
  recommending a system allocator; see the scalable allocator in
  [allocators](allocators.md).

## Backlog

- Lock-free stack
- **`ConcurrentObjectArena` versus `ConcurrentVector`.** `ConcurrentVector` also grows concurrently
  without moving its elements, so it can serve as an arena. Benchmark the two on the arena's
  workloads, concurrent `grow_by()` and indexed access; there is no arena benchmark yet. If the
  arena has no performance advantage, deprecate it in favour of `ConcurrentVector` and remove it in
  2.0. ThreadPool's `rings_` and `stealRings_` would then move to `ConcurrentVector` or an internal
  type, keeping their cache-line alignment. Either way, the arena's copy operations go in 2.0 (see
  [language_and_api](language_and_api.md)).

### Code TODOs

Open `TODO` comments in the source, each named by file and function. See
[the README](../README.md#conventions) for how these are kept in step.

- **Runtime capacity for `SPSCRingBuffer`** (`dispenso/spsc_ring_buffer.h`): a `kDynamicCapacity` specialization that heap-allocates storage of a capacity chosen at run time.
