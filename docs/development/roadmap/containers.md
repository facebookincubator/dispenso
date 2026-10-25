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

## Backlog

- Lock-free stack

### Code TODOs

Open `TODO` comments in the source, each named by file and function. See
[the README](../README.md#conventions) for how these are kept in step.

- **Runtime capacity for `SPSCRingBuffer`** (`dispenso/spsc_ring_buffer.h`): a `kDynamicCapacity` specialization that heap-allocates storage of a capacity chosen at run time.
