<!-- Part of the dispenso roadmap; index at ../roadmap.md -->

# Roadmap: allocators

`SmallBufferAllocator`, `PoolAllocator`, and a possible scalable allocator.

## Planned

| Feature | Description | Priority |
|---------|-------------|----------|
| Scalable allocator | Custom allocator for containers like ConcurrentVector. System allocators show opposing strengths: tcmalloc is best for ConcurrentVector's geometrically-growing variable-size buffers under contention, while jemalloc is best for fixed-size small allocations (SBA's pattern) but worst for ConcurrentVector. A purpose-built allocator can optimize for dispenso's specific patterns rather than relying on any single system allocator. | Medium |

NUMA-aware allocation is tracked with the rest of the NUMA work in
[core_scheduling](core_scheduling.md).

## Backlog

### Code TODOs

Open `TODO` comments in the source, each named by file and function. See
[the README](../README.md#conventions) for how these are kept in step.

- **Return memory from `SmallBufferAllocator`** (`SmallBufferAllocator::recycleToCentralStore`, `dispenso/detail/small_buffer_allocator_impl.h`). Recycled buffers go to the central store and are never returned to the system. Decide whether some garbage collection is needed.
- **Return memory from `PoolAllocator`** (`PoolAllocatorT::dealloc`, `dispenso/pool_allocator.cpp`). Memory is released only at destruction. When ready chunks have stayed above some threshold for a while, the allocator could start tracking allocations and reassemble whole backing allocations to free. That makes allocation slower, but avoids keeping a burst's peak memory forever.

## Open investigations

- [SmallBufferAllocator vs system malloc](../investigations/smallbufferallocator_vs_system_malloc.md)
- [PoolAllocator thread-local optimization](../investigations/poolallocator_thread_local.md)
