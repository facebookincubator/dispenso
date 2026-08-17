<!-- Part of the dispenso roadmap; index at ../roadmap.md -->

# Roadmap: allocators

`SmallBufferAllocator`, `PoolAllocator`, and a possible scalable allocator.

## Planned

| Feature | Description | Priority |
|---------|-------------|----------|
| Scalable allocator | Custom allocator for containers like ConcurrentVector. System allocators show opposing strengths: tcmalloc is best for ConcurrentVector's geometrically-growing variable-size buffers under contention, while jemalloc is best for fixed-size small allocations (SBA's pattern) but worst for ConcurrentVector. A purpose-built allocator can optimize for dispenso's specific patterns rather than relying on any single system allocator. | Medium |

NUMA-aware allocation is tracked with the rest of the NUMA work in
[core_scheduling](core_scheduling.md).

## Open investigations

- [SmallBufferAllocator vs system malloc](../investigations/smallbufferallocator_vs_system_malloc.md)
- [PoolAllocator thread-local optimization](../investigations/poolallocator_thread_local.md)
