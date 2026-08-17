# SmallBufferAllocator vs System Malloc Performance

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
