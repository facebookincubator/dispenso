# PoolAllocator Thread-Local Optimization

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
