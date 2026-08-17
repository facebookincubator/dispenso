# Dispenso Roadmap

Planned work, organised by area. Each area keeps its own roadmap; this page is
the index plus the cross-cutting state.

## Areas

| Area | Covers |
|------|--------|
| [Core scheduling](roadmap/core_scheduling.md) | Thread pool, wake, locality, graphs, futures |
| [Parallel algorithms](roadmap/parallel_algorithms.md) | `parallel_for`, sorting, std-style algorithms |
| [Containers](roadmap/containers.md) | ConcurrentVector, ConcurrentHashMap, queues |
| [Allocators](roadmap/allocators.md) | SmallBufferAllocator, PoolAllocator, scalable allocator |
| [Language and API](roadmap/language_and_api.md) | C++20 concepts, coroutines, 2.0 breaking changes |
| [Release, packaging, outreach](roadmap/packaging_and_outreach.md) | vcpkg, Conan, Godbolt, listings, benchmark publishing |
| [fast_math](roadmap/fast_math.md) | SIMD transcendental functions |

## In Progress

| Feature | Area | Notes |
|---------|------|-------|
| ConcurrentHashMap | [Containers](roadmap/containers.md) | High-value concurrent container |
| vcpkg / Conan packages | [Packaging](roadmap/packaging_and_outreach.md) | PRs open upstream |

## Completed

[CHANGELOG.md](../../CHANGELOG.md) is the full record of shipped work. This
table lists only items that were carried on this roadmap, so their entries do
not dangle once done.

| Feature | Version | Notes |
|---------|---------|-------|
| `dispenso.h` convenience header | 1.5.0 | Includes all public headers |
| `util.h` public utilities | 1.5.0 | Exposes internal utilities |
| OpenMP migration guide | 1.4.x | docs/migrating_from_openmp.md |
| TBB migration guide | 1.4.x | docs/migrating_from_tbb.md |
| Fork-join scheduling and thread groups | 1.6.0 | Locality rings, steal rings, wake cascade, CpuSet topology. See [three_tier_scheduling.md](architecture/three_tier_scheduling.md) and [wake_cascade.md](architecture/wake_cascade.md) |

## Investigation Items

Open questions and measurements that are not yet commitments. Each is a
standalone note; results here inform roadmap items rather than being them.

| Investigation |
|---------------|
| [SmallBufferAllocator vs System Malloc Performance](investigations/smallbufferallocator_vs_system_malloc.md) |
| [PoolAllocator Thread-Local Optimization](investigations/poolallocator_thread_local.md) |
| [Nested parallel_for Benchmark Optimization Artifact](investigations/nested_parallel_for_benchmark_artifact.md) |
| [Decouple sleep mask from group concept](investigations/decouple_sleep_mask_from_groups.md) |
| [Idle-Pool Wake Latency vs CPU Cost (mostly-idle then burst)](investigations/idle_pool_wake_latency.md) |
| [Recursive fork-join vs TBB task_group (deep / heavy trees)](investigations/recursive_forkjoin_vs_tbb_task_group.md) |
| [Topology-hierarchy-aware scheduling (beyond L3-as-NUMA-proxy)](investigations/topology_hierarchy_aware_scheduling.md) |
| [Experiment: externalize per-group locality; simplify thread-pool tiers](investigations/externalize_per_group_locality.md) |
