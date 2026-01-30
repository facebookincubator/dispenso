# Dispenso {#mainpage}

**A high-performance C++14 library for task parallelism**

Dispenso provides mechanisms for thread pools, task sets, parallel for loops,
futures, pipelines, task graphs, and more.

## Quick Start

```cpp
#include <dispenso/parallel_for.h>

dispenso::parallel_for(0, 1000, [](size_t i) {
    // Process item i in parallel
});
```

## API Modules

- @ref group_core "Core Components" - Thread pools and task sets
- @ref group_parallel "Parallel Loops" - parallel_for and for_each
- @ref group_async "Async & Futures" - Futures and async operations
- @ref group_graph "Graphs & Pipelines" - Task graphs and pipelines
- @ref group_containers "Concurrent Containers" - Thread-safe data structures
- @ref group_sync "Synchronization" - Locks, latches, and events
- @ref group_alloc "Allocators" - Memory allocation utilities
- @ref group_util "Utilities" - Platform, timing, and helpers

## Resources

- [GitHub Repository](https://github.com/facebookincubator/dispenso)
