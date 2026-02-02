# Getting Started {#getting_started}

This guide walks through the core features of dispenso with working examples.
Each section includes a complete, compilable example that you can build and run.

## Installation

See the [README](https://github.com/facebookincubator/dispenso) for installation
instructions. Dispenso requires C++14 and CMake 3.12+.

To build the examples:

```bash
mkdir build && cd build
cmake .. -DDISPENSO_BUILD_EXAMPLES=ON
make
```

## Basic Concepts

### Thread Pools

At the heart of dispenso is the `ThreadPool`. A thread pool manages a set of
worker threads that execute tasks. You can use the global thread pool or create
your own:

```cpp
#include <dispenso/thread_pool.h>

// Use the global thread pool (recommended for most cases)
dispenso::ThreadPool& pool = dispenso::globalThreadPool();

// Or create a custom pool with a specific number of threads
dispenso::ThreadPool myPool(4);  // 4 worker threads
```

### Task Sets

A `TaskSet` groups related tasks and provides a way to wait for their completion:

```cpp
#include <dispenso/task_set.h>

dispenso::TaskSet taskSet(dispenso::globalThreadPool());

taskSet.schedule([]() { /* task 1 */ });
taskSet.schedule([]() { /* task 2 */ });

taskSet.wait();  // Block until all tasks complete
```

---

## Your First Parallel Loop

The simplest way to parallelize work is with `parallel_for`. It distributes
loop iterations across available threads.

@example parallel_for_example.cpp

Key points:
- Use the simple form for independent per-element work
- Use chunked ranges when you want to control work distribution
- Per-thread state enables efficient reductions
- Options let you control parallelism and chunking strategy

---

## Parallel Iteration with for_each

When you have a container rather than an index range, use `for_each`:

@example for_each_example.cpp

Key points:
- Works with any iterator type (including non-random-access iterators)
- `for_each_n` takes an explicit count
- Pass a `TaskSet` for external synchronization control

---

## Working with Tasks

For more complex task patterns, use `TaskSet` and `ConcurrentTaskSet` directly:

@example task_set_example.cpp

Key points:
- `TaskSet` is for single-threaded scheduling
- `ConcurrentTaskSet` allows scheduling from multiple threads
- Both support cancellation for cooperative early termination
- The destructor waits for all tasks to complete

---

## Futures for Async Results

When you need return values from async operations, use `Future`:

@example future_example.cpp

Key points:
- `async()` launches work and returns a `Future`
- `then()` chains dependent computations
- `when_all()` waits for multiple futures
- `make_ready_future()` creates an already-completed future

---

## Task Graphs

For complex dependency patterns, build a task graph:

@example graph_example.cpp

Key points:
- Use `dependsOn()` to specify prerequisites
- Multiple executors available: single-thread, parallel_for, ConcurrentTaskSet
- Graphs can be re-executed after calling `setAllNodesIncomplete()`
- Subgraphs help organize large graphs

---

## Pipelines

For streaming data through stages, use pipelines:

@example pipeline_example.cpp

Key points:
- Generator stage produces values (returns `OpResult<T>` or `std::optional<T>`)
- Transform stages process values (can filter by returning empty result)
- Sink stage consumes final values
- Use `stage()` with a limit for parallel stages

---

## Thread-Safe Containers

### ConcurrentVector

A vector that supports concurrent push_back and growth:

@example concurrent_vector_example.cpp

Key points:
- Iterators and references remain stable during growth
- Use `grow_by()` for efficient batch insertion
- Reserve capacity upfront when size is known
- Not all operations are concurrent-safe (see docs)

---

## Synchronization Primitives

### Latch

A one-shot barrier for thread synchronization:

@example latch_example.cpp

Key points:
- `arrive_and_wait()` decrements and blocks
- `count_down()` decrements without blocking
- `wait()` blocks without decrementing
- Cannot be reset (one-shot)

---

## Resource Pooling

Manage expensive-to-create resources with `ResourcePool`:

@example resource_pool_example.cpp

Key points:
- Resources automatically return to pool when RAII wrapper destructs
- `acquire()` blocks if no resources available
- Good for database connections, buffers, etc.
- Can be used to limit concurrency

---

## Next Steps

- Browse the [API Reference](modules.html) for complete documentation
- Check out the [tests](https://github.com/facebookincubator/dispenso/tree/main/tests)
  for more usage examples
- See the [benchmarks](https://github.com/facebookincubator/dispenso/tree/main/benchmarks)
  for performance testing patterns
