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

> **Note:** `globalThreadPool()` defaults to `std::thread::hardware_concurrency() - 1`
> worker threads, since the calling thread typically participates in computation.
> Use `dispenso::resizeGlobalThreadPool(n)` to change it.

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

<!-- @example parallel_for_example.cpp -->

**Simple per-element parallel loop:**

```cpp
#include <dispenso/parallel_for.h>

// Process each element independently in parallel
dispenso::parallel_for(0, kArraySize, [&](size_t i) { output[i] = std::sqrt(input[i]); });
```

**Reduction with per-thread state:**

```cpp
std::vector<double> partialSums;
dispenso::parallel_for(
    partialSums,
    []() { return 0.0; }, // State initializer
    size_t{0},
    kArraySize,
    [&](double& localSum, size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        localSum += input[i];
      }
    });

// Combine partial sums
double totalSum = 0.0;
for (double partial : partialSums) {
  totalSum += partial;
}
```

See [full example](../examples/parallel_for_example.cpp).

Key points:
- Use the simple form for independent per-element work
- Use chunked ranges when you want to control work distribution
- Per-thread state enables efficient reductions
- Options let you control parallelism and chunking strategy

---

## Parallel Iteration with for_each

When you have a container rather than an index range, use `for_each`:

<!-- @example for_each_example.cpp -->

**Parallel for_each on a vector:**

```cpp
#include <dispenso/for_each.h>

std::vector<double> values = {1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0};

// Apply square root to each element in parallel
dispenso::for_each(values.begin(), values.end(), [](double& val) { val = std::sqrt(val); });
```

**for_each_n with explicit count:**

```cpp
std::vector<int> partial = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

// Only process first 5 elements
dispenso::for_each_n(partial.begin(), 5, [](int& n) { n += 100; });
```

See [full example](../examples/for_each_example.cpp).

Key points:
- Works with any iterator type (including non-random-access iterators)
- `for_each_n` takes an explicit count
- Pass a `TaskSet` for external synchronization control

---

## Working with Tasks

For more complex task patterns, use `TaskSet` and `ConcurrentTaskSet` directly:

<!-- @example task_set_example.cpp -->

**Basic TaskSet:**

```cpp
#include <dispenso/task_set.h>

dispenso::TaskSet taskSet(dispenso::globalThreadPool());
std::atomic<int> counter(0);

for (int i = 0; i < 10; ++i) {
  taskSet.schedule([&counter, i]() { counter.fetch_add(i, std::memory_order_relaxed); });
}

taskSet.wait();
```

**ConcurrentTaskSet with nested scheduling:**

```cpp
dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());
std::atomic<int> total(0);

for (int i = 0; i < 5; ++i) {
  taskSet.schedule([&taskSet, &total, i]() {
    // Each task schedules two sub-tasks
    for (int j = 0; j < 2; ++j) {
      taskSet.schedule(
          [&total, i, j]() { total.fetch_add(i * 10 + j, std::memory_order_relaxed); });
    }
  });
}

taskSet.wait();
```

See [full example](../examples/task_set_example.cpp).

Key points:
- `TaskSet` is for single-threaded scheduling
- `ConcurrentTaskSet` allows scheduling from multiple threads
- Both support cancellation for cooperative early termination
- The destructor waits for all tasks to complete

---

## Futures for Async Results

When you need return values from async operations, use `Future`:

<!-- @example future_example.cpp -->

**Basic async and get:**

```cpp
#include <dispenso/future.h>

dispenso::Future<int> future = dispenso::async([]() {
  int result = 0;
  for (int i = 1; i <= 100; ++i) {
    result += i;
  }
  return result;
});

int result = future.get();  // blocks until ready
```

**Chaining with then():**

```cpp
dispenso::Future<double> chainedFuture = dispenso::async([]() {
                                           return 16.0;
                                         })
                                             .then([](dispenso::Future<double>&& prev) {
                                               return std::sqrt(prev.get());
                                             })
                                             .then([](dispenso::Future<double>&& prev) {
                                               return prev.get() * 2.0;
                                             });
```

**when_all for multiple futures:**

```cpp
dispenso::Future<int> f1 = dispenso::async([]() { return 10; });
dispenso::Future<int> f2 = dispenso::async([]() { return 20; });
dispenso::Future<int> f3 = dispenso::async([]() { return 30; });

auto allFutures = dispenso::when_all(std::move(f1), std::move(f2), std::move(f3));
auto tuple = allFutures.get();
int sum = std::get<0>(tuple).get() + std::get<1>(tuple).get() + std::get<2>(tuple).get();
```

See [full example](../examples/future_example.cpp).

Key points:
- `async()` launches work and returns a `Future`
- `then()` chains dependent computations
- `when_all()` waits for multiple futures
- `make_ready_future()` creates an already-completed future

---

## Task Graphs

For complex dependency patterns, build a task graph:

<!-- @example graph_example.cpp -->

**Diamond dependency pattern:**

```cpp
#include <dispenso/graph.h>
#include <dispenso/graph_executor.h>

//         A
//        / \
//       B   C
//        \ /
//         D
dispenso::Graph graph;

dispenso::Node& A = graph.addNode([&]() { r[0] = 1.0f; });
dispenso::Node& B = graph.addNode([&]() { r[1] = r[0] * 2.0f; });
dispenso::Node& C = graph.addNode([&]() { r[2] = r[0] + 5.0f; });
dispenso::Node& D = graph.addNode([&]() { r[3] = r[1] + r[2]; });

B.dependsOn(A);
C.dependsOn(A);
D.dependsOn(B, C);

setAllNodesIncomplete(graph);
dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());
dispenso::ConcurrentTaskSetExecutor executor;
executor(taskSet, graph);
```

See [full example](../examples/graph_example.cpp).

Key points:
- Use `dependsOn()` to specify prerequisites
- Multiple executors available: single-thread, parallel_for, ConcurrentTaskSet
- Graphs can be re-executed after calling `setAllNodesIncomplete()`
- Subgraphs help organize large graphs

---

## Pipelines

For streaming data through stages, use pipelines:

<!-- @example pipeline_example.cpp -->

**3-stage pipeline (generator -> transform -> sink):**

```cpp
#include <dispenso/pipeline.h>

std::vector<int> results;
int counter = 0;

dispenso::pipeline(
    // Stage 1: Generator - produces values
    [&counter]() -> dispenso::OpResult<int> {
      if (counter >= 10) {
        return {}; // Empty result signals end of input
      }
      return counter++;
    },
    // Stage 2: Transform - squares the value
    [](int value) { return value * value; },
    // Stage 3: Sink - collects results
    [&results](int value) { results.push_back(value); });
```

See [full example](../examples/pipeline_example.cpp).

Key points:
- Generator stage produces values (returns `OpResult<T>` or `std::optional<T>`)
- Transform stages process values (can filter by returning empty result)
- Sink stage consumes final values
- Use `stage()` with a limit for parallel stages

---

## Thread-Safe Containers

### ConcurrentVector

A vector that supports concurrent push_back and growth:

<!-- @example concurrent_vector_example.cpp -->

**Concurrent push_back from multiple threads:**

```cpp
#include <dispenso/concurrent_vector.h>
#include <dispenso/parallel_for.h>

dispenso::ConcurrentVector<int> vec;

dispenso::parallel_for(0, 1000, [&vec](size_t i) { vec.push_back(static_cast<int>(i)); });
```

**Iterator stability during concurrent modification:**

```cpp
dispenso::ConcurrentVector<int> vec;
vec.push_back(1);
vec.push_back(2);
vec.push_back(3);

auto it = vec.begin();
int& firstElement = *it;

// Push more elements concurrently
dispenso::parallel_for(0, 100, [&vec](size_t i) { vec.push_back(static_cast<int>(i + 100)); });

// Original iterator and reference are still valid
assert(*it == 1);
assert(firstElement == 1);
```

See [full example](../examples/concurrent_vector_example.cpp).

Key points:
- Iterators and references remain stable during growth
- Use `grow_by()` for efficient batch insertion
- Reserve capacity upfront when size is known
- Not all operations are concurrent-safe (see docs)

---

## Synchronization Primitives

### Latch

A one-shot barrier for thread synchronization:

<!-- @example latch_example.cpp -->

**count_down + wait pattern:**

```cpp
#include <dispenso/latch.h>

constexpr int kNumWorkers = 3;
dispenso::Latch workComplete(kNumWorkers);
std::vector<int> results(kNumWorkers, 0);

dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());

for (int i = 0; i < kNumWorkers; ++i) {
  taskSet.schedule([&workComplete, &results, i]() {
    results[static_cast<size_t>(i)] = (i + 1) * 10;
    workComplete.count_down();  // Signal work is done (non-blocking)
  });
}

workComplete.wait();  // Main thread waits for all workers
```

See [full example](../examples/latch_example.cpp).

Key points:
- `arrive_and_wait()` decrements and blocks
- `count_down()` decrements without blocking
- `wait()` blocks without decrementing
- Cannot be reset (one-shot)

---

## Resource Pooling

Manage expensive-to-create resources with `ResourcePool`:

<!-- @example resource_pool_example.cpp -->

**Basic buffer pool with RAII:**

```cpp
#include <dispenso/resource_pool.h>

// Create a pool of 4 buffers
dispenso::ResourcePool<Buffer> bufferPool(4, []() { return Buffer(); });

dispenso::parallel_for(0, 100, [&bufferPool](size_t i) {
  // Acquire a resource from the pool (blocks if none available)
  auto resource = bufferPool.acquire();

  // Use the resource
  resource.get().process(static_cast<int>(i));

  // Resource automatically returned to pool when 'resource' goes out of scope
});
```

See [full example](../examples/resource_pool_example.cpp).

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
