# Migrating from Intel TBB to Dispenso

This guide helps you migrate parallel code from Intel Threading Building Blocks (TBB)
to dispenso. While TBB has more features overall, dispenso offers advantages in
several areas and provides a simpler, more focused API.

## Why Migrate?

| Aspect | TBB | Dispenso |
|--------|-----|----------|
| **Sanitizer support** | Often problematic with ASAN/TSAN | Clean with all sanitizers |
| **Futures** | Not available | Full std::experimental::future-like API |
| **API complexity** | Large, complex API | Focused, simpler API |
| **Dependencies** | Heavy library | Minimal dependencies |
| **Non-Intel hardware** | May not be optimized | Platform-neutral implementation |
| **Nested parallelism** | Good | Excellent (work-stealing optimized) |

## Quick Reference

| TBB | Dispenso |
|-----|----------|
| `tbb::parallel_for` | `dispenso::parallel_for` |
| `tbb::parallel_reduce` | `dispenso::parallel_for` with state |
| `tbb::parallel_for_each` | `dispenso::for_each` |
| `tbb::task_group` | `dispenso::TaskSet` |
| `tbb::task_group::run` | `dispenso::TaskSet::schedule` |
| `tbb::task_group::wait` | `dispenso::TaskSet::wait` |
| `tbb::concurrent_vector` | `dispenso::ConcurrentVector` |
| `tbb::flow::graph` | `dispenso::Graph` |
| `tbb::global_control` | `dispenso::ThreadPool` configuration |
| `tbb::task_arena` | `dispenso::ThreadPool` (multiple pools) |

## Parallel For with Index

### TBB
```cpp
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

tbb::parallel_for(tbb::blocked_range<size_t>(0, N),
    [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
            process(data[i]);
        }
    });
```

### Dispenso
```cpp
#include <dispenso/parallel_for.h>

// Simple form - per-element
dispenso::parallel_for(0, N, [&](size_t i) {
    process(data[i]);
});

// Chunked form - like TBB's blocked_range
dispenso::parallel_for(
    dispenso::makeChunkedRange(0, N, dispenso::ParForChunking::kAuto),
    [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            process(data[i]);
        }
    });
```

## Parallel Reduce

### TBB
```cpp
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

double sum = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, N),
    0.0,
    [&](const tbb::blocked_range<size_t>& range, double init) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
            init += compute(data[i]);
        }
        return init;
    },
    std::plus<double>()
);
```

### Dispenso

Use the state-per-thread `parallel_for` overload:

```cpp
#include <dispenso/parallel_for.h>

std::vector<double> partialSums;
dispenso::parallel_for(
    partialSums,
    []() { return 0.0; },  // Initialize each thread's accumulator
    size_t{0}, N,
    [&](double& localSum, size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            localSum += compute(data[i]);
        }
    });

double sum = 0.0;
for (double partial : partialSums) {
    sum += partial;
}
```

## Parallel For Each

### TBB
```cpp
#include <tbb/parallel_for_each.h>

std::vector<Item> items;
tbb::parallel_for_each(items.begin(), items.end(), [](Item& item) {
    process(item);
});
```

### Dispenso
```cpp
#include <dispenso/for_each.h>

std::vector<Item> items;
dispenso::for_each(items.begin(), items.end(), [](Item& item) {
    process(item);
});
```

## Task Groups

### TBB
```cpp
#include <tbb/task_group.h>

tbb::task_group tg;
tg.run([]{ taskA(); });
tg.run([]{ taskB(); });
tg.wait();

tg.run([]{ taskC(); });
tg.wait();
```

### Dispenso
```cpp
#include <dispenso/task_set.h>

dispenso::TaskSet tasks(dispenso::globalThreadPool());
tasks.schedule([]{ taskA(); });
tasks.schedule([]{ taskB(); });
tasks.wait();

tasks.schedule([]{ taskC(); });
tasks.wait();
```

For recursive task parallelism (where tasks spawn more tasks), use `ConcurrentTaskSet`:

```cpp
#include <dispenso/task_set.h>

dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());

void recursiveWork(dispenso::ConcurrentTaskSet& tasks, int depth) {
    if (depth == 0) return;

    tasks.schedule([&tasks, depth]{ recursiveWork(tasks, depth - 1); });
    tasks.schedule([&tasks, depth]{ recursiveWork(tasks, depth - 1); });
}

recursiveWork(tasks, 10);
tasks.wait();
```

## Concurrent Vector

### TBB
```cpp
#include <tbb/concurrent_vector.h>

tbb::concurrent_vector<int> vec;

tbb::parallel_for(size_t(0), N, [&](size_t i) {
    vec.push_back(compute(i));
});

// Access elements
for (const auto& val : vec) {
    use(val);
}
```

### Dispenso

Dispenso's `ConcurrentVector` has a superset of TBB's API:

```cpp
#include <dispenso/concurrent_vector.h>
#include <dispenso/parallel_for.h>

dispenso::ConcurrentVector<int> vec;

dispenso::parallel_for(size_t{0}, N, [&](size_t i) {
    vec.push_back(compute(i));
});

// Access elements - same as TBB
for (const auto& val : vec) {
    use(val);
}
```

Additional dispenso features:

```cpp
// Grow by multiple elements efficiently
vec.grow_by(100);  // Add 100 default-constructed elements

// Grow with generator (avoids repeated locking)
vec.grow_by_generator(100, [i = 0]() mutable { return i++; });
```

## Task Arenas / Thread Pool Control

### TBB
```cpp
#include <tbb/task_arena.h>
#include <tbb/global_control.h>

// Limit global parallelism
tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 4);

// Or use arena for isolated execution
tbb::task_arena arena(4);  // 4 threads
arena.execute([&] {
    tbb::parallel_for(...);
});
```

### Dispenso

Create explicit thread pools:

```cpp
#include <dispenso/thread_pool.h>
#include <dispenso/parallel_for.h>

// Create a pool with 4 threads
dispenso::ThreadPool pool(4);

// Use the pool for parallel work
dispenso::parallel_for(pool, 0, N, [&](size_t i) {
    process(data[i]);
});

// Or use TaskSet with specific pool
dispenso::TaskSet tasks(pool);
tasks.schedule(work);
```

Multiple pools can coexist for different workloads:

```cpp
dispenso::ThreadPool computePool(8);   // For CPU-bound work
dispenso::ThreadPool ioPool(2);        // For I/O-bound work

dispenso::parallel_for(computePool, 0, N, compute);
dispenso::parallel_for(ioPool, 0, M, ioWork);
```

## Flow Graphs

### TBB
```cpp
#include <tbb/flow_graph.h>

tbb::flow::graph g;

tbb::flow::function_node<int, int> nodeA(g, tbb::flow::unlimited,
    [](int v) { return processA(v); });
tbb::flow::function_node<int, int> nodeB(g, tbb::flow::unlimited,
    [](int v) { return processB(v); });

tbb::flow::make_edge(nodeA, nodeB);

nodeA.try_put(input);
g.wait_for_all();
```

### Dispenso

Dispenso's `Graph` is optimized for task DAGs with potential partial re-execution:

```cpp
#include <dispenso/graph.h>

dispenso::Graph graph;

auto& nodeA = graph.addNode([]{ processA(); });
auto& nodeB = graph.addNode([]{ processB(); });
auto& nodeC = graph.addNode([]{ processC(); });

// Define dependencies
nodeB.dependsOn(nodeA);
nodeC.dependsOn(nodeA);

// Execute the graph
dispenso::execute(graph, dispenso::globalThreadPool());

// For repeated execution with partial updates:
nodeA.setIncomplete();  // Mark as needing re-execution
dispenso::execute(graph, dispenso::globalThreadPool());  // Only re-runs A and dependents
```

## Futures (Dispenso Advantage)

TBB doesn't have a futures interface. Dispenso provides one:

```cpp
#include <dispenso/future.h>

// Async computation
dispenso::Future<int> future = dispenso::async([]() {
    return expensiveComputation();
});

// Do other work while computation runs...

// Get result (blocks if not ready)
int result = future.get();

// Chaining with .then()
auto future2 = dispenso::async([]{ return 42; })
    .then([](int x) { return x * 2; })
    .then([](int x) { return std::to_string(x); });

std::string result = future2.get();  // "84"

// Combining futures
auto f1 = dispenso::async([]{ return computeA(); });
auto f2 = dispenso::async([]{ return computeB(); });

auto combined = dispenso::when_all(f1, f2).then([](auto&& tuple) {
    return std::get<0>(tuple).get() + std::get<1>(tuple).get();
});
```

## Pipelines

### TBB
```cpp
#include <tbb/pipeline.h>

tbb::parallel_pipeline(
    maxTokens,
    tbb::make_filter<void, Data>(tbb::filter::serial_in_order,
        [&](tbb::flow_control& fc) -> Data {
            if (done) { fc.stop(); return {}; }
            return readInput();
        }) &
    tbb::make_filter<Data, Data>(tbb::filter::parallel,
        [](Data d) { return process(d); }) &
    tbb::make_filter<Data, void>(tbb::filter::serial_in_order,
        [](Data d) { writeOutput(d); })
);
```

### Dispenso

Dispenso pipelines are simpler to construct:

```cpp
#include <dispenso/pipeline.h>

dispenso::pipeline(
    dispenso::globalThreadPool(),
    // Stage 1: serial input
    dispenso::stage([]() -> std::optional<Data> {
        if (done) return std::nullopt;
        return readInput();
    }, 1),  // 1 = serial
    // Stage 2: parallel processing
    dispenso::stage([](Data d) { return process(d); }, 0),  // 0 = parallel
    // Stage 3: serial output
    dispenso::stage([](Data d) { writeOutput(d); }, 1)  // 1 = serial
);
```

## Partitioners / Chunking

### TBB
```cpp
// Auto partitioner (default)
tbb::parallel_for(range, body, tbb::auto_partitioner());

// Static partitioner
tbb::parallel_for(range, body, tbb::static_partitioner());

// Affinity partitioner (cache-friendly)
tbb::affinity_partitioner ap;
tbb::parallel_for(range, body, ap);
```

### Dispenso
```cpp
#include <dispenso/parallel_for.h>

// Auto chunking (like auto_partitioner)
dispenso::parallel_for(
    dispenso::makeChunkedRange(0, N, dispenso::ParForChunking::kAuto),
    body);

// Static chunking (like static_partitioner)
dispenso::parallel_for(
    dispenso::makeChunkedRange(0, N, dispenso::ParForChunking::kStatic),
    body);
```

## Spin Mutexes

### TBB
```cpp
#include <tbb/spin_mutex.h>

tbb::spin_mutex mutex;

{
    tbb::spin_mutex::scoped_lock lock(mutex);
    // Critical section
}
```

### Dispenso
```cpp
#include <dispenso/rw_lock.h>
#include <mutex>

// For exclusive access, use std::mutex or dispenso::RWLock
std::mutex mutex;
{
    std::lock_guard<std::mutex> lock(mutex);
    // Critical section
}

// For read-heavy workloads, use RWLock
dispenso::RWLock rwLock;
{
    dispenso::RWLock::ReadGuard rlock(rwLock);
    // Read-only access
}
{
    dispenso::RWLock::WriteGuard wlock(rwLock);
    // Exclusive write access
}
```

## Common Migration Patterns

### Pattern 1: Replace blocked_range with makeChunkedRange

```cpp
// TBB
tbb::parallel_for(tbb::blocked_range<size_t>(0, N, grainSize), body);

// Dispenso
dispenso::ParForOptions options;
options.minItemsPerChunk = grainSize;
dispenso::parallel_for(options,
    dispenso::makeChunkedRange(0, N, dispenso::ParForChunking::kStatic),
    body);
```

### Pattern 2: Replace task_group recursion with ConcurrentTaskSet

```cpp
// TBB
void recursive(tbb::task_group& tg, int depth) {
    if (depth == 0) return;
    tg.run([&tg, depth]{ recursive(tg, depth-1); });
    tg.run([&tg, depth]{ recursive(tg, depth-1); });
}

// Dispenso
void recursive(dispenso::ConcurrentTaskSet& tasks, int depth) {
    if (depth == 0) return;
    tasks.schedule([&tasks, depth]{ recursive(tasks, depth-1); });
    tasks.schedule([&tasks, depth]{ recursive(tasks, depth-1); });
}
```

### Pattern 3: Combining parallel_for with futures

```cpp
// Dispenso allows mixing paradigms easily
auto future = dispenso::async([&]() {
    dispenso::parallel_for(0, N, [&](size_t i) {
        process(data[i]);
    });
    return computeResult(data);
});

// Do other work...
auto result = future.get();
```

## Performance Considerations

1. **Dispenso is faster for nested loops** - TBB's nested parallelism can have higher
   overhead; dispenso's work-stealing is optimized for this case

2. **Dispenso has lower overhead for small loops** - simpler scheduling means less
   overhead for fine-grained parallelism

3. **TBB may be faster for very large, uniform workloads** - TBB's cache-affinity
   partitioner can help in specific scenarios

4. **Use appropriate chunking** - `kStatic` for uniform work, `kAuto` for variable work

5. **Reuse pools and TaskSets** - creation has overhead; reuse when possible

## Further Reading

- [Dispenso Documentation](https://facebookincubator.github.io/dispenso)
- [Getting Started Guide](getting_started.md)
- [Migrating from OpenMP](migrating_from_openmp.md)
- [API Reference](https://facebookincubator.github.io/dispenso/modules.html)
