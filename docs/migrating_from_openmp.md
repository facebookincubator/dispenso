# Migrating from OpenMP to Dispenso

This guide helps you migrate parallel code from OpenMP to dispenso. Dispenso offers
several advantages over OpenMP for many use cases, including better nested parallelism,
explicit thread pool control, and sanitizer-clean code.

## Why Migrate?

| Aspect | OpenMP | Dispenso |
|--------|--------|----------|
| **Nested parallelism** | Can cause thread explosion | Work-stealing prevents oversubscription |
| **Thread pool control** | Implicit, global | Explicit, multiple pools supported |
| **Sanitizer support** | Often problematic with TSAN | Clean with ASAN/TSAN |
| **Portability** | Requires compiler support | Pure C++14, any compiler |
| **Futures** | Not available | Full futures API |
| **Task graphs** | Limited | Rich graph support with partial re-execution |

## Quick Reference

| OpenMP | Dispenso |
|--------|----------|
| `#pragma omp parallel for` | `dispenso::parallel_for()` |
| `#pragma omp parallel for reduction(+:sum)` | `dispenso::parallel_for()` with `ParForChunking` + local accumulators |
| `#pragma omp critical` | `std::mutex` or `dispenso::RWLock` |
| `#pragma omp task` | `dispenso::TaskSet::schedule()` |
| `#pragma omp taskwait` | `dispenso::TaskSet::wait()` |
| `#pragma omp single` | Execute outside parallel region |
| `omp_get_num_threads()` | `pool.numThreads()` |
| `omp_get_thread_num()` | Not directly available (usually not needed) |

## Basic Parallel For

### OpenMP
```cpp
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    process(data[i]);
}
```

### Dispenso
```cpp
#include <dispenso/parallel_for.h>

dispenso::parallel_for(0, N, [&](size_t i) {
    process(data[i]);
});
```

## Parallel For with Reduction

### OpenMP
```cpp
double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < N; ++i) {
    sum += compute(data[i]);
}
```

### Dispenso

Dispenso doesn't have built-in reduction syntax, but you can achieve the same result
with thread-local accumulators:

```cpp
#include <dispenso/parallel_for.h>

std::atomic<double> sum{0.0};
dispenso::parallel_for(0, N, [&](size_t i) {
    // For simple reductions, atomic works well
    double val = compute(data[i]);
    double expected = sum.load();
    while (!sum.compare_exchange_weak(expected, expected + val)) {}
});
```

For better performance with many updates, use chunked iteration with local accumulators:

```cpp
#include <dispenso/parallel_for.h>
#include <mutex>

double sum = 0.0;
std::mutex sumMutex;

dispenso::parallel_for(
    dispenso::makeChunkedRange(0, N, dispenso::ParForChunking::kStatic),
    [&](size_t begin, size_t end) {
        double localSum = 0.0;
        for (size_t i = begin; i < end; ++i) {
            localSum += compute(data[i]);
        }
        std::lock_guard<std::mutex> lock(sumMutex);
        sum += localSum;
    });
```

Or use the state-per-thread overload of `parallel_for`:

```cpp
#include <dispenso/parallel_for.h>

std::vector<double> partialSums;
dispenso::parallel_for(
    partialSums,
    []() { return 0.0; },  // Initialize each thread's state
    size_t{0}, static_cast<size_t>(N),
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

## Nested Parallel Loops

This is where dispenso shines. OpenMP can create exponentially many threads with
nested parallel regions, while dispenso's work-stealing handles this gracefully.

### OpenMP (Problematic)
```cpp
// WARNING: With OpenMP, this can create numThreads^2 threads!
#pragma omp parallel for
for (int i = 0; i < M; ++i) {
    #pragma omp parallel for
    for (int j = 0; j < N; ++j) {
        process(i, j);
    }
}
```

### Dispenso (Safe)
```cpp
#include <dispenso/parallel_for.h>

// Dispenso uses work-stealing - nested parallelism is safe and efficient
dispenso::parallel_for(0, M, [&](size_t i) {
    dispenso::parallel_for(0, N, [&](size_t j) {
        process(i, j);
    });
});
```

With dispenso, the total number of threads is bounded by the thread pool size,
regardless of nesting depth.

## Critical Sections

### OpenMP
```cpp
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    double val = compute(data[i]);
    #pragma omp critical
    {
        results.push_back(val);
    }
}
```

### Dispenso
```cpp
#include <dispenso/parallel_for.h>
#include <mutex>

std::mutex resultsMutex;
dispenso::parallel_for(0, N, [&](size_t i) {
    double val = compute(data[i]);
    std::lock_guard<std::mutex> lock(resultsMutex);
    results.push_back(val);
});
```

Or use `dispenso::ConcurrentVector` to avoid locking entirely:

```cpp
#include <dispenso/parallel_for.h>
#include <dispenso/concurrent_vector.h>

dispenso::ConcurrentVector<double> results;
dispenso::parallel_for(0, N, [&](size_t i) {
    double val = compute(data[i]);
    results.push_back(val);
});
```

## Task Parallelism

### OpenMP
```cpp
#pragma omp parallel
{
    #pragma omp single
    {
        #pragma omp task
        taskA();

        #pragma omp task
        taskB();

        #pragma omp taskwait
        // A and B are done

        #pragma omp task
        taskC();
    }
}
```

### Dispenso
```cpp
#include <dispenso/task_set.h>

dispenso::TaskSet tasks(dispenso::globalThreadPool());

tasks.schedule(taskA);
tasks.schedule(taskB);
tasks.wait();  // A and B are done

tasks.schedule(taskC);
// TaskSet destructor waits for C
```

## Controlling Thread Count

### OpenMP
```cpp
omp_set_num_threads(4);
// or
#pragma omp parallel for num_threads(4)
```

### Dispenso

Create a thread pool with the desired number of threads:

```cpp
#include <dispenso/thread_pool.h>
#include <dispenso/parallel_for.h>

dispenso::ThreadPool pool(4);  // 4 threads

dispenso::parallel_for(pool, 0, N, [&](size_t i) {
    process(data[i]);
});
```

Or use `ParForOptions` to limit parallelism:

```cpp
dispenso::ParForOptions options;
options.maxThreads = 4;

dispenso::parallel_for(options, 0, N, [&](size_t i) {
    process(data[i]);
});
```

## Conditional Parallelism

### OpenMP
```cpp
#pragma omp parallel for if(N > 1000)
for (int i = 0; i < N; ++i) {
    process(data[i]);
}
```

### Dispenso
```cpp
if (N > 1000) {
    dispenso::parallel_for(0, N, [&](size_t i) {
        process(data[i]);
    });
} else {
    for (size_t i = 0; i < N; ++i) {
        process(data[i]);
    }
}
```

Or use `minItemsPerChunk` to let dispenso decide:

```cpp
dispenso::ParForOptions options;
options.minItemsPerChunk = 100;  // Don't parallelize if fewer than 100 items per thread

dispenso::parallel_for(options, 0, N, [&](size_t i) {
    process(data[i]);
});
```

## Static vs Dynamic Scheduling

### OpenMP
```cpp
// Static scheduling
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; ++i) { ... }

// Dynamic scheduling
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < N; ++i) { ... }
```

### Dispenso
```cpp
#include <dispenso/parallel_for.h>

// Static-like chunking
dispenso::parallel_for(
    dispenso::makeChunkedRange(0, N, dispenso::ParForChunking::kStatic),
    [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            process(data[i]);
        }
    });

// Dynamic-like (auto) chunking - adapts to workload
dispenso::parallel_for(
    dispenso::makeChunkedRange(0, N, dispenso::ParForChunking::kAuto),
    [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            process(data[i]);
        }
    });
```

## Thread-Local Storage

### OpenMP
```cpp
#pragma omp threadprivate(myThreadLocalVar)
int myThreadLocalVar;

#pragma omp parallel
{
    myThreadLocalVar = omp_get_thread_num();
    // use myThreadLocalVar
}
```

### Dispenso

Use C++11 `thread_local` or the state-per-thread `parallel_for` overload:

```cpp
// Option 1: C++ thread_local
thread_local int myThreadLocalVar;

dispenso::parallel_for(0, N, [&](size_t i) {
    // myThreadLocalVar is thread-local
});

// Option 2: State-per-thread parallel_for
std::vector<MyState> states;
dispenso::parallel_for(
    states,
    []() { return MyState{}; },  // Initialize
    0, N,
    [&](MyState& state, size_t begin, size_t end) {
        // 'state' is unique to this thread
    });
```

## Common Pitfalls When Migrating

### 1. Lambda Captures

OpenMP uses shared variables by default. With dispenso, be explicit about captures:

```cpp
// OpenMP - x is shared by default
int x = 0;
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    // x is shared
}

// Dispenso - be explicit
int x = 0;
dispenso::parallel_for(0, N, [&x](size_t i) {  // Capture x by reference
    // x is shared
});
```

### 2. Index Types

OpenMP typically uses `int`. Dispenso uses `size_t`:

```cpp
// OpenMP
#pragma omp parallel for
for (int i = 0; i < N; ++i) { ... }

// Dispenso - use size_t
dispenso::parallel_for(size_t{0}, static_cast<size_t>(N), [&](size_t i) {
    // ...
});
```

### 3. Return Values

OpenMP parallel regions don't return values. With dispenso futures, you can:

```cpp
#include <dispenso/future.h>

auto future = dispenso::async([]() {
    return expensiveComputation();
});

// Do other work...

int result = future.get();  // Get the result
```

## Performance Tips

1. **Use chunked ranges** for better cache locality when iteration order doesn't matter

2. **Avoid over-synchronization** - dispenso's `ConcurrentVector` is often faster than
   mutex-protected `std::vector`

3. **Reuse thread pools** - creating pools is expensive; create once and reuse

4. **Consider static chunking** for uniform workloads, auto chunking for variable workloads

## Further Reading

- [Dispenso Documentation](https://facebookincubator.github.io/dispenso)
- [Getting Started Guide](getting_started.md)
- [parallel_for API Reference](https://facebookincubator.github.io/dispenso/group__parallel.html)
