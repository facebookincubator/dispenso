# Dispenso Coroutine Support Design Document

## Overview

This document outlines the design for C++20 coroutine integration with dispenso. The goal is to
allow coroutines to run on dispenso's thread pool infrastructure, enabling ergonomic async
programming while leveraging dispenso's work stealing and nested parallelism capabilities.

## Requirements

**C++20 Required**: This feature requires C++20 coroutine support. We will not backport
coroutine machinery to C++14/C++17.

```cpp
#if defined(__cpp_impl_coroutine) && __cpp_impl_coroutine >= 201902L
#define DISPENSO_HAS_COROUTINES 1
#include <coroutine>
#endif
```

## Motivation

Traditional async patterns with dispenso:

```cpp
// Blocking: ties up a thread at each .get()
Future<int> f1 = async(pool, work1);
int a = f1.get();  // Thread blocked waiting
Future<int> f2 = async(pool, work2);
int b = f2.get();  // Thread blocked again
```

With coroutines:

```cpp
// Non-blocking: coroutine suspends, thread is freed
Task<int> example() {
    int a = co_await async(pool, work1);  // Suspends, thread freed
    int b = co_await async(pool, work2);  // Suspends, thread freed
    co_return a + b;
}
```

Benefits:
1. **Thread efficiency** - Suspended coroutines don't block threads
2. **Composability** - Chain async operations naturally with sequential-looking code
3. **Integration** - Works with dispenso's thread pools and work stealing

## Design Goals

1. **Lazy tasks** - Coroutines don't start until awaited or explicitly spawned
2. **Inline resumption by default** - Minimize scheduling overhead for compute workloads
3. **Explicit redistribution** - Users can fan out work to the pool when parallelism is needed
4. **I/O thread isolation** - I/O completions dispatch to compute pool, not inline
5. **Work stealing compatibility** - Scheduled continuations participate in work stealing

## Core Concepts

### Task<T>

A lazy coroutine return type representing async work that produces a value of type `T`.

```cpp
namespace dispenso {

template<typename T = void>
class Task {
 public:
  using promise_type = /* implementation defined */;

  // Not copyable
  Task(const Task&) = delete;
  Task& operator=(const Task&) = delete;

  // Movable
  Task(Task&& other) noexcept;
  Task& operator=(Task&& other) noexcept;

  // Awaitable
  auto operator co_await() && -> /* awaiter */;

  // Check if task holds a valid coroutine
  explicit operator bool() const noexcept;
};

}  // namespace dispenso
```

**Lazy semantics**: Calling a coroutine function returns a `Task` immediately without
executing any code. Execution begins when the task is awaited.

```cpp
Task<int> compute() {
    std::cout << "Starting\n";  // NOT printed until awaited
    co_return 42;
}

Task<int> t = compute();  // Nothing printed yet
int x = co_await t;       // NOW "Starting" is printed
```

### schedule_on

Suspend the current coroutine and resume on a thread pool.

```cpp
namespace dispenso {

auto schedule_on(ThreadPool& pool) -> /* awaitable */;

// Convenience: use global thread pool
auto schedule_on() -> /* awaitable */;

}  // namespace dispenso
```

Usage:

```cpp
Task<void> example() {
    // Running on some thread (maybe main)
    co_await schedule_on(pool);
    // Now running on a pool thread
    do_compute_work();
}
```

This is an explicit redistribution point - useful when you want to move execution to
the pool or yield to allow other work to run.

### Awaitable Future<T>

Make `dispenso::Future<T>` directly awaitable.

```cpp
namespace dispenso {

template<typename T>
auto operator co_await(Future<T>& future) -> /* awaiter */;

template<typename T>
auto operator co_await(Future<T>&& future) -> /* awaiter */;

}  // namespace dispenso
```

Usage:

```cpp
Task<int> example() {
    Future<int> fut = async(pool, [] { return expensive_compute(); });

    // Do other work while compute runs...

    int result = co_await fut;  // Suspend until future ready
    co_return result * 2;
}
```

## Resume Policies

Where a coroutine resumes after `co_await` depends on what was awaited:

| Awaitable | Resume Policy | Rationale |
|-----------|---------------|-----------|
| `Future<T>` | Inline | Throughput; completing thread continues |
| `schedule_on(pool)` | On pool | Explicit redistribution request |
| `spawn_on(pool, task)` | On pool | Child task runs on pool |
| I/O operations | On compute pool | Keep I/O thread free |

### Inline Resumption

When a `Future` completes, the thread that completed it directly resumes the
waiting coroutine. This minimizes overhead:

```
Pool Thread P1: [runs future work] -> [resumes coroutine] -> [continues coroutine]
```

No scheduling, no context switch - maximum throughput for compute-bound chains.

### Pool Resumption

When explicit redistribution is needed, the continuation is scheduled as new work:

```
Thread X: [completes operation] -> [schedules continuation on pool]
Pool Thread P2: [picks up continuation] -> [resumes coroutine]
```

This integrates with dispenso's work stealing - if the pool is saturated, the
scheduling thread may execute the continuation inline (preventing deadlock).

## Parallelism Primitives

### spawn_on

Start a task on a thread pool, returning a handle to await later.

```cpp
namespace dispenso {

template<typename T>
Task<T> spawn_on(ThreadPool& pool, Task<T> task);

// Convenience: use global thread pool
template<typename T>
Task<T> spawn_on(Task<T> task);

}  // namespace dispenso
```

Usage:

```cpp
Task<void> parallel_work() {
    // Fan out: start tasks on pool
    Task<int> a = spawn_on(pool, compute_a());
    Task<int> b = spawn_on(pool, compute_b());

    // Both running concurrently on pool threads

    // Fan in: await results
    int result_a = co_await a;
    int result_b = co_await b;

    co_return;
}
```

### when_all

Await multiple tasks concurrently, returning when all complete.

```cpp
namespace dispenso {

template<typename... Tasks>
Task<std::tuple</* result types */>> when_all(Tasks&&... tasks);

template<typename T>
Task<std::vector<T>> when_all(std::vector<Task<T>> tasks);

}  // namespace dispenso
```

Usage:

```cpp
Task<void> process_all(std::vector<Item>& items) {
    std::vector<Task<Result>> tasks;
    for (auto& item : items) {
        tasks.push_back(spawn_on(pool, process(item)));
    }

    std::vector<Result> results = co_await when_all(std::move(tasks));
}
```

### when_any (Future)

Await multiple tasks, returning when the first completes. Lower priority than `when_all`.

## Blocking Wait

For integration with non-coroutine code, provide a way to block until a task completes.

```cpp
namespace dispenso {

template<typename T>
T sync_wait(Task<T> task);

template<typename T>
T sync_wait(Task<T> task, ThreadPool& pool);

}  // namespace dispenso
```

Usage:

```cpp
int main() {
    Task<int> t = async_computation();
    int result = sync_wait(std::move(t));  // Block main thread until done
    std::cout << result << "\n";
}
```

## I/O Integration (Future Phase)

For high-performance async I/O, we would add an `IoContext` that manages platform-native
async I/O and dispatches completions to the compute pool.

### Architecture

```
                    ┌─────────────────┐
                    │  Compute Pool   │
                    │  (N threads)    │
                    └────────▲────────┘
                             │ schedule resumption
                    ┌────────┴────────┐
                    │   I/O Thread    │
                    │  (1-2 threads)  │
                    └────────▲────────┘
                             │ completion events
                    ┌────────┴────────┐
                    │     Kernel      │
                    │ io_uring/kqueue │
                    │     /IOCP       │
                    └─────────────────┘
```

The I/O thread:
1. Submits I/O requests to the kernel
2. Waits for completions (blocking on io_uring/kqueue/IOCP)
3. For each completion, schedules the coroutine resumption on the compute pool
4. Never runs compute work itself

### Platform Backends

| Platform | Backend | Notes |
|----------|---------|-------|
| Linux | io_uring | Kernel 5.1+, best performance |
| Linux (fallback) | epoll | Older kernels |
| macOS | kqueue | Native async events |
| Windows | IOCP | Overlapped I/O |

### API Sketch

```cpp
namespace dispenso::io {

class IoContext {
 public:
  explicit IoContext(ThreadPool& compute_pool);

  // Start the I/O thread
  void run();

  // Stop the I/O thread
  void stop();
};

// File operations
Task<std::vector<std::byte>> read_file(IoContext& ctx, const fs::path& path);
Task<void> write_file(IoContext& ctx, const fs::path& path, std::span<const std::byte> data);

// Potentially: sockets, timers, etc.

}  // namespace dispenso::io
```

## Error Handling

Exceptions thrown in coroutines are captured and rethrown when the task is awaited:

```cpp
Task<int> failing_task() {
    co_await schedule_on(pool);
    throw std::runtime_error("oops");
    co_return 42;  // Never reached
}

Task<void> caller() {
    try {
        int x = co_await failing_task();
    } catch (const std::runtime_error& e) {
        // Exception caught here
    }
}
```

## Cancellation (Future Consideration)

Potential integration with `std::stop_token` for cooperative cancellation:

```cpp
Task<int> cancellable_work(std::stop_token token) {
    for (int i = 0; i < 1000; ++i) {
        if (token.stop_requested()) {
            co_return -1;  // Early exit
        }
        co_await process_chunk(i);
    }
    co_return 0;
}
```

This is deferred to a future phase.

## Performance Considerations

### Coroutine Frame Allocation

Coroutine frames are typically heap-allocated. For performance-critical code:

1. **HALO (Heap Allocation eLision Optimization)** - Compilers can elide allocation
   when the coroutine lifetime is bounded by the caller
2. **Custom allocators** - `promise_type` can define `operator new` for custom allocation
3. **Frame pooling** - Reuse coroutine frames for frequently-created coroutines

### Symmetric Transfer

C++20 supports symmetric transfer via `await_suspend` returning a `coroutine_handle<>`.
This allows direct transfer between coroutines without growing the call stack:

```cpp
auto await_suspend(std::coroutine_handle<> h) -> std::coroutine_handle<> {
    // Instead of: h.resume(); return;
    // Return the handle to transfer to:
    return continuation_;  // Direct transfer, no stack growth
}
```

We should use symmetric transfer where applicable to prevent stack overflow in
deep coroutine chains.

### Comparison with std::execution::par

| Aspect | Coroutines | parallel_for |
|--------|------------|--------------|
| Overhead per operation | ~50-100ns (allocation) | ~10-20ns |
| Best for | I/O, async chains, latency | Compute, throughput |
| Thread blocking | Never | Never |
| Code style | Sequential-looking async | Explicit parallelism |

Use coroutines for async workflows; use `parallel_for` for data parallelism.

## Header Organization

```
dispenso/
├── coroutine.h              // Main header, includes all coroutine support
├── coroutine/
│   ├── task.h               // Task<T> definition
│   ├── schedule.h           // schedule_on, spawn_on
│   ├── sync_wait.h          // sync_wait for blocking
│   ├── when_all.h           // when_all combinator
│   └── future_awaitable.h   // operator co_await for Future<T>
└── io/                      // Future phase
    ├── io_context.h         // IoContext
    └── file.h               // File operations
```

## Implementation Phases

### Phase 1: Core Primitives

1. `Task<T>` with lazy semantics
2. `schedule_on(ThreadPool&)`
3. `sync_wait(Task<T>)`
4. Awaitable `Future<T>`

### Phase 2: Parallelism Combinators

1. `spawn_on(ThreadPool&, Task<T>)`
2. `when_all(Tasks...)`
3. `when_all(vector<Task<T>>)`

### Phase 3: I/O (Optional/Future)

1. `IoContext` with platform backends
2. File read/write operations
3. Potential: sockets, timers

## Testing Strategy

Each component should have tests for:

1. **Basic functionality** - Happy path execution
2. **Exception propagation** - Errors caught at await point
3. **Cancellation** - Coroutine destroyed before completion
4. **Thread safety** - Concurrent await, spawn, etc.
5. **Work stealing interaction** - Behavior under pool saturation
6. **Nested coroutines** - Coroutines spawning coroutines
7. **Memory** - No leaks, proper frame destruction

## Open Questions

1. **Naming**: `spawn_on` vs `start_on` vs `dispatch`?

2. **Default pool**: Should `schedule_on()` (no args) use a global pool, or require explicit pool?
   Recommendation: Require explicit pool for clarity, but could add a thread-local "current pool" concept.

3. **Task move semantics**: Should awaiting a Task consume it (require `co_await std::move(t)`)?
   Recommendation: Yes, prevents accidental double-await.

4. **Integration with dispenso::Future**: Should `Task<T>` be convertible to `Future<T>`?
   This would allow mixing coroutine and callback-based code.

5. **Allocator support**: Should `Task` support custom allocators for frame allocation?
   Recommendation: Defer to later phase; standard allocation is usually fine.

## References

- [C++20 Coroutines](https://en.cppreference.com/w/cpp/language/coroutines)
- [Lewis Baker's Coroutine Theory](https://lewissbaker.github.io/)
- [cppcoro Library](https://github.com/lewissbaker/cppcoro)
- [libcoro Library](https://github.com/jbaldwin/libcoro)
- [concurrencpp Library](https://github.com/David-Haim/concurrencpp)
- [libfork Library](https://github.com/ConorWilliams/libfork)
- [libunifex](https://github.com/facebookexperimental/libunifex)
- [io_uring](https://kernel.dk/io_uring.pdf)

## Existing Libraries Analysis

We surveyed existing C++ coroutine libraries to inform our design decisions.

### Library Comparison

| Library | C++ Version | Task Pattern | Thread Pool | Work Stealing | I/O Support | License |
|---------|-------------|--------------|-------------|---------------|-------------|---------|
| cppcoro | C++17 (experimental) | Lazy, symmetric transfer | static_thread_pool | Yes | io_service | MIT |
| libcoro | C++20 | Lazy, symmetric transfer | thread_pool | No (FIFO) | io_scheduler | MIT |
| concurrencpp | C++20 | Executor-based | Multiple types | No | No | MIT |
| libfork | C++20 | Fork-join focused | busy_pool, lazy_pool | Yes (NUMA-aware) | No | MPL-2.0 |

### Key Patterns to Adopt

**From cppcoro (Lewis Baker - designer of C++ coroutines):**

1. **Lazy Task semantics** - `initial_suspend()` returns `suspend_always`
2. **Symmetric transfer** - `await_suspend` returns `coroutine_handle<>` for stack-safe chaining
3. **final_awaitable pattern** - Continuation chaining without scheduler overhead

```cpp
// Pattern from cppcoro - final_awaitable for inline continuation
struct final_awaitable {
    bool await_ready() const noexcept { return false; }

    std::coroutine_handle<> await_suspend(std::coroutine_handle<promise_type> h) noexcept {
        // Symmetric transfer to continuation - no stack growth
        return h.promise().m_continuation;
    }

    void await_resume() noexcept {}
};
```

**From libcoro (modern C++20 reference):**

1. **Clean C++20 implementation** - Uses `<coroutine>` header (not experimental)
2. **std::variant for result storage** - Type-safe value/exception storage
3. **schedule_operation pattern** - Awaitable type for thread pool scheduling

### Why Not Depend on an Existing Library

1. **dispenso already has a superior ThreadPool** - Our work-stealing implementation is
   battle-tested and optimized. Adding cppcoro/libcoro would duplicate or conflict.

2. **Thin integration layer** - The core Task machinery is ~300-400 lines. We can adopt
   the patterns without taking a dependency.

3. **Control over resume policy** - We need fine-grained control over inline vs pool
   resumption to integrate with dispenso's work stealing.

4. **Licensing simplicity** - Keeping implementation in-house avoids license management.

### Implementation Approach

1. **Task<T>**: Adapt the cppcoro/libcoro pattern (lazy, symmetric transfer, final_awaitable)
2. **schedule_on**: Create awaitable that schedules continuation on dispenso::ThreadPool
3. **Awaitable Future<T>**: Wrap dispenso::Future with coroutine await interface
4. **when_all**: Reference libcoro's implementation for clean variadic/vector patterns

## Testing Strategy

Each component should have tests for:

1. **Basic functionality** - Happy path execution
2. **Exception propagation** - Errors caught at await point
3. **Cancellation** - Coroutine destroyed before completion
4. **Thread safety** - Concurrent await, spawn, etc.
5. **Work stealing interaction** - Behavior under pool saturation
6. **Nested coroutines** - Coroutines spawning coroutines
7. **Memory** - No leaks, proper frame destruction

### Test Categories

| Component | Test Coverage |
|-----------|---------------|
| `Task<T>` | Construction, move, await, exception, void specialization, reference types |
| `Task<void>` | Return void, exception propagation |
| `schedule_on` | Switch threads, work stealing under load, nested scheduling |
| `sync_wait` | Block until complete, exception propagation, timeout |
| `spawn_on` | Concurrent execution, result collection |
| `when_all` | Variadic, vector, mixed types, partial failure |
| `Future<T>` awaitable | Already-ready, pending, exception |

## Benchmarking Strategy

We will benchmark against existing coroutine libraries to validate our implementation
achieves competitive or superior performance, especially for dispenso's target use cases.

### Benchmark Libraries

- **cppcoro** - Baseline reference implementation
- **libcoro** - Modern C++20 comparison
- **libfork** - Work-stealing coroutine comparison
- **folly::coro** (if available) - Production-grade comparison

### Benchmark Scenarios

| Benchmark | Description | Metrics |
|-----------|-------------|---------|
| **Task creation/destruction** | Create and destroy N tasks without execution | ops/sec, memory |
| **Sequential await chain** | Chain of N co_await operations | latency, throughput |
| **Fan-out/fan-in** | spawn N tasks, when_all to join | throughput, scalability |
| **Nested parallelism** | Recursive task spawning (tree structure) | throughput, work stealing efficiency |
| **Mixed workload** | Combination of short/long tasks | fairness, tail latency |
| **sync_wait overhead** | Cost of blocking wait vs raw Future::get() | latency comparison |
| **schedule_on overhead** | Cost of thread switch vs direct pool.schedule() | latency comparison |
| **Contention** | Many threads spawning tasks concurrently | throughput under contention |

### Key Metrics

1. **Throughput** - Tasks completed per second
2. **Latency** - Time from spawn to completion (p50, p99, p999)
3. **Scalability** - Throughput vs thread count
4. **Memory overhead** - Per-task allocation overhead
5. **Context switch cost** - schedule_on latency

### Benchmark Implementation

```cpp
// Example: Fan-out/fan-in benchmark
void BM_FanOutFanIn(benchmark::State& state) {
    const int num_tasks = state.range(0);
    dispenso::ThreadPool pool(std::thread::hardware_concurrency());

    for (auto _ : state) {
        auto result = sync_wait(fan_out_fan_in(pool, num_tasks));
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * num_tasks);
}
BENCHMARK(BM_FanOutFanIn)->Range(8, 8192);
```

### Comparison Methodology

1. **Equivalent workloads** - Same task logic across libraries
2. **Same hardware** - All benchmarks on identical machines
3. **Warm-up** - Discard initial iterations for JIT/cache effects
4. **Statistical rigor** - Multiple runs, report mean/stddev
5. **Profile-guided analysis** - Use perf/vtune to understand bottlenecks

## Implementation Checklist

Primitives in implementation order. Check off as completed.

### Phase 1: Core Primitives
- [ ] `Task<T>` (lazy coroutine type)
- [ ] `Task<void>` specialization
- [ ] `schedule_on(ThreadPool&)`
- [ ] `sync_wait(Task<T>)`
- [ ] Awaitable `Future<T>`

### Phase 2: Parallelism Combinators
- [ ] `spawn_on(ThreadPool&, Task<T>)`
- [ ] `when_all(Tasks...)` (variadic)
- [ ] `when_all(vector<Task<T>>)`
- [ ] `when_any` (optional)

### Phase 3: I/O (Future)
- [ ] `IoContext` base infrastructure
- [ ] Linux io_uring backend
- [ ] Linux epoll fallback
- [ ] macOS kqueue backend
- [ ] Windows IOCP backend
- [ ] `read_file` / `write_file`

### Testing & Benchmarks
- [ ] Unit tests for all Phase 1 primitives
- [ ] Unit tests for all Phase 2 combinators
- [ ] Benchmark: Task creation/destruction
- [ ] Benchmark: Sequential await chain
- [ ] Benchmark: Fan-out/fan-in
- [ ] Benchmark: Nested parallelism
- [ ] Benchmark: Comparison vs cppcoro
- [ ] Benchmark: Comparison vs libcoro
- [ ] Benchmark: Comparison vs libfork
