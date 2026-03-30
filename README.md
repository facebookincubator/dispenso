[![Build and test](https://github.com/facebookincubator/dispenso/actions/workflows/build.yml/badge.svg)](https://github.com/facebookincubator/dispenso/actions/workflows/build.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://facebookincubator.github.io/dispenso)
[![codecov](https://codecov.io/gh/facebookincubator/dispenso/branch/main/graph/badge.svg)](https://codecov.io/gh/facebookincubator/dispenso)
[![Conan Center](https://img.shields.io/conan/v/dispenso)](https://conan.io/center/recipes/dispenso)
[![vcpkg](https://img.shields.io/vcpkg/v/dispenso)](https://vcpkg.io/en/package/dispenso)
[![Homebrew](https://img.shields.io/homebrew/v/dispenso)](https://formulae.brew.sh/formula/dispenso)
[![MacPorts](https://img.shields.io/badge/macports-dispenso-blue)](https://ports.macports.org/port/dispenso/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/dispenso/badges/version.svg)](https://anaconda.org/conda-forge/dispenso)

# Dispenso

**A high-performance C++ thread pool and parallel algorithms library**

Dispenso is a modern **C++ parallel computing library** that provides work-stealing thread pools, parallel for loops, futures, task graphs, and concurrent containers. It serves as a powerful **alternative to OpenMP and Intel TBB**, offering better nested parallelism, sanitizer-clean code, and explicit thread pool control. Dispenso is used in hundreds of projects at Meta (formerly Facebook) and has been heavily tested and iterated on in production.

**Key advantages over OpenMP and TBB:**
- **No thread explosion** with nested parallel loops - dispenso's work-stealing prevents deadlocks and oversubscription
- **Clean with ASAN/TSAN** - fully sanitizer-compatible, unlike many TBB versions
- **Thread-safe shared futures** - `std::experimental::shared_future`-like API that TBB lacks, safe for multiple concurrent waiters, with much better performance than `std::future`
- **Portable** - C++14 compatible with no compiler-specific pragmas or extensions; C++20 builds gain concept constraints for clearer error messages

## Table of Contents

- [Choose Dispenso If...](#choosedispenso)
- [Features](#features)
- [Quick Start](#quickstart)
- [Comparison vs Other Libraries](#comparison)
- [Migration Guides](#migrationguides)
- [When Not to Use Dispenso](#nottouse)
- [Documentation and Examples](#examples)
- [Benchmark Results](#benchresults)
- [Installing](#installing)
- [Building](#building)
- [Known Issues](#knownissues)
- [License](#license)

<div id='choosedispenso'/>

## Choose Dispenso If...

- You need **nested parallelism** without thread explosion
- You want **sanitizer-clean** (ASAN/TSAN) concurrent code
- You want **explicit control over thread pools** rather than implicit global state
- You need **compute-bound futures**, not I/O-bound async
- You want **stable APIs** and minimal dependencies
- You need **cross-platform portability** from a C++14 baseline
- You have **multiple independent parallel loops** that can overlap (cascading `parallel_for`)

<div id='features'/>

## Features

Dispenso provides a comprehensive set of parallel programming primitives:

**Core runtime:**
* **[`ThreadPool`](https://facebookincubator.github.io/dispenso/classdispenso_1_1_thread_pool.html)** — work-stealing thread pool backing all dispenso parallelism
* **[`TaskSet`](https://facebookincubator.github.io/dispenso/classdispenso_1_1_task_set.html) / [`ConcurrentTaskSet`](https://facebookincubator.github.io/dispenso/classdispenso_1_1_concurrent_task_set.html)** — task grouping with wait, cancellation, and recursive scheduling

**Parallel algorithms:**
* **[`parallel_for`](docs/getting_started.md#your-first-parallel-loop)** — parallel loops over indices, blocking or non-blocking (cascaded); cascading `parallel_for` enables overlapping independent loops without oversubscription
* **[`for_each`](docs/getting_started.md#parallel-iteration-with-for_each)** — parallel `std::for_each` / `std::for_each_n`
* **[`Future`](docs/getting_started.md#futures-for-async-results)** — high-performance thread-safe shared futures with `then()`, `when_all()`, and an API matching `std::experimental::shared_future`
* **[`Graph`](docs/getting_started.md#task-graphs)** — task graph execution with subgraph support and incremental re-evaluation
* **[`pipeline`](docs/getting_started.md#pipelines)** — parallel pipelining of streaming workloads

**Concurrent containers and synchronization:**
* **[`ConcurrentVector`](docs/getting_started.md#concurrentvector)** — concurrent growable vector, superset of TBB `concurrent_vector` API
* **[`Latch`](docs/getting_started.md#latch)** — one-shot barrier for thread synchronization
* **[`RWLock`](https://facebookincubator.github.io/dispenso/classdispenso_1_1_r_w_lock.html)** — reader-writer spin lock, outperforms `std::shared_mutex` under low write contention
* **`SPSCRingBuffer`** — lock-free single-producer single-consumer ring buffer *(1.5.0)*

**General-purpose utilities:**
* **`SmallVector`** — inline-storage vector (not thread-aware; similar to `folly::small_vector`) *(1.5.0)*
* **`OnceFunction`** — lightweight move-only `void()` callable
* **`PoolAllocator`** — pool allocator with pluggable backing allocation (e.g. CUDA)
* **`SmallBufferAllocator`** — fast concurrent allocation for temporary objects
* **[`ResourcePool`](docs/getting_started.md#resource-pooling)** — semaphore-like guard around pooled resources
* **`CompletionEvent`** — notifiable event with wait and timed wait
* **`AsyncRequest`** — lightweight constrained message passing
* **`ConcurrentObjectArena`** — fast same-type object arena

<div id='quickstart'/>

## Quick Start

**Parallel for loop** - the most common use case:

```cpp
#include <dispenso/parallel_for.h>

// Sequential
for (size_t i = 0; i < N; ++i) {
    process(data[i]);
}

// Parallel with dispenso - just wrap it!
dispenso::parallel_for(0, N, [&](size_t i) {
    process(data[i]);
});
```

**Install via your favorite package manager:**

```bash
# Conda
conda install -c conda-forge dispenso

# Fedora/RHEL
sudo dnf install dispenso-devel

# Or build from source (see below)
```

<div id='comparison'/>

## Comparison vs Other Libraries

### TBB (Intel Threading Building Blocks)

TBB has more functionality overall, but we built dispenso for three reasons:
1. **Sanitizer compatibility** — TBB doesn't work well with ASAN/TSAN
2. **Thread-safe shared futures** — TBB lacks a futures interface; dispenso provides `std::experimental::shared_future`-like futures safe for multiple concurrent waiters
3. **Non-Intel hardware** — we needed to control performance on diverse platforms

**Performance:** Dispenso tends to be faster for small and medium parallel loops, and on par for large ones. When many loops run independently, dispenso's cascading `parallel_for` avoids oversubscription and has delivered **32-50% speedups in production workloads** after porting from TBB at Meta. TBB lacks an equivalent mechanism.

See [Migrating from TBB](docs/migrating_from_tbb.md) for a step-by-step porting guide.

### OpenMP

OpenMP has simple syntax for basic loops but grows complex for advanced constructs. Nested `#pragma omp parallel for` inside threaded code risks thread explosion and machine exhaustion. Dispenso outperforms OpenMP for medium and large loops. OpenMP has an advantage for very small loops due to direct compiler support, though dispenso's `minItemsPerChunk` option can close this gap by tuning the parallelism threshold for small/fast loops.

See [Migrating from OpenMP](docs/migrating_from_openmp.md) for a step-by-step porting guide.

### Folly

Folly excels at asynchronous I/O with coroutine support. Dispenso is designed for **compute-bound** work. Dispenso's futures are lighter-weight and faster for compute workloads; Folly is the better choice for I/O-heavy applications.

### TaskFlow

TaskFlow focuses on task graph execution. Dispenso has faster graph construction, faster full and partial graph execution, much lower `parallel_for` overhead (10-100x in benchmarks), and simpler/faster pipeline construction. TaskFlow does offer CUDA graph mappings, which dispenso does not currently provide.

### Others (GCD, C++ std parallelism)

GCD is Apple-specific with ports to other platforms. C++ parallel algorithms are still evolving — we are interested in enabling dispenso as a backend for `std::execution` and C++ coroutines. Contributions and benchmarks are welcome.

<div id='migrationguides'/>

### Migration Guides

- **[Migrating from TBB](docs/migrating_from_tbb.md)** — API mappings, thread pool differences, and common porting patterns
- **[Migrating from OpenMP](docs/migrating_from_openmp.md)** — Replacing `#pragma omp` with dispenso equivalents, handling reductions and nested parallelism

<div id='nottouse'/>

## When Not to Use Dispenso
Dispenso isn't really designed for high-latency task offload, it works best for compute-bound tasks.  Using the thread pool for networking, disk, or in cases with frequent TLB misses (really any scenario with kernel context switches) may result in less than ideal performance.

In these kernel context switch scenarios, `dispenso::Future` can be used with `dispenso::NewThreadInvoker`, which should be roughly equivalent with std::future performance.

If you need async I/O, Folly is likely a good choice (though it still doesn't fix e.g. TLB misses).

<div id='examples'/>

## Documentation and Examples
[Documentation can be found here](https://facebookincubator.github.io/dispenso)

Here are some simple examples of what you can do in dispenso. See tests and benchmarks for more examples.

### parallel\_for

A simple sequential loop can be parallelized with minimal changes:

```cpp
for(size_t j = 0; j < kLoops; ++j) {
  vec[j] = someFunction(j);
}
```

Becomes:

```cpp
dispenso::parallel_for(0, kLoops, [&vec] (size_t j) {
  vec[j] = someFunction(j);
});
```

### TaskSet

Schedule multiple tasks and wait for them to complete:

```cpp
void randomWorkConcurrently() {
  dispenso::TaskSet tasks(dispenso::globalThreadPool());
  tasks.schedule([&stateA]() { stateA = doA(); });
  tasks.schedule([]() { doB(); });
  // Do some work on current thread
  tasks.wait(); // After this, A, B done.
  tasks.schedule(doC);
  tasks.schedule([&stateD]() { doD(stateD); });
} // TaskSet's destructor waits for all scheduled tasks to finish
```

### ConcurrentTaskSet

Build a tree in parallel using recursive task scheduling:

```cpp
struct Node {
  int val;
  std::unique_ptr<Node> left, right;
};
void buildTree(dispenso::ConcurrentTaskSet& tasks, std::unique_ptr<Node>& node, int depth) {
  if (depth) {
    node = std::make_unique<Node>();
    node->val = depth;
    tasks.schedule([&tasks, &left = node->left, depth]() { buildTree(tasks, left, depth - 1); });
    tasks.schedule([&tasks, &right = node->right, depth]() { buildTree(tasks, right, depth - 1); });
  }
}
void buildTreeParallel() {
  std::unique_ptr<Node> root;
  dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());
  buildTree(tasks, root, 20);
  tasks.wait();  // tasks would also wait here in destructor if we omitted this line
}
```

### Future

Compose asynchronous operations with futures:

```cpp
dispenso::Future<size_t> ThingProcessor::processThings() {
  auto expensiveFuture = dispenso::async([this]() {
    return processExpensiveThing(expensive_);
  });
  auto futureOfManyCheap = dispenso::async([this]() {
    size_t sum = 0;
    for (auto &thing : cheapThings_) {
      sum += processCheapThing(thing);
    }
    return sum;
  });
  return dispenso::when_all(expensiveFuture, futureOfManyCheap).then([](auto &&tuple) {
    return std::get<0>(tuple).get() + std::get<1>(tuple).get();
  });
}

auto result = thingProc->processThings();
useResult(result.get());
```

### ConcurrentVector

Safely grow a vector from multiple threads:

```cpp
ConcurrentVector<std::unique_ptr<int>> values;
dispenso::parallel_for(
  dispenso::makeChunkedRange(0, length, dispenso::ParForChunking::kStatic),
  [&values](int i, int end) {
    values.grow_by_generator(end - i, [i]() mutable { return std::make_unique<int>(i++); });
  });
```

<div id='benchresults'/>

## Benchmark Results

Dispenso is benchmarked across Linux (x64), macOS (ARM64), Windows (x64), and Android (ARM64),
comparing against OpenMP, TBB, TaskFlow, folly, and `std::async` across thread pools, parallel
loops, futures, graphs, concurrent containers, and more.

**[Interactive Benchmark Dashboard](./docs/benchmarks/index.html)** — explore all results
with platform switching, dark/light theme, and detailed per-benchmark charts.

<div id='installing'/>

## Installing
Binary builds of Dispenso are available through several package managers:

- **Conda**: `conda install -c conda-forge dispenso`
- **Conan**: `conan install --requires=dispenso/1.5.0`
- **vcpkg**: `vcpkg install dispenso`
- **Homebrew**: `brew install dispenso`
- **MacPorts**: `sudo port install dispenso`
- **Fedora/RHEL**: `sudo dnf install dispenso-devel`

If your platform is not on the list, see [the next section](#building) for instructions to build from source.

[![Packaging status](https://repology.org/badge/vertical-allrepos/dispenso.svg)](https://repology.org/project/dispenso/versions)

<div id='building'/>

## Building

**Linux and macOS:**
```bash
mkdir build && cd build
cmake PATH_TO_DISPENSO_ROOT
make -j
```

**Windows** (from Developer Command Prompt):
```bash
mkdir build && cd build
cmake PATH_TO_DISPENSO_ROOT
cmake --build . --config Release
```

For detailed instructions including CMake prerequisites, installation, testing, and
benchmarking, see [docs/building.md](docs/building.md).

<div id='knownissues'/>

## Known Issues

* A subset of dispenso tests are known to fail on 32-bit PPC Mac.  If you have access to such a machine and are willing to help debug, it would be appreciated!

## TODO
* Enable Windows benchmarks through CMake. *(may be resolved soon — actively being worked on)*

<div id='license'/>

## License

The library is released under the MIT license, but also relies on the (excellent) moodycamel concurrentqueue library, which is released under the Simplified BSD and Zlib licenses.  See the top of the source at `dispenso/third-party/moodycamel/*.h` for details.
