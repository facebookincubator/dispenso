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
- **Portable** - C++14 compatible with no compiler-specific pragmas or extensions; C++17 swaps the compatibility shims for standard facilities and C++20 adds concept constraints for clearer error messages (see [Building](docs/building.md#c-standard))

## Table of Contents

- [Choose Dispenso If...](#choosedispenso)
- [Features](#features)
- [Quick Start](#quickstart)
- [Comparison vs Other Libraries](#comparison)
- [Migration Guides](#migrationguides)
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
* **[`parallel_for`](docs/getting_started.md#your-first-parallel-loop)** — parallel loops over indices, blocking or non-blocking (cascaded), with static or adaptive (work-stealing) chunking; cascading `parallel_for` enables overlapping independent loops without oversubscription
* **[`parallel_invoke`](docs/getting_started.md#parallel-invoke)** — fork-join invocation of heterogeneous tasks; composes naturally with recursive divide-and-conquer
* **[`for_each`](docs/getting_started.md#parallel-iteration-with-for_each)** — parallel `std::for_each` / `std::for_each_n`
* **[`Future`](docs/getting_started.md#futures-for-async-results)** — high-performance thread-safe shared futures with `then()`, `when_all()`, `when_any()`, and an API matching `std::experimental::shared_future`
* **[`Graph`](docs/getting_started.md#task-graphs)** — task graph execution with subgraph support and incremental re-evaluation
* **[`pipeline`](docs/getting_started.md#pipelines)** — parallel pipelining of streaming workloads

**Concurrent containers and synchronization:**
* **[`ConcurrentVector`](docs/getting_started.md#concurrentvector)** — concurrent growable vector, superset of TBB `concurrent_vector` API
* **[`ChaseLevDeque`](docs/getting_started.md#chaselevdeque)** — lock-free SPMC work-stealing deque
* **[`MpmcRingBuffer`](https://facebookincubator.github.io/dispenso/classdispenso_1_1_mpmc_ring_buffer.html)** — bounded multi-producer multi-consumer ring buffer
* **`SPSCRingBuffer`** — lock-free single-producer single-consumer ring buffer
* **[`Latch`](docs/getting_started.md#latch)** — one-shot barrier for thread synchronization
* **[`RWLock`](https://facebookincubator.github.io/dispenso/classdispenso_1_1_r_w_lock.html)** — reader-writer spin lock, outperforms `std::shared_mutex` under low write contention

**General-purpose utilities:**
* **[`CpuSet`](docs/getting_started.md#cpu-affinity-and-topology)** — portable CPU affinity, NUMA topology, and cache-aware thread group building
* **[`SmallVector`](docs/getting_started.md#smallvector)** — inline-storage vector (not thread-aware; similar to `folly::small_vector`)
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

<div id='examples'/>

## Documentation and Examples

- **[Getting Started](docs/getting_started.md)** — tutorials with compilable examples for parallel_for, tasks, futures, graphs, pipelines, containers, and more
- **[API Reference](https://facebookincubator.github.io/dispenso)** — full Doxygen documentation
- **[FAQ](docs/faq.md)** — common questions about performance, exception behavior, and when to use dispenso vs alternatives

<div id='benchresults'/>

## Benchmark Results

Dispenso is benchmarked across Linux (x64), macOS (ARM64), Windows (x64), and Android (ARM64),
comparing against OpenMP, TBB, TaskFlow, folly, and `std::async` across thread pools, parallel
loops, futures, graphs, concurrent containers, and more.

**[Interactive Benchmark Dashboard](https://facebookincubator.github.io/dispenso/benchmarks/)** — explore all results
with platform switching, dark/light theme, and detailed per-benchmark charts.

<div id='installing'/>

## Installing
Binary builds of Dispenso are available through several package managers:

- **Conda**: `conda install -c conda-forge dispenso`
- **Conan**: `conan install --requires=dispenso/1.6.1`
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

**FreeBSD:**
```bash
mkdir build && cd build
cmake PATH_TO_DISPENSO_ROOT
cmake --build . --parallel $(sysctl -n hw.ncpu)
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

## Known Issues and Limitations

* **Parallel reduction** is not a first-class operation. Use `parallel_for` with per-thread state accumulation (see [Getting Started](docs/getting_started.md#your-first-parallel-loop)). A dedicated reduction API is planned.
* **macOS CPU affinity**: `CpuSet::bindCurrentThread()` is a no-op on macOS — the OS does not support explicit CPU pinning. Topology queries work.
* See [GitHub Issues](https://github.com/facebookincubator/dispenso/issues) for the full list.

<div id='license'/>

## License

Dispenso itself is released under the MIT license (see `LICENSE`).

It also bundles the (excellent) moodycamel concurrentqueue library, under
`dispenso/third-party/moodycamel/`, which carries its own terms:

- `concurrentqueue.h` and `blockingconcurrentqueue.h` are dual-licensed under
  the Simplified BSD license **or** the Boost Software License 1.0, at your
  option.
- `lightweightsemaphore.h` additionally embeds Jeff Preshing's semaphore
  implementation, which is under the Zlib license.

The full text is in `dispenso/third-party/moodycamel/LICENSE.md`, which is
installed alongside the headers, so it travels with a packaged copy rather
than only existing in the source tree. Building with
`-DDISPENSO_USE_SYSTEM_CONCURRENTQUEUE=ON` uses an external concurrentqueue
instead and installs none of it.
