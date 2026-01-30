[![Build and test](https://github.com/facebookincubator/dispenso/actions/workflows/build.yml/badge.svg)](https://github.com/facebookincubator/dispenso/actions/workflows/build.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://facebookincubator.github.io/dispenso)

# Dispenso

**A high-performance C++ thread pool and parallel algorithms library**

Dispenso is a modern **C++ parallel computing library** that provides work-stealing thread pools, parallel for loops, futures, task graphs, and concurrent containers. It serves as a powerful **alternative to OpenMP and Intel TBB**, offering better nested parallelism, sanitizer-clean code, and explicit thread pool control.

**Key advantages over OpenMP and TBB:**
- **No thread explosion** with nested parallel loops - dispenso's work-stealing prevents deadlocks and oversubscription
- **Clean with ASAN/TSAN** - fully sanitizer-compatible, unlike many TBB versions
- **Futures support** - std::experimental::future-like API that TBB lacks
- **Portable** - pure C++14 with no compiler-specific pragmas or extensions
- **Battle-tested** - used in hundreds of projects at Meta (formerly Facebook)

## Table of Contents

- [Features](#features)
- [Quick Start](#quickstart)
- [Comparison vs Other Libraries](#comparison)
- [When Not to Use Dispenso](#nottouse)
- [Documentation and Examples](#examples)
- [Installing](#installing)
- [Building](#building)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Known Issues](#knownissues)
- [License](#license)

<div id='features'/>

## Features

Dispenso provides a comprehensive set of parallel programming primitives:
* **`AsyncRequest`**: Asynchronous request/response facilities for lightweight constrained message passing
* **`CompletionEvent`**: A notifiable event type with wait and timed wait
* **`ConcurrentObjectArena`**: An object arena for fast allocation of objects of the same type
* **`ConcurrentVector`**: A vector-like type with a superset of the TBB concurrent_vector API
* **`for_each`**: Parallel version of `std::for_each` and `std::for_each_n`
* **`Future`**: A futures implementation that strives for interface similarity with std::experimental::future, but with dispenso types as backing thread pools
* **`Graph`**: Utilities for fast and lightweight execution of task graphs.  Task graphs may be reused to avoid setup costs, and subgraph execution is automated based on updated and downstream nodes.
* **`OnceFunction`**: A lightweight function-like interface for `void()` functions that can only be called once
* **`parallel_for`**: Parallel for loops over indices that can be blocking or non-blocking
* **`pipeline`**: Parallel pipelining of workloads
* **`PoolAllocator`**: A pool allocator with facilities to supply a backing allocation/deallocation, making this suitable for use with e.g. CUDA allocation
* **`ResourcePool`**: A type that acts similar to a semaphore around guarded objects
* **`RWLock`**: A minimal reader-writer spin lock that outperforms std::shared_mutex under low write contention
* **`SmallBufferAllocator`**: An allocator that enables fast concurrent allocation for temporary objects
* **`TaskSet`**: Sets of tasks that can be waited on together
* **`ThreadPool`**: The backing thread pool type used by many other dispenso features

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
TBB has significant overlap with dispenso, though TBB has more functionality, and is likely to continue having more utilities for some time. We chose to build and use dispenso for a few primary reasons:
1. TBB is built on older C++ standards, and doesn't deal well with compiler sanitizers
2. TBB lacks an interface for futures
3. We wanted to ensure we could control performance and availability on non-Intel hardware

Dispenso is faster than TBB in some scenarios and slower in other scenarios.  For example, with parallel for loops, dispenso tends to be faster for small and medium loops, and on-par with TBB for large loops.  When many loops can run independently of one another, dispenso shines and can perform significantly better than TBB.  Anecdotally speaking, we have seen one workload with independent parallel for loops at Meta where porting to dispenso led to a 50% speedup.

### OpenMP
OpenMP has very simple semantics for parallelizing simple for loops, but gets quite complex for more complicated loops and constructs.  OpenMP wasn't as portable in the past, though the number of compilers supporting it is increasing.  If not used carefully, nesting of OpenMP constructs inside of other threads (e.g. nested parallel for) can lead to a large number of threads, which can exhaust machines.

Performance-wise, dispenso tends to outperform simple OpenMP for loops for medium and large workloads, but OpenMP has a significant advantage for small loops.  This is because it has direct compiler support and can understand the cost of the code it is running.  This allows it to forgo running in parallel if the tradeoffs aren't worthwhile.

### Folly
Folly is a library from Meta that has several concurrency utilities including thread pools and futures.  The library has very good support for new C++ coroutines functionality, and makes writing asynchronous code (e.g. I/O) easy and performant.  Folly as a library can be tricky to work with.  For example, the forward/backward compatibility of code isn't a specific goal of the project.

Folly does not have a parallel loop concept, nor task sets and parallel pipelines.  When comparing Folly's futures against dispenso's, dispenso tries to maintain an API that is closely matched to a combination of std::experimental::future and std::experimental::shared_future (dispenso's futures are all shared).  Additionally, for compute-bound applications, dispenso's futures tend to be much faster and lighter-weight than Folly's.

### TaskFlow
TaskFlow is a library that seems to have been initially designed for parallel execution of task graphs.  It has some similarities with Dispenso: It has a backing thread pool, it has a parallel_for-like functionality, task graphs, and pipelines.  There are also utilities in each library that don't overlap (yet).  TaskFlow task graphs are pretty high performance, and are sometimes faster than dispenso for full graph execution (depending on platform).  Dispenso has higher-performance task graph building times and has the ability to run partially-modified task graphs much faster.  Dispenso's parallel_for seems to be much lower overhead than TaskFlow's for_each_index (Dispenso is 10x to 100x faster in overhead benchmarks).  Dispenso's pipelines are also much faster and much simpler to construct than TaskFlow's in our benchmarks.

### Grand Central Dispatch (GCD), new std C++ parallelism, others
We haven't done a strong comparison vs these other mechanisms. GCD is an Apple technology used by many people for Mac and iOS platforms, and there are ports to other platforms (though the mechanism for submitting closures is different).  Much of the C++ parallel algorithms work is still TBD, but we would be very interested to enable dispenso to be a basis for parallelization of those algorithms.  Additionally, we have interest in enabling dispenso to back the new coroutines interface.  We'd be interested in any contributions people would like to make around benchmarking/summarizing other task parallelism libraries, and also integration with C++ parallel algorithms and coroutines.

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

<div id='installing'/>

## Installing
Binary builds of Dispenso are available for various Linux distributions through their native package managers, as well as for Windows, macOS, and Linux via the Conda package manager. If your distribution or platform is not on the list, see [the next section](#building) for instructions to build it yourself.

[![Packaging status](https://repology.org/badge/vertical-allrepos/dispenso.svg)](https://repology.org/project/dispenso/versions)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/dispenso/badges/version.svg)](https://anaconda.org/conda-forge/dispenso)

<div id='building'/>

## Building

### Install CMake
Internally to Meta, we use the Buck build system, but as that relies on a monorepo for relevant dependencies, we do not (yet) ship our BUCK build files.  To enable easy use outside of Meta monorepos, we ship a CMake build.  Improvements to the CMake build and build files for additional build systems are welcome, as are instructions for building on other platforms, including BSD variants, Windows+Clang, etc.

<!--- Note that we should probably expand this section into its own page if we add new build systems) --->

### Fedora/RPM-based distros
`sudo dnf install cmake`

### macOS
`brew install cmake`

### Windows
Install CMake from <https://cmake.org/download/>

### Build Dispenso

#### Linux and macOS
1. `mkdir build && cd build`
1. `cmake PATH_TO_DISPENSO_ROOT`
1. `make -j`

#### Windows
Install Build Tools for Visual Studio. All commands should be run from the Developer Command Prompt.
1. `mkdir build && cd build`
1. `cmake PATH_TO_DISPENSO_ROOT`
1. `cmake --build . --config Release`

### Install Dispenso

Once built, the library can be installed by building the "install" target.
Typically on Linux and macOS, this is done with

`make install`

On Windows (and works on any platform), instead do

`cmake --build . --target install`

### Use an Installed Dispenso

Once installed, a downstream CMake project can be pointed to it by using
`CMAKE_PREFIX_PATH` or `Dispenso_DIR`, either as an environment variable or
CMake variable. All that is required to use the library is link the imported
CMake target `Dispenso::dispenso`, which might look like

```cmake
find_package(Dispenso REQUIRED)
target_link_libraries(myDispensoApp Dispenso::dispenso)
```

This brings in all required include paths, library files to link, and any other
properties to the `myDispensoApp` target (your library or application).

<div id='testing'/>

## Testing

To keep dependencies to an absolute minimum, we do not build tests or benchmarks by default, but only the core library. Building tests requires [GoogleTest](https://github.com/google/googletest).

### Build and Run Tests

#### Linux and macOS
1. `mkdir build && cd build`
1. `cmake PATH_TO_DISPENSO_ROOT -DDISPENSO_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release`
1. `make -j`
1. `ctest`

#### Windows
All commands should be run from the Developer Command Prompt.
1. `mkdir build && cd build`
1. `cmake PATH_TO_DISPENSO_ROOT -DDISPENSO_BUILD_TESTS=ON`
1. `cmake --build . --config Release`
1. `ctest`

<div id='benchmarking'/>

## Benchmarking
Dispenso has several benchmarks, and some of these can benchmark against OpenMP, TBB, and/or folly variants.  If benchmarks are turned on via `-DDISPENSO_BUILD_BENCHMARKS=ON`, the build will attempt to find these libraries, and if found, will enable those variants in the benchmarks.  It is important to note that none of these dependencies are dependencies of the dispenso library, but only the benchmark binaries.

The folly variant is turned off by default, because unfortunately it appears to be common to find build issues in many folly releases; note however that the folly code does run and provide benchmark data on our internal Meta platform.

OpenMP should already be available on most platforms that support it (it must be partially built into the compiler after all), but TBB can be had by e.g. `sudo dnf install tbb-devel`.

After you have the deps you want, you can build and run:

#### Linux and macOS
1. `mkdir build && cd build`
1. `cmake PATH_TO_DISPENSO_ROOT -DDISPENSO_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release`
1. `make -j`
1. (e.g.) `bin/once_function_benchmark`

#### Windows
Not currently supported through CMake.

<div id='benchresults'/>

## Benchmark Results
Here are some limited benchmark results.  Unless otherwise noted, these were run on a dual Epyc Rome machine with 128 cores and 256 threads.  One benchmark here was repeated on a Threadripper 2990WX with 32 cores and 64 threads.

Some additional notes about the benchmarks: Your mileage may vary based on compiler, OS/platform, and processor.  These benchmarks were run with default glibc malloc, but use of tcmalloc or jemalloc can significantly boost performance, especially for ConcurrentVector growth operations (`grow_by` and `push_back`).

![plot](./docs/benchmarks/par_tree_build.png)

---

![plot](./docs/benchmarks/nested_for_small.png)
![plot](./docs/benchmarks/nested_for_medium.png)
![plot](./docs/benchmarks/nested_for_large.png)

---

![plot](./docs/benchmarks/pipelines_256thread.png)
![plot](./docs/benchmarks/pipelines_64thread.png)

---

![plot](./docs/benchmarks/concurrent_vector.png)

<div id='knownissues'/>

## Known Issues

* A subset of dispenso tests are known to fail on 32-bit PPC Mac.  If you have access to such a machine and are willing to help debug, it would be appreciated!

## TODO
* Enable Windows benchmarks through CMake.

<div id='license'/>

## License

The library is released under the MIT license, but also relies on the (excellent) moodycamel concurrentqueue library, which is released under the Simplified BSD and Zlib licenses.  See the top of the source at `dispenso/third-party/moodycamel/*.h` for details.
