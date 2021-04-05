# Dispenso
Dispenso is a library for working with sets of tasks.  It provides mechanisms for thread pools, task sets, parallel for loops, futures, pipelines, and more.

## Comparison of dispenso vs other libraries
### TBB
TBB has significant overlap with dispenso, though TBB has more functionality, and is likely to continue having more utilities for some time.   We chose to build and use dispenso for a few primary reasons like
1. TBB is built on older C++ standards, and doesn't deal well with compiler sanitizers
2. TBB lacks an interface for futures
3. We wanted to ensure we could control performance and availability on non-Intel hardware

Dispenso is faster than TBB in some scenarios and slower in other scenarios.  For example, with parallel for loops, dispenso tends to be faster for small and medium loops, and on-par with TBB for large loops.  When many loops can run independently of one another, dispenso shines and can perform significantly better than TBB.  See the benchmarks (todo) for some concrete examples.

### OpenMP
OpenMP has very simple semantics for parallelizing simple for loops, but gets quite complex for more complicated loops and constructs.  OpenMP wasn't as portable in the past, though the number of compiler supporting it is increasing.  If not used carefully, nesting of OpenMP constructs inside of other threads (e.g. nested parallel for) can lead to large number of threads, which can exhaust machines.

Performance-wise, dispenso tends to outperform simple OpenMP for loops for medium and large workloads, but OpenMP has a significant advantage for small loops.  This is because it has direct compiler support and can understand the cost of the code it is running.  This allows it to forgo running in parallel if the tradeoffs aren't worthwhile.

### Folly
Folly is a library from Facebook that has several concurrency utilities including thread pools and futures.  The library has very good support for new C++ coroutines functionality, and makes writing asynchronous code (e.g. I/O) easy and performant.  Folly as a library can be tricky to work with.  For example, the forward/backward compatibility of code isn't a specific goal of the project.

Folly does not have a parallel loop concept, nor task sets and parallel pipelines.  When comparing Folly's futures against dispenso's, dispenso tries to maintain an API that is closely matched to a combination of std::experimental::future and std::experimental::shared_future (dispenso's futures are all shared).  Additionally, for compute-bound applications, dispenso's futures tend to be much faster and lighter-weight than Folly's.

### Grand central dispatch, new std C++ parallelism, others
We haven't done a strong comparison vs these other mechanisms.  GCD is an Apple technology used by many people for Mac and iOS platforms, and there are ports to other platforms (though the mechanism for submitting closures is different).  Much of the C++ parallel algorithms work is still TBD, but we would be very interested to enable dispenso to be a basis for parallelization of those algorithms.  Additionally, we have interest in enabling dispenso to back the new coroutines interface.  We'd be interested in any contributions people would like to make around benchmarking/summarizing other task parallelism libraries, and also integration with C++ parallel algorithms and coroutines.

## When (currently) *not* to use dispenso
Dispenso isn't really designed for high-latency task offload, it is for computation.  Using the thread pool for networking, disk, or in cases with frequent TLB misses (really any scenario with kernel context switches) may result in poor performance.

If working with futures, `dispenso::Future` can be used with `dispeno::NewThreadInvoker`, which should be roughly equivalent with std::future performance.

If you need async I/O, Folly is likely a good choice (though it still doesn't fix e.g. TLB misses).

## Examples
(todo)

## TODO

* Find a more streamlined approach to obtaining and including dependencies.  
* Add documentation of the benchmark results, and also some examples in the example section.
* GitHub Actions or CircleCI continuous integration testing for linux, mac, windows
* Remove legacy build scripts
* Push to Open Source


# Building dispenso

## Install CMake
Internally to Facebook, we use the Buck build system, but as that relies on a monorepo for relevant dependencies, we do not (yet) ship our BUCK build files.  To enable easy use outside of Facebook monorepos, we ship a CMake build.  Improvements to the CMake build and build files for additional build systems are welcome, as are instructions for building on other platforms, including BSD variants, Windows+Clang, etc... (Note that we should probably expand this section into its own page if we add new build systems).

### Fedora/RPM-based distros
`sudo dnf install cmake`

### MacOS
TODO(bbudge)

### Windows
Install CMake from <https://cmake.org/download/>

## Build dispenso

### Linux and MacOS
1. `mkdir build && cd build`
1. `cmake PATH_TO_DISPENSO_ROOT`
1. `make -j`

### Windows
Install Build Tools for Visual Studio. All commands should be run from the Developer Command Prompt.
1. `mkdir build && cd build`
1. `cmake PATH_TO_DISPENSO_ROOT`
1. `cmake --build . --config Release`

# Building and running dispenso tests
To keep dependencies to an absolute minimum, we do not build tests or benchmarks by default, but only the core library. Building tests requires [GoogleTest](https://github.com/google/googletest).

## Install GoogleTest development libraries

### Fedora/RPM-based distros
`sudo dnf install gtest-devel gmock-devel`

### MacOS
TODO(bbudge)

### Windows
Run `getGTest_windows.sh` in the setupScripts folder (again from the Developer Command Prompt). This will create a folder called thirdparty alongside the dispenso folder.

## Build and run dispenso tests

### Linux and MacOS
1. `mkdir build && cd build`
1. `cmake PATH_TO_DISPENSO_ROOT -DDISPENSO_BUILD_TESTS=ON`
1. `make -j`
1. `ctest`

### Windows
All commands should be run from the Developer Command Prompt.
1. `mkdir build && cd build`
1. `cmake PATH_TO_DISPENSO_ROOT -DDISPENSO_BUILD_TESTS=ON`
1. `cmake --build . --config Release`
1. `ctest`

# Building and running dispenso benchmarks
Dispenso has several benchmarks, and some of these can benchmark against OpenMP, TBB, and/or folly variants.  If benchmarks are turned on via `-DDISPENSO_BUILD_BENCHMARKS=ON`, the build will attempt to find these libraries, and if found, will enable those variants in the benchmarks.  It is important to note that none of these dependencies are dependencies of the dispenso library, but only the benchmark binaries.

The folly variant is turned off by default, because unfortunately it appears to be common to find big issues in many folly releases; note however that the folly code does run and provide benchmark data on our internal facebook platform.

OpenMP should already be available on most platforms that support it (it must be partially built into the compiler after all), but TBB can be had by e.g. `sudo dnf install tbb-devel`.

After you have the deps you want, you can build and run:
### Linux and MacOS
1. `mkdir build && cd build`
1. `cmake PATH_TO_DISPENSO_ROOT -DDISPENSO_BUILD_BENCHMARKS=ON`
1. `make -j`
1. (e.g.) `bin/once_function_benchmark`

### Windows
Not currently supported.

# Known issues

Currently running tests on Windows via `ctest` exhibits failures for Visual Studio 2019, only when building shared dlls.  It appears that the test programs segfault after all tests complete with PASS (after exiting main). 

Visual Studio 2017 works fine.  Static libs also work fine.  Also running the tests standalone outside of `ctest` works fine.  Understanding this problem is a work in progress. 

# License

The library is released under the MIT license, but also relies on the (excellent) moodycamel concurrentqueue library, which is released under the Simplified BSD and Zlib licenses.  See the top of the source at `dispenso/ext/moodycamel/*.h` for details.
