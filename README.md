# Dispenso
Dispenso is a library for working with sets of tasks.  It provides mechanisms for TaskSets, parallel for loops, etc...

## TODO

Find a more streamlined approach to obtaining and including dependencies.

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

Currently running tests on Windows via `ctest` exhibits failures for Visual Studio 2019, only when building shared dlls.  It appears that the test programs segfault after all tests complete with PASS. 

Visual Studio 2017 works fine.  Static libs also work fine.  Also running the tests standalone outside of `ctest` works fine.  Understanding this problem is a work in progress. 

# License

The library is released under the MIT license, but also relies on the (excellent) moodycamel concurrentqueue library, which is released under the Simplified BSD and Zlib licenses.  See the top of the source at `dispenso/ext/moodycamel/*.h` for details.
