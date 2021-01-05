# Dispenso
Dispenso is a library for working with sets of tasks.  It provides mechanisms for TaskSets, parallel for loops, etc...

# TODO

Find a more streamlined approach to obtaining and including dependencies.

# Building dispenso

## First install cmake
Internally to Facebook, we use the Buck build system, but as that relies on a monorepo for relevant dependencies, we do not (yet) ship our BUCK build files.  To enable easy use outside of Facebook monorepos, we ship a CMake build.  Improvements to the CMake build and build files for additional build systems are welcome, as are instructions for building on other platforms, including BSD variants, Windows+Clang, etc... (Note that we should probably expand this section into its own page if we add new build systems).

### Fedora/RPM-based distros
`sudo dnf install cmake`

### MacOS
TODO(bbudge)

### Windows
TODO(bbudge)

## Build dispenso
### Linux and MacOS
1. `mkdir build && cd build`
1. `cmake PATH_TO_DISPENSO_ROOT`
1. `make -j`

### Windows/VSCode 
TODO(bbudge)

# Building and running dispenso tests
To keep dependencies to an absolute minimum, we do not build tests or benchmarks by default, but only the core library.  Building tests requires gtest and gmock libraries.

## Installing gtest/gmock development libraries

### Fedora/RPM-based distros
`sudo dnf install gtest-devel gmock-devel`

### MacOS
TODO(bbudge)

### Windows
TODO(bbudge)

## Build and run dispenso tests
### Linux and MacOS
1. `mkdir build && cd build`
1. `cmake PATH_TO_DISPENSO_ROOT -DDISPENSO_BUILD_TESTS=ON`
1. `make -j`
1. `make test`

### Windows
TODO(bbudge)

# Building and running dispenso benchmarks
TODO(bbudge)

# Known issues
* On some platforms an error with folly stating "Folly::folly" includes non-existent path "//include", if this is the case navigate to installed folly-targets.cmake (typically found in /usr/local/lib/cmake/folly) and remove that entry. You may preemptively address this issue by changing the file from the cloned git files of Folly before running cmake.
* On Windows this project builds with MT flag; if MD is desired, changed to cmakelists and setupscripts will be needed.

# License

The library is released under the MIT license, but also relies on the (excellent) moodycamel concurrentqueue library, which is released under the Simplified BSD and Zlib licenses.  See the top of the source at `dispenso/ext/moodycamel/*.h` for details.
