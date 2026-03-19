# Building Dispenso

Dispenso uses CMake as its build system for open-source builds. Internally at Meta, the Buck
build system is used, but Buck build files are not shipped externally.

Improvements to the CMake build and build files for additional build systems are welcome, as
are instructions for building on other platforms (BSD variants, Windows+Clang, etc).

## Prerequisites

### CMake

#### Fedora/RPM-based distros
```bash
sudo dnf install cmake
```

#### macOS
```bash
brew install cmake
```

#### Windows
Install CMake from <https://cmake.org/download/>

## Building the Library

### Linux and macOS
```bash
mkdir build && cd build
cmake PATH_TO_DISPENSO_ROOT
make -j
```

### Windows
All commands should be run from the Developer Command Prompt (install Build Tools for
Visual Studio).
```bash
mkdir build && cd build
cmake PATH_TO_DISPENSO_ROOT
cmake --build . --config Release
```

## Installing

Once built, install by building the "install" target:

**Linux and macOS:**
```bash
make install
```

**Windows (also works on any platform):**
```bash
cmake --build . --target install
```

## Using an Installed Dispenso

A downstream CMake project can be pointed to an installed dispenso by using
`CMAKE_PREFIX_PATH` or `Dispenso_DIR`, either as an environment variable or CMake
variable. All that is required is to link the imported CMake target `Dispenso::dispenso`:

```cmake
find_package(Dispenso REQUIRED)
target_link_libraries(myDispensoApp Dispenso::dispenso)
```

This brings in all required include paths, library files to link, and any other properties
to the `myDispensoApp` target.

## Testing

Tests are not built by default to keep dependencies minimal. Building tests requires
[GoogleTest](https://github.com/google/googletest).

### Linux and macOS
```bash
mkdir build && cd build
cmake PATH_TO_DISPENSO_ROOT -DDISPENSO_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
make -j
ctest
```

### Windows
All commands should be run from the Developer Command Prompt.
```bash
mkdir build && cd build
cmake PATH_TO_DISPENSO_ROOT -DDISPENSO_BUILD_TESTS=ON
cmake --build . --config Release
ctest
```

## Benchmarking

Dispenso has several benchmarks that can optionally benchmark against OpenMP, TBB, and/or
Folly. When benchmarks are enabled via `-DDISPENSO_BUILD_BENCHMARKS=ON`, the build will
attempt to find these libraries and enable their variants if found. None of these are
dependencies of the dispenso library itself — only the benchmark binaries.

The Folly variant is off by default due to common build issues across Folly releases.
However, the Folly benchmarks do run successfully on Meta's internal platform.

OpenMP should already be available on most platforms that support it. TBB can be installed
via e.g. `sudo dnf install tbb-devel`.

### Linux and macOS
```bash
mkdir build && cd build
cmake PATH_TO_DISPENSO_ROOT -DDISPENSO_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
make -j
bin/once_function_benchmark  # example benchmark
```

### Windows
Not currently supported through CMake.
