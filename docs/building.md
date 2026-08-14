# Building Dispenso {#building}

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

### C++ Standard

Dispenso is fully supported at its C++14 baseline: every API works and the
build defaults to `-DCMAKE_CXX_STANDARD=14`. Building at C++17 or newer is
nonetheless recommended where your project allows it, because dispenso drops
its compatibility shims in favour of the standard facilities:

| Standard | What dispenso does differently |
| --- | --- |
| C++17 | Uses `std::optional` in place of the bundled `detail::OpResult` (`AsyncRequest`, pipelines), `std::invoke_result_t` in place of deprecated `std::result_of`, and the language's over-aligned `new`/`delete` in place of dispenso's own aligned allocation operators. `[[deprecated]]` also becomes valid on enumerators, so deprecated values such as `ParForChunking::kAuto` start producing warnings. |
| C++20 | Enables `DISPENSO_HAS_CONCEPTS`, which turns the `DISPENSO_REQUIRES` constraints into real `requires` clauses. Misused callables then fail at the call site with a named unsatisfied constraint instead of an error inside a template instantiation. |

Newer standards also tend to produce marginally faster code from the same
compiler, independent of the above. Dispenso's published benchmark results are
built at C++20; the compiler and standard used for each platform are recorded
in `machine_info.compiler` in the result JSON and shown on the benchmark
dashboard.

```bash
cmake PATH_TO_DISPENSO_ROOT -DCMAKE_CXX_STANDARD=20
```

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

## Building the Documentation

The API documentation is generated with Doxygen (graphviz supplies the diagrams):

```bash
cd docs
doxygen Doxyfile
```

Output lands in `docs/doxygen/html`, and warnings are written to
`docs/doxygen_warnings.log`. CI fails the build whenever that log is non-empty, so
treat any warning as an error.

**Doxygen 1.11.0 or newer is required.** Releases 1.9.2 through 1.9.8 fail to resolve
markdown links to a page that declares an explicit `{#label}` anchor, which the
cross-page links in these documents depend on; against those versions the build reports
unresolved-reference warnings that do not reflect a problem in the source. Distribution
packages are frequently older than the minimum — check with `doxygen --version` and
install an official release from
[doxygen.nl](https://www.doxygen.nl/download.html) if yours falls short.
