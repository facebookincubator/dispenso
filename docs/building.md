# Building Dispenso {#building}

Dispenso builds with CMake. A copy is also built with Buck inside Meta's monorepo; no Buck
files ship in this repository.

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

When dispenso is built as part of another project (`add_subdirectory()` or
`FetchContent`), it does not set `CMAKE_CXX_STANDARD`; it builds with whatever
standard the parent project uses. The `dispenso` target requires at least C++14
through `target_compile_features`, so a parent on an older standard is raised to
C++14 for dispenso and for targets that link it.

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

### With CMake presets

With CMake 3.21 or newer, the presets in `CMakePresets.json` replace the commands
above. Each builds into `build/<preset>` under the source tree:

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

| Preset | Configuration |
| --- | --- |
| `default` | The library alone at C++17 in Release, using system-installed dependencies |
| `vcpkg` | As `default`, with dependencies from vcpkg; `VCPKG_ROOT` must be set |
| `dev` | Tests, benchmarks and examples at C++20 in Release, with dependencies from vcpkg |
| `asan`, `tsan` | `dev` built as RelWithDebInfo with Address (plus UndefinedBehavior) or Thread Sanitizer |
| `win-dev` | `dev` on Windows with Ninja, built under `%TEMP%`; run from a Developer Command Prompt |

Release and RelWithDebInfo both define `NDEBUG`, so asserts are off in every preset,
the sanitizer presets included. Configure with `-DCMAKE_BUILD_TYPE=Debug` to enable them.

Personal presets belong in `CMakeUserPresets.json`, which git ignores.

## Build Options Affecting Linkage

Two options change what the built library is, and both matter to packagers:

- `DISPENSO_SHARED_LIB` (default **ON**) selects a shared library. Set it to
  `OFF` for a static build. Note the default: a packager wanting a static
  library has to ask for one. A shared build also defines the preprocessor
  macro `DISPENSO_SHARED_LIB` for consumers of the `Dispenso::dispenso`
  target; on Windows that makes dispenso's headers import their symbols from
  the DLL. Builds that do not use the CMake target must define
  `DISPENSO_SHARED_LIB` themselves when linking against a shared dispenso.
- `DISPENSO_USE_SYSTEM_CONCURRENTQUEUE` (default OFF) builds against an
  installed [concurrentqueue](https://github.com/cameron314/concurrentqueue)
  rather than the copy bundled in `dispenso/third-party/moodycamel`. The
  bundled copy tracks the same upstream version.

If you build with `DISPENSO_USE_SYSTEM_CONCURRENTQUEUE=ON`, do **not** add a
version argument to its `find_package`. concurrentqueue's own `CMakeLists.txt`
declares `project(concurrentqueue VERSION 1.0.0)` at every release, so the
config file it installs always reports `1.0.0` -- a value matching no released
tag. Any version constraint therefore rejects a correct installation. dispenso
1.6.1 shipped such a constraint and had to be superseded by 1.6.2 within a day.

`moodycamel::ConcurrentQueue` is a by-value member of `dispenso::ThreadPool`,
so whichever copy is used forms part of dispenso's ABI. Building dispenso
against one version while a consumer compiles against another is an ODR
violation that nothing in the build will diagnose.

### Include paths without CMake

dispenso's public headers include `<moodycamel/concurrentqueue.h>`. Where that
resolves from depends on which copy you built against, which matters if you are
driving the compiler yourself rather than consuming the exported CMake target:

| Build | Header location | Include flag needed |
|---|---|---|
| Bundled (default) | `<prefix>/include/moodycamel/` | `-I<prefix>/include` |
| System concurrentqueue | `<cq-prefix>/include/concurrentqueue/moodycamel/` | also `-I<cq-prefix>/include/concurrentqueue` |

The extra level in the second row is concurrentqueue's own install layout, not
something dispenso or any packager adds. Its CMake target sets the matching
include directory, so `find_package(Dispenso)` needs none of this; only
hand-rolled builds do.

## Other Build Options

| Option | Default | Effect |
| --- | --- | --- |
| `DISPENSO_BUILD_TESTS` | OFF | Build the unit tests (needs GoogleTest; see Testing below) |
| `DISPENSO_BUILD_BENCHMARKS` | OFF | Build the benchmarks (see Benchmarking below) |
| `DISPENSO_BUILD_EXAMPLES` | OFF | Build the programs in `examples/` |
| `DISPENSO_WERROR` | ON (OFF as a subproject) | Make compiler warnings fatal. Warnings are reported either way; turn this off when building with a compiler dispenso is not tested against, so a new warning does not fail the build. Passing `-Wno-error` in `CMAKE_CXX_FLAGS` does not work, because dispenso's own flags come later. |
| `ADDRESS_SANITIZER` | OFF | Build everything with `-fsanitize=address -fsanitize=undefined` (GCC and Clang) |
| `THREAD_SANITIZER` | OFF | Build everything with `-fsanitize=thread` (GCC and Clang). Use Clang: GCC's ThreadSanitizer does not model `std::atomic_thread_fence` and reports false races in the lock-free queues. |
| `DISPENSO_BUILD_FAST_MATH` | OFF | Build the experimental `fast_math` sublibrary. Its API is unstable. |
| `DISPENSO_FAST_MATH_SIMD` | `none` | SIMD instruction set for the `fast_math` targets: `none`, `native`, `sse4.1`, `avx2`, `avx512` or `neon` |
| `DISPENSO_FAST_MATH_HIGHWAY` | OFF | Add the [Highway](https://github.com/google/highway) backend to `fast_math`, fetching Highway if it is not installed |
| `DISPENSO_FAST_MATH_CUDA` | OFF | Build the `fast_math` CUDA tests (needs the CUDA toolkit) |

The sanitizer options are mutually exclusive. The `asan` and `tsan` presets set them for you.

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

## Using Dispenso as a Subproject

dispenso can also be built as part of your project with `add_subdirectory()` or
`FetchContent` (see the [FAQ](faq.md)). Link the same `Dispenso::dispenso`
target as above. A few defaults differ from a standalone build:

- `DISPENSO_WERROR` defaults to **OFF**, so a warning that only your (possibly
  newer) compiler emits does not fail your build. Set it to `ON` to opt in.
- `DISPENSO_SHARED_LIB` is not defined, so dispenso is built as a static
  library, and `BUILD_SHARED_LIBS` is not consulted. Set `DISPENSO_SHARED_LIB`
  before adding dispenso to get a shared library. To link the static library
  into one of your shared libraries, enable `CMAKE_POSITION_INDEPENDENT_CODE`.
- `CMAKE_CXX_STANDARD` is left to your project; see "C++ Standard" above.
- No install rules are generated.

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
ctest -C Release
```

Visual Studio generators build several configurations into one build tree, so `ctest` needs
`-C` to say which one to test; without it, no tests are found.

## Benchmarking

Dispenso has several benchmarks that can optionally benchmark against OpenMP, TBB, and/or
Folly. When benchmarks are enabled via `-DDISPENSO_BUILD_BENCHMARKS=ON`, the build will
attempt to find these libraries and enable their variants if found. None of these are
dependencies of the dispenso library itself — only the benchmark binaries.

The Folly variant is off by default due to common build issues across Folly releases; it
builds in environments that pin a compatible Folly.

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
All commands should be run from the Developer Command Prompt.
```bash
mkdir build && cd build
cmake PATH_TO_DISPENSO_ROOT -DDISPENSO_BUILD_BENCHMARKS=ON
cmake --build . --config Release
bin\Release\once_function_benchmark.exe  # example benchmark
```

OpenMP comes with MSVC; TBB is easiest to get from vcpkg (see the `vcpkg` preset above).
[`scripts/BENCHMARKING.md`](https://github.com/facebookincubator/dispenso/blob/main/scripts/BENCHMARKING.md)
covers running the full suite and recording results on each platform, Windows included.

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
