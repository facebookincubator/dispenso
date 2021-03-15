cmake_minimum_required(VERSION 3.12)
project(Dispenso VERSION 0.1.0)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Windows-specific
if (WIN32)
  add_compile_definitions(NOMINMAX)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)
endif (WIN32)

# C++ version
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Other configuration
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add front most directories to include path
include_directories(./)
include_directories(./dispenso/)
include_directories(./dispenso/third-party/moodycamel/)

############# Libraries
add_subdirectory(dispenso)

set(DISPENSO_BUILD_TESTS OFF CACHE BOOL "Should tests be built?")
set(DISPENSO_BUILD_BENCHMARKS OFF CACHE BOOL "Should benchmarks be built?")

if(DISPENSO_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif (DISPENSO_BUILD_TESTS)

if(DISPENSO_BUILD_BENCHMARKS)
  # Sadly any given release of folly seems to have some problem or another.  Leave disabled by default.
  set(BENCHMARK_WITHOUT_FOLLY ON CACHE BOOL "Should folly benchmarks be disabled?")
  add_subdirectory(benchmarks)
endif (DISPENSO_BUILD_BENCHMARKS)
