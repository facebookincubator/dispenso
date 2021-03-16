cmake_minimum_required(VERSION 3.12)
project(
  Dispenso
  VERSION 0.1.0
  DESCRIPTION "Dispenso is a library for working with sets of tasks"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Main project setup
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

  # Windows-specific
  if(WIN32)
    add_compile_definitions(NOMINMAX)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)
  endif()
endif()

###########################################################
# Targets
add_subdirectory(dispenso)

set(DISPENSO_BUILD_TESTS OFF CACHE BOOL "Should tests be built?")
set(DISPENSO_BUILD_BENCHMARKS OFF CACHE BOOL "Should benchmarks be built?")

if(DISPENSO_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(DISPENSO_BUILD_BENCHMARKS)
  # Sadly any given release of folly seems to have some problem or another.  Leave disabled by default.
  set(BENCHMARK_WITHOUT_FOLLY ON CACHE BOOL "Should folly benchmarks be disabled?")
  add_subdirectory(benchmarks)
endif()
