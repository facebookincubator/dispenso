# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.12)


# Try to find Taskflow: system install, DISPENSO_DEPS_DIR, ~/dispenso_deps, then FetchContent.
find_package(Taskflow QUIET)
if(NOT Taskflow_FOUND)
  # Check well-known local directories for a Taskflow source tree.
  # Searches for both "taskflow" and versioned names like "taskflow-3.11.0".
  if(DEFINED DISPENSO_DEPS_DIR)
    file(GLOB _taskflow_dep_candidates "${DISPENSO_DEPS_DIR}/taskflow*")
    list(APPEND _taskflow_search_dirs ${_taskflow_dep_candidates})
  endif()
  file(GLOB _taskflow_home_candidates "$ENV{HOME}/dispenso_deps/taskflow*")
  list(APPEND _taskflow_search_dirs ${_taskflow_home_candidates})

  set(_taskflow_local_dir "")
  foreach(_dir ${_taskflow_search_dirs})
    if(EXISTS "${_dir}/taskflow/taskflow.hpp")
      set(_taskflow_local_dir "${_dir}")
      break()
    endif()
  endforeach()

  if(_taskflow_local_dir)
    message(STATUS "Found local Taskflow at ${_taskflow_local_dir}")
    add_library(taskflow INTERFACE)
    target_include_directories(taskflow INTERFACE "${_taskflow_local_dir}")
  else()
    message(STATUS "Taskflow not found locally, fetching from GitHub...")
    include(FetchContent)
    message(STATUS "Using up-to-date taskflow")
    FetchContent_Declare(
      taskflow
      GIT_REPOSITORY https://github.com/taskflow/taskflow.git
      GIT_TAG        v3.6.0
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
    )
    FetchContent_GetProperties(taskflow)
    if(NOT taskflow_POPULATED)
      FetchContent_Populate(taskflow)
    endif()
    FetchContent_MakeAvailable(taskflow)
    add_library(taskflow INTERFACE)
    target_include_directories(taskflow INTERFACE ${taskflow_SOURCE_DIR})
  endif()
else()
  message(STATUS "Found system Taskflow")
  if(NOT TARGET taskflow)
    add_library(taskflow INTERFACE)
    target_link_libraries(taskflow INTERFACE Taskflow::Taskflow)
  endif()
endif()

find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
  message(STATUS "Google Benchmark not found locally, fetching from GitHub...")
  include(FetchContent)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests")
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable gtest tests")
  FetchContent_MakeAvailable(benchmark)
else()
  message(STATUS "Found system Google Benchmark")
endif()
# OpenMP support (including Windows/MSVC)
find_package(OpenMP)

# TBB support - prefer CONFIG mode to find TBBConfig.cmake from vcpkg/modern installs
find_package(TBB CONFIG QUIET)
if(NOT TBB_FOUND)
  # Fall back to old FindTBB.cmake module
  find_package(TBB QUIET)
endif()

find_package(folly)

if (WIN32)
  set (REQUIRED_LIBS dispenso benchmark::benchmark benchmark::benchmark_main taskflow)
else (WIN32)
  set (REQUIRED_LIBS dispenso benchmark::benchmark benchmark::benchmark_main pthread taskflow)
endif (WIN32)

if (TBB_FOUND)
  # Handle both old TBB (lowercase 'tbb' target) and new oneTBB (TBB::tbb target)
  if(TARGET TBB::tbb)
    set (OPTIONAL_LIBS ${OPTIONAL_LIBS} TBB::tbb)
  elseif(TARGET tbb)
    set (OPTIONAL_LIBS ${OPTIONAL_LIBS} tbb)
  else()
    set (OPTIONAL_LIBS ${OPTIONAL_LIBS} TBB::tbb)
  endif()
else (TBB_FOUND)
  add_compile_definitions(BENCHMARK_WITHOUT_TBB)
endif (TBB_FOUND)

if (OpenMP_CXX_FOUND)
  set (OPTIONAL_LIBS ${OPTIONAL_LIBS} OpenMP::OpenMP_CXX)
endif (OpenMP_CXX_FOUND)

if (FOLLY_LIBRARIES AND NOT ${BENCHMARK_WITHOUT_FOLLY})
  find_package(gflags)
  set (OPTIONAL_LIBS ${OPTIONAL_LIBS} ${FOLLY_LIBRARIES})
else (FOLLY_LIBRARIES AND NOT ${BENCHMARK_WITHOUT_FOLLY})
  add_compile_definitions(BENCHMARK_WITHOUT_FOLLY)
endif (FOLLY_LIBRARIES AND NOT ${BENCHMARK_WITHOUT_FOLLY})

# Record the comparison libraries that were actually found, so published
# benchmark results say what they were measured against. `run_benchmarks.py`
# reads this file and merges it into the results JSON.
#
# Taskflow states its version only as a header macro, and the header may come
# from a package, a local tree, or FetchContent, so resolve it from whichever
# applies. TF_VERSION packs the parts as MAJOR*100000 + MINOR*100 + PATCH.
set(_dispenso_taskflow_version "${Taskflow_VERSION}")
if(NOT _dispenso_taskflow_version)
  set(_dispenso_taskflow_hpp "")
  if(_taskflow_local_dir)
    set(_dispenso_taskflow_hpp "${_taskflow_local_dir}/taskflow/taskflow.hpp")
  elseif(DEFINED taskflow_SOURCE_DIR)
    set(_dispenso_taskflow_hpp "${taskflow_SOURCE_DIR}/taskflow/taskflow.hpp")
  endif()
  if(EXISTS "${_dispenso_taskflow_hpp}")
    file(STRINGS "${_dispenso_taskflow_hpp}" _tf_ver_line
         REGEX "^#define TF_VERSION [0-9]+")
    if(_tf_ver_line)
      string(REGEX MATCH "[0-9]+" _tf_ver "${_tf_ver_line}")
      math(EXPR _tf_major "${_tf_ver} / 100000")
      math(EXPR _tf_minor "${_tf_ver} / 100 % 1000")
      math(EXPR _tf_patch "${_tf_ver} % 100")
      set(_dispenso_taskflow_version "${_tf_major}.${_tf_minor}.${_tf_patch}")
    endif()
  endif()
endif()

# folly deliberately has no entry: it ships neither a CMake version variable
# nor a version macro, so there is nothing to read. Pass it to the runner with
# `--dep-version folly=<version>` when it matters for reproducibility.
set(_dispenso_dep_versions "${CMAKE_BINARY_DIR}/dispenso_dep_versions.json")
file(WRITE  "${_dispenso_dep_versions}" "{\n")
file(APPEND "${_dispenso_dep_versions}" "  \"tbb\": \"${TBB_VERSION}\",\n")
file(APPEND "${_dispenso_dep_versions}" "  \"google_benchmark\": \"${benchmark_VERSION}\",\n")
file(APPEND "${_dispenso_dep_versions}" "  \"taskflow\": \"${_dispenso_taskflow_version}\",\n")
file(APPEND "${_dispenso_dep_versions}" "  \"openmp\": \"${OpenMP_CXX_VERSION}\",\n")
file(APPEND "${_dispenso_dep_versions}" "  \"openmp_spec_date\": \"${OpenMP_CXX_SPEC_DATE}\"\n")
file(APPEND "${_dispenso_dep_versions}" "}\n")
message(STATUS "Wrote dependency versions to ${_dispenso_dep_versions}")

file(GLOB BENCHMARK_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/benchmarks/*.cpp)

foreach(BENCHMARK_FILE ${BENCHMARK_FILES})
  set(BENCHMARK_NAME)
  get_filename_component(BENCHMARK_NAME ${BENCHMARK_FILE} NAME_WE)
  add_executable(${BENCHMARK_NAME} ${BENCHMARK_FILE})
  target_compile_features(${BENCHMARK_NAME} PRIVATE cxx_std_20)
  target_link_libraries(${BENCHMARK_NAME} ${REQUIRED_LIBS} ${OPTIONAL_LIBS})
endforeach()

if(DISPENSO_BUILD_FAST_MATH)
  add_subdirectory(fast_math)
endif()
