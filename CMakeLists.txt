# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


cmake_minimum_required(VERSION 3.12)

# Use /Z7 (embedded debug info) instead of /Zi (PDB server) on MSVC+Ninja.
# /Zi requires mspdbsrv.exe which can fail with C1902 "Program database manager
# mismatch" when Ninja launches cl.exe as a subprocess.
if(POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
endif()

# CMake 4.x enables C++20 module scanning by default with Ninja, which breaks
# try_compile() checks. Dispenso doesn't use C++20 modules.
if(POLICY CMP0155)
  cmake_policy(SET CMP0155 OLD)
endif()
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

project(
  Dispenso
  VERSION 1.5.0
  DESCRIPTION "Dispenso is a library for working with sets of parallel tasks"
  LANGUAGES CXX)

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  set(DISPENSO_STANDALONE TRUE)
else()
  set(DISPENSO_STANDALONE FALSE)
endif()

if (DISPENSO_STANDALONE)
  include(GNUInstallDirs)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Main project setup
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  option(DISPENSO_SHARED_LIB "Build Dispenso shared library" ON)


  # Windows-specific
  if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)
  endif()
endif()

option(ADDRESS_SANITIZER "Use Address Sanitizer, incompatible with THREAD_SANITIZER" OFF)
option(THREAD_SANITIZER "Use Thread Sanitizer, incompatible with ADDRESS_SANITIZER" OFF)

if (ADDRESS_SANITIZER)
  add_compile_options(-fsanitize=address -fsanitize=undefined)
  add_link_options(-fsanitize=address -fsanitize=undefined)
elseif (THREAD_SANITIZER)
  add_compile_options(-fsanitize=thread)
  add_link_options(-fsanitize=thread)
endif()

set(CMAKE_CXX_STANDARD 14 CACHE STRING "the C++ standard to use for this project")

###########################################################
# Targets
add_subdirectory(dispenso)

set(DISPENSO_BUILD_TESTS OFF CACHE BOOL "Should tests be built?")
set(DISPENSO_BUILD_BENCHMARKS OFF CACHE BOOL "Should benchmarks be built?")
set(DISPENSO_BUILD_EXAMPLES OFF CACHE BOOL "Should examples be built?")
set(DISPENSO_BUILD_FAST_MATH OFF CACHE BOOL "Build experimental fast_math sublibrary (API unstable)")
set(DISPENSO_FAST_MATH_SIMD "none" CACHE STRING "SIMD ISA for fast_math targets: none, native, sse4.1, avx2, avx512, neon")
set(DISPENSO_FAST_MATH_HIGHWAY OFF CACHE BOOL "Enable Highway SIMD backend for fast_math (fetches Highway if not found)")

if(DISPENSO_BUILD_FAST_MATH)
  # Set compiler flags for the chosen SIMD backend.
  # Each level is cumulative: avx2 implies sse4.1, avx512 implies avx2+sse4.1.
  # "native" auto-detects all ISA extensions supported by the build machine.
  set(DISPENSO_FAST_MATH_SIMD_FLAGS "")
  if(DISPENSO_FAST_MATH_SIMD STREQUAL "native")
    if(MSVC)
      # MSVC doesn't have -march=native; AVX2 is the best portable flag.
      set(DISPENSO_FAST_MATH_SIMD_FLAGS /arch:AVX2)
    else()
      set(DISPENSO_FAST_MATH_SIMD_FLAGS -march=native)
    endif()
  elseif(DISPENSO_FAST_MATH_SIMD STREQUAL "sse4.1")
    if(NOT MSVC)
      # MSVC enables SSE4.1 by default on x64.
      set(DISPENSO_FAST_MATH_SIMD_FLAGS -msse4.1)
    endif()
  elseif(DISPENSO_FAST_MATH_SIMD STREQUAL "avx2")
    if(MSVC)
      set(DISPENSO_FAST_MATH_SIMD_FLAGS /arch:AVX2)
    else()
      set(DISPENSO_FAST_MATH_SIMD_FLAGS -msse4.1 -mavx2 -mfma)
    endif()
  elseif(DISPENSO_FAST_MATH_SIMD STREQUAL "avx512")
    if(MSVC)
      set(DISPENSO_FAST_MATH_SIMD_FLAGS /arch:AVX512)
    else()
      set(DISPENSO_FAST_MATH_SIMD_FLAGS -msse4.1 -mavx2 -mfma -mavx512f -mavx512bw -mavx512dq)
    endif()
  elseif(DISPENSO_FAST_MATH_SIMD STREQUAL "neon")
    # NEON is implicit on aarch64; no extra flags needed.
  elseif(NOT DISPENSO_FAST_MATH_SIMD STREQUAL "none")
    message(FATAL_ERROR "Unknown DISPENSO_FAST_MATH_SIMD value: ${DISPENSO_FAST_MATH_SIMD}. "
      "Valid options: none, native, sse4.1, avx2, avx512, neon")
  endif()

  # Find or fetch Highway if requested.
  if(DISPENSO_FAST_MATH_HIGHWAY)
    find_package(hwy QUIET)
    if(NOT hwy_FOUND)
      include(FetchContent)
      FetchContent_Declare(
        highway
        GIT_REPOSITORY https://github.com/google/highway.git
        GIT_TAG        1.2.0
      )
      set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
      set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
      set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
      FetchContent_MakeAvailable(highway)
    endif()
  endif()
endif()

if(DISPENSO_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(DISPENSO_BUILD_BENCHMARKS)
  # Sadly any given release of folly seems to have some problem or another.  Leave disabled by default.
  set(BENCHMARK_WITHOUT_FOLLY ON CACHE BOOL "Should folly benchmarks be disabled?")
  add_subdirectory(benchmarks)
endif()

if(DISPENSO_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
