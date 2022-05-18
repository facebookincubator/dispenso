# 
# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 

cmake_minimum_required(VERSION 3.12)
project(
  Dispenso
  VERSION 1.0.0
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
    add_compile_definitions(NOMINMAX)
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
