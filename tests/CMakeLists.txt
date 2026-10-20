# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.18)

# TODO: CI runs these tests in Release and Debug, and under TSAN and ASAN, but not yet with
# -fno-exceptions.

# Try system-installed GTest first, fall back to FetchContent from GitHub.
find_package(GTest QUIET)
if(GTest_FOUND)
  message(STATUS "Found system GoogleTest")
  # System GTest provides namespaced targets (GTest::gmock_main); create
  # non-namespaced aliases so existing target_link_libraries calls work.
  if(NOT TARGET gmock_main AND TARGET GTest::gmock_main)
    add_library(gmock_main ALIAS GTest::gmock_main)
  endif()
else()
  message(STATUS "System GoogleTest not found, fetching from GitHub...")
  include(FetchContent)
  FetchContent_Declare(
    GoogleTest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.12.1
  )
  # For Windows, Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(GoogleTest)
endif()
include(GoogleTest)

macro(package_add_test TEST_NAME LABEL TEST_FILE)
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_compile_options(${TEST_NAME} PRIVATE
      # /wd4316: see dispenso/CMakeLists.txt — over-aligned OnceFunction in
      # make_shared/make_unique (incl. std::thread internals) is benign on MSVC.
      $<$<CXX_COMPILER_ID:MSVC>:/W3 /WX /wd4316>
      # GCC false positives: stringop-overflow through deeply inlined atomics
      # (seen with GCC 13 on Linux, GCC 14 on macOS ARM64), and
      # maybe-uninitialized through inlined try_pop + gtest macros (GCC 13).
      $<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overflow -Wno-maybe-uninitialized>
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Wconversion -Wno-sign-conversion -Werror>
      )
    target_link_libraries(${TEST_NAME} gmock_main dispenso)
    gtest_discover_tests(${TEST_NAME}
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      LABELS "${LABEL}"
      DISCOVERY_MODE PRE_TEST)
    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()

file(GLOB TEST_FILES CONFIGURE_DEPENDS "*test.cpp")

# shared_pool_test checks that two separate shared libraries see the same globalThreadPool(),
# which only holds when dispenso itself is a shared library; it is registered below.
LIST(REMOVE_ITEM TEST_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/shared_pool_test.cpp)

# Filter out these tests specifically because they are inherently flaky because they rely on OS behaviors that are not
# guaranteed, and only really useful for manual runs when making changes to the related functionality.  Note that
# possibly an even better test for both priority and timed_task behavior is to use the timed_task_benchmark.
LIST(REMOVE_ITEM TEST_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/priority_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/timed_task_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/timing_test.cpp)

foreach(TEST_FILE ${TEST_FILES})
  set(TEST_NAME)
  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
  package_add_test(${TEST_NAME} unittest ${TEST_FILE})
endforeach()

if(DISPENSO_SHARED_LIB)
  # forward_shared_pool.cpp is compiled twice, each copy exporting globalThreadPool()'s address
  # under its own name, so the test can compare what two independent libraries see.
  foreach(SUFFIX A B)
    set(LIB_NAME shared_pool_${SUFFIX})
    add_library(${LIB_NAME} SHARED forward_shared_pool.cpp)
    target_compile_definitions(${LIB_NAME} PRIVATE DISPENSO_EXPORT_NAME=sharedPool${SUFFIX})
    target_link_libraries(${LIB_NAME} PRIVATE dispenso)
    set_target_properties(${LIB_NAME} PROPERTIES FOLDER tests)
  endforeach()
  package_add_test(shared_pool_test unittest ${CMAKE_CURRENT_SOURCE_DIR}/shared_pool_test.cpp)
  target_link_libraries(shared_pool_test shared_pool_A shared_pool_B)
endif()

SET(FLAKY_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/priority_test.cpp ${CMAKE_CURRENT_SOURCE_DIR}/timed_task_test.cpp ${CMAKE_CURRENT_SOURCE_DIR}/timing_test.cpp)

foreach(TEST_FILE ${FLAKY_TEST_FILES})
  set(TEST_NAME)
  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
  package_add_test(${TEST_NAME} flaky ${TEST_FILE})
endforeach()

if(DISPENSO_BUILD_FAST_MATH)
  add_subdirectory(fast_math)
endif()
