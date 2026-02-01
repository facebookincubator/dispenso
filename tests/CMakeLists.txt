# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.14)

### TODO(bbudge): Set up testing with github actions
# Ideally these tests are all run in (Release, Debug) X (N/A, TSAN, ASAN, -fno-exceptions)
###

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
include(GoogleTest)

macro(package_add_test TEST_NAME LABEL TEST_FILE)
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_compile_features(${TEST_NAME} PRIVATE cxx_std_14)
    target_compile_options(${TEST_NAME} PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/W3 /WX>
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Wconversion -Wno-sign-conversion -Werror>
      )
    target_link_libraries(${TEST_NAME} gmock_main dispenso)
    gtest_discover_tests(${TEST_NAME}
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      LABELS "${LABEL}")
    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()

file(GLOB TEST_FILES CONFIGURE_DEPENDS "*test.cpp")

#TODO(elliotsegal, bbudge): Help add the shared_pool_test for CMake
LIST(REMOVE_ITEM TEST_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/shared_pool_test.cpp)

# Filter out priority_test specifically because it is inherently flaky - it relies on OS scheduling
# behaviors that are not guaranteed, and is only really useful for manual runs when making changes
# to the related functionality.
LIST(REMOVE_ITEM TEST_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/priority_test.cpp)

foreach(TEST_FILE ${TEST_FILES})
  set(TEST_NAME)
  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
  package_add_test(${TEST_NAME} unittest ${TEST_FILE})
endforeach()

SET(FLAKY_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/priority_test.cpp)

foreach(TEST_FILE ${FLAKY_TEST_FILES})
  set(TEST_NAME)
  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
  package_add_test(${TEST_NAME} flaky ${TEST_FILE})
endforeach()
