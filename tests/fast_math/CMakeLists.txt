# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.14)

# Apply SIMD ISA flags and Highway linkage from parent.
set(FAST_MATH_EXTRA_FLAGS ${DISPENSO_FAST_MATH_SIMD_FLAGS})
set(FAST_MATH_EXTRA_LIBS "")
if(DISPENSO_FAST_MATH_HIGHWAY AND TARGET hwy)
  list(APPEND FAST_MATH_EXTRA_LIBS hwy)
  # Treat Highway as a system include to suppress warnings from its headers.
  if(MSVC)
    list(APPEND FAST_MATH_EXTRA_FLAGS /external:I ${hwy_SOURCE_DIR})
  else()
    list(APPEND FAST_MATH_EXTRA_FLAGS -isystem${hwy_SOURCE_DIR})
  endif()
endif()

# Scalar function tests (link eval.cpp helper)
file(GLOB FAST_MATH_TEST_FILES CONFIGURE_DEPENDS "*_test.cpp")

# Separate SIMD backend tests (no eval.cpp dependency)
set(SIMD_TEST_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/sse_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/avx_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/avx512_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/neon_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hwy_test.cpp
)

foreach(SIMD_FILE ${SIMD_TEST_FILES})
  list(REMOVE_ITEM FAST_MATH_TEST_FILES ${SIMD_FILE})
endforeach()

# Scalar / util tests: each links eval.cpp
foreach(TEST_FILE ${FAST_MATH_TEST_FILES})
  set(TEST_NAME)
  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
  set(TEST_NAME "fast_math_${TEST_NAME}")
  add_executable(${TEST_NAME} ${TEST_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp)
  target_compile_features(${TEST_NAME} PRIVATE cxx_std_17)
  target_compile_options(${TEST_NAME} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /WX>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Wconversion -Wno-sign-conversion -Werror>
    ${FAST_MATH_EXTRA_FLAGS}
  )
  target_link_libraries(${TEST_NAME} gmock_main gtest dispenso ${FAST_MATH_EXTRA_LIBS})
  gtest_discover_tests(${TEST_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    LABELS "fast_math")
  set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests/fast_math)
endforeach()

# SIMD backend tests: no eval.cpp dependency.
foreach(TEST_FILE ${SIMD_TEST_FILES})
  set(TEST_NAME)
  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
  set(TEST_NAME "fast_math_${TEST_NAME}")
  add_executable(${TEST_NAME} ${TEST_FILE})
  target_compile_features(${TEST_NAME} PRIVATE cxx_std_17)
  target_compile_options(${TEST_NAME} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /WX>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Wconversion -Wno-sign-conversion -Werror>
    ${FAST_MATH_EXTRA_FLAGS}
  )
  target_link_libraries(${TEST_NAME} gmock_main gtest dispenso ${FAST_MATH_EXTRA_LIBS})
  gtest_discover_tests(${TEST_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    LABELS "fast_math")
  set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests/fast_math)
endforeach()
