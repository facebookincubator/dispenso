# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.12)

# Apply SIMD ISA flags and Highway linkage from parent.
set(FAST_MATH_EXTRA_FLAGS ${DISPENSO_FAST_MATH_SIMD_FLAGS})
set(FAST_MATH_EXTRA_LIBS "")
if(DISPENSO_FAST_MATH_HIGHWAY AND TARGET hwy)
  list(APPEND FAST_MATH_EXTRA_LIBS hwy)
  list(APPEND FAST_MATH_EXTRA_FLAGS -isystem${hwy_SOURCE_DIR})
endif()

file(GLOB FAST_MATH_BENCHMARK_FILES CONFIGURE_DEPENDS *.cpp)

foreach(BENCHMARK_FILE ${FAST_MATH_BENCHMARK_FILES})
  set(BENCHMARK_NAME)
  get_filename_component(BENCHMARK_NAME ${BENCHMARK_FILE} NAME_WE)
  set(BENCHMARK_NAME "fast_math_${BENCHMARK_NAME}")
  add_executable(${BENCHMARK_NAME} ${BENCHMARK_FILE})
  target_compile_features(${BENCHMARK_NAME} PRIVATE cxx_std_17)
  target_compile_options(${BENCHMARK_NAME} PRIVATE ${FAST_MATH_EXTRA_FLAGS})
  target_link_libraries(${BENCHMARK_NAME} ${REQUIRED_LIBS} ${OPTIONAL_LIBS} ${FAST_MATH_EXTRA_LIBS})
endforeach()
