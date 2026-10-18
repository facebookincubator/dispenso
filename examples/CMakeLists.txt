# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.12)

# Macro to add example executables. Each one is also registered as a CTest
# smoke test (label "example"), so CI runs the examples rather than only
# compiling them; see run_example.cmake for what counts as passing.
macro(add_example EXAMPLE_NAME)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}.cpp)
    target_link_libraries(${EXAMPLE_NAME} dispenso)
    target_compile_features(${EXAMPLE_NAME} PRIVATE cxx_std_14)
    # Same warnings as the library and tests; fatal under DISPENSO_WERROR.
    target_compile_options(${EXAMPLE_NAME} PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4316>
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Wconversion -Wno-sign-conversion>
      $<$<AND:$<BOOL:${DISPENSO_WERROR}>,$<CXX_COMPILER_ID:MSVC>>:/WX>
      $<$<AND:$<BOOL:${DISPENSO_WERROR}>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>
    )

    # Examples that announce completion must get there.
    file(STRINGS ${EXAMPLE_NAME}.cpp _example_done REGEX "completed successfully")
    if(_example_done)
      set(_example_expect_done ON)
    else()
      set(_example_expect_done OFF)
    endif()
    add_test(NAME example.${EXAMPLE_NAME}
      COMMAND ${CMAKE_COMMAND}
        -DEXAMPLE=$<TARGET_FILE:${EXAMPLE_NAME}>
        -DEXPECT_DONE=${_example_expect_done}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_example.cmake)
    set_tests_properties(example.${EXAMPLE_NAME} PROPERTIES
      LABELS example
      TIMEOUT 120)
endmacro()

# Add all examples
add_example(parallel_for_example)
add_example(for_each_example)
add_example(task_set_example)
add_example(future_example)
add_example(graph_example)
add_example(pipeline_example)
add_example(concurrent_vector_example)
add_example(latch_example)
add_example(resource_pool_example)
add_example(cpu_set_example)
add_example(chase_lev_deque_example)
add_example(mpmc_ring_buffer_example)
add_example(spsc_ring_buffer_example)
add_example(parallel_invoke_example)
add_example(small_vector_example)
add_example(chunking_comparison_example)
