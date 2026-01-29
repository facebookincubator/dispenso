# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.12)

# Macro to add example executables
macro(add_example EXAMPLE_NAME)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}.cpp)
    target_link_libraries(${EXAMPLE_NAME} dispenso)
    target_compile_features(${EXAMPLE_NAME} PRIVATE cxx_std_14)
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
