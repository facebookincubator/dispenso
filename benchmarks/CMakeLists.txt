cmake_minimum_required(VERSION 3.12)

#B---------- thread_benchmark_common
add_library(
    thread_benchmark_common STATIC 
    ./thread_benchmark_common.h
    ../dispenso/empty.cpp
)

target_include_directories(thread_benchmark_common PUBLIC ../thirdparty/benchmark/include/)
set_target_properties(thread_benchmark_common PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(
	thread_benchmark_common 
	PUBLIC benchmark benchmark_main
)

if (NOT WIN32)
target_link_libraries(
    thread_benchmark_common
	PUBLIC pthread
)
endif (NOT WIN32)

target_link_directories(thread_benchmark_common PUBLIC ../thirdparty/benchmark/build/src)
target_include_directories(thread_benchmark_common PUBLIC ../thirdparty/benchmark/include/)


#B---------- nested_for_benchmark
add_executable(
    nested_for_benchmark
    ./nested_for_benchmark.cpp
)

target_link_libraries(
    nested_for_benchmark
    PUBLIC parallel_for
    PUBLIC thread_benchmark_common
    PUBLIC tbb
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(nested_for_benchmark PUBLIC OpenMP::OpenMP_CXX)
endif()


#B---------- future_benchmark
add_executable(
    future_benchmark
    ./future_benchmark.cpp
)

target_link_libraries(
    future_benchmark
    PUBLIC thread_pool
    PUBLIC task_set
    PUBLIC thread_benchmark_common
    PUBLIC folly
    PUBLIC ${Boost_LIBRARIES} ${FOLLY_LIBRARIES} ${FOLLY_LINK_LIBRARIES}
    PUBLIC concurrentqueue
    PUBLIC gflags
)
if (WIN32)
target_link_libraries(
    future_benchmark
    PUBLIC ws2_32
    PUBLIC synchronization
)
endif (WIN32)

#B---------- once_function_benchmark
add_executable(
    once_function_benchmark
    ./once_function_benchmark.cpp
)

target_link_libraries(
    once_function_benchmark
    PUBLIC once_function
    PUBLIC thread_benchmark_common
    PUBLIC concurrentqueue
)

#B---------- small_buffer_benchmark
add_executable(
    small_buffer_benchmark
    ./small_buffer_benchmark.cpp
)

target_link_libraries(
    small_buffer_benchmark
    PUBLIC small_buffer_allocator
    PUBLIC thread_benchmark_common
)


#B---------- simple_pool_benchmark
add_executable(
    simple_pool_benchmark
    ./simple_pool_benchmark.cpp
)

target_link_libraries(
    simple_pool_benchmark
    PUBLIC task_set
    PUBLIC thread_benchmark_common
    PUBLIC tbb
)


#B---------- trivial_compute_benchmark
add_executable(
    trivial_compute_benchmark
    ./trivial_compute_benchmark.cpp
)

target_link_libraries(
    trivial_compute_benchmark
    PUBLIC parallel_for
    PUBLIC thread_benchmark_common
    PUBLIC tbb
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(trivial_compute_benchmark PUBLIC OpenMP::OpenMP_CXX)
endif()

#B---------- simple_for_benchmark
add_executable(
    simple_for_benchmark
    ./simple_for_benchmark.cpp
)

target_link_libraries(
    simple_for_benchmark
    PUBLIC parallel_for
    PUBLIC thread_benchmark_common
    PUBLIC tbb
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(simple_for_benchmark PUBLIC OpenMP::OpenMP_CXX)
endif()


#B---------- summing_for_benchmark
add_executable(
    summing_for_benchmark
    ./simple_for_benchmark.cpp
)

target_link_libraries(
    summing_for_benchmark
    PUBLIC parallel_for
    PUBLIC thread_benchmark_common
    PUBLIC tbb
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(summing_for_benchmark PUBLIC OpenMP::OpenMP_CXX)
endif()

