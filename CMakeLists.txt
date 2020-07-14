cmake_minimum_required(VERSION 3.12)

project(Dispenso VERSION 0.1.0)


if (WIN32)
add_compile_definitions(NOMINMAX)
endif (WIN32)

#normally compiles with 14 but MSVC requeired 17
if (WIN32)
set(CMAKE_CXX_STANDARD 17)
else (WIN32)
set(CMAKE_CXX_STANDARD 14)
endif (WIN32)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (WIN32)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif (WIN32)

# Setup VCPKG for windows
if (WIN32)
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE static)
set(VCPKG_LIBRARY_LINKAGE static)
endif (WIN32)


# Locate GTest
if (NOT WIN32)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
else (NOT WIN32)
add_subdirectory(../thirdparty/googletest googletestDir)
endif (NOT WIN32)



# Locate google benchmark
set(BENCHMARK_ENABLE_TESTING OFF)
add_subdirectory(../thirdparty/benchmark benchmarkDir)

include_directories(./dispenso/benchmarks)


# Locate OpenMP
if (NOT WIN32)
find_package(OpenMP)
endif (NOT WIN32)

# Locate TBB
add_subdirectory(../thirdparty/tbb tbbDir)

# Locate folly
if (NOT WIN32)
    find_package(folly CONFIG REQUIRED)
    include_directories(${FOLLY_INCLUDE_DIR})
endif (NOT WIN32)

# Locate gflags
if (NOT WIN32)
find_package(gflags REQUIRED)
endif (NOT WIN32)

# Add front most directories to include path
include_directories(./)
include_directories(./dispenso/)


############# External libraries
add_library(
    concurrentqueue STATIC 
    ./dispenso/ext/moodycamel/concurrentqueue.h
    ./dispenso/empty.cpp
)

target_include_directories(concurrentqueue PUBLIC ./dispenso/ext/moodycamel/)
set_target_properties(concurrentqueue PROPERTIES LINKER_LANGUAGE CXX)


add_library(
    lightweightsemaphore STATIC 
    ./dispenso/ext/moodycamel/lightweightsemaphore.h
    ./dispenso/empty.cpp
)

target_include_directories(lightweightsemaphore PUBLIC ./dispenso/ext/moodycamel/)
set_target_properties(lightweightsemaphore PROPERTIES LINKER_LANGUAGE CXX)


add_library(
    blockingconcurrentqueue STATIC 
    ./dispenso/ext/moodycamel/blockingconcurrentqueue.h
    ./dispenso/empty.cpp
)

target_link_libraries(
    blockingconcurrentqueue
    PUBLIC concurrentqueue
    PUBLIC lightweightsemaphore
)

target_include_directories(blockingconcurrentqueue PUBLIC ./dispenso/ext/moodycamel/)
set_target_properties(blockingconcurrentqueue PROPERTIES LINKER_LANGUAGE CXX)


############# Libraries
#----------- platform
add_library(
    platform STATIC 
    ./dispenso/platform.h
    ./dispenso/empty.cpp
)

set_target_properties(platform PROPERTIES LINKER_LANGUAGE CXX)


#----------- completion_event
add_library(
    completion_event STATIC 
    ./dispenso/platform.h
    ./dispenso/completion_event.h
    ./dispenso/detail/completion_event_impl.h
    ./dispenso/empty.cpp
)

set_target_properties(completion_event PROPERTIES LINKER_LANGUAGE CXX)
if (WIN32)
    target_link_libraries(completion_event PUBLIC synchronization)
endif (WIN32)


#----------- small_buffer_allocator
add_library(
    small_buffer_allocator STATIC 
    ./dispenso/small_buffer_allocator.h
    ./dispenso/empty.cpp
)

set_target_properties(small_buffer_allocator PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(
    small_buffer_allocator
    PUBLIC concurrentqueue
    PUBLIC platform
)


#----------- once_function
add_library(
    once_function STATIC 
    ./dispenso/once_function.h
    ./dispenso/detail/once_callable_impl.h
    ./dispenso/empty.cpp
)

set_target_properties(once_function PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(
    once_function
    PUBLIC small_buffer_allocator
)


############# Detail
#----------- per_pool_per_thread_info
add_library(
    per_pool_per_thread_info STATIC 
    ./dispenso/detail/per_thread_info.h
    ./dispenso/detail/per_thread_info.cpp
)

target_include_directories(per_pool_per_thread_info PUBLIC ./)

target_link_libraries(
    once_function
    PUBLIC concurrentqueue
    PUBLIC platform
)


############# Libraries (cont.)
#----------- thread_pool
add_library(
    thread_pool STATIC 
    ./dispenso/thread_pool.h
    ./dispenso/thread_pool.cpp
)

target_include_directories(thread_pool PUBLIC ./)

target_link_libraries(
    thread_pool
    PUBLIC concurrentqueue
    PUBLIC once_function
    PUBLIC per_pool_per_thread_info
    PUBLIC platform
)


#----------- task_set
add_library(
    task_set STATIC 
    ./dispenso/task_set.h
    ./dispenso/task_set.cpp
)

target_link_libraries(
    task_set
    PUBLIC thread_pool
)

#----------- parallel_for
add_library(
    parallel_for STATIC 
    ./dispenso/parallel_for.h
    ./dispenso/empty.cpp
)

set_target_properties(parallel_for PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(
    parallel_for
    PUBLIC per_pool_per_thread_info
    PUBLIC task_set
)

#----------- future
add_library(
    future STATIC 
    ./dispenso/future.h
    ./dispenso/detail/future_impl.h
    ./dispenso/detail/future_impl2.h
    ./dispenso/empty.cpp
)

target_link_libraries(
    future
    PUBLIC completion_event
    PUBLIC once_function
    PUBLIC task_set
)

#----------- resource_pool
add_library(
    resource_pool STATIC 
    ./dispenso/resource_pool.h
    ./dispenso/empty.cpp
)

set_target_properties(resource_pool PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(
    resource_pool
    PUBLIC blockingconcurrentqueue
    PUBLIC platform
)


############# Tests
#----------- task_set_no_exceptions
add_library(
    task_set_no_exceptions STATIC 
    ./dispenso/task_set.cpp
    ./dispenso/task_set.h
)

target_compile_options(task_set_no_exceptions PUBLIC -fno-exceptions)

target_link_libraries(
    task_set_no_exceptions
    PUBLIC thread_pool
    PUBLIC once_function
    PUBLIC task_set
)

#T---------- thread_pool_test
add_executable(
    thread_pool_test
    ./tests/thread_pool_test.cpp
)

target_link_libraries(
    thread_pool_test
    PUBLIC ${GTEST_LIBRARIES}
    PUBLIC gtest_main
    PUBLIC thread_pool)
	
if (NOT WIN32)
target_link_libraries(
    thread_pool_test
	PUBLIC pthread
)
endif (NOT WIN32)

#T---------- task_set_test
add_executable(
    task_set_test
    ./tests/task_set_test.cpp
)

target_link_libraries(
    task_set_test
    PUBLIC ${GTEST_LIBRARIES}
    PUBLIC gtest_main
    PUBLIC task_set
)

if (NOT WIN32)
target_link_libraries(
    task_set_test
	PUBLIC pthread
)
endif (NOT WIN32)

#T---------- task_set_test_sans_exceptions
add_executable(
    task_set_test_sans_exceptions
    ./tests/task_set_test.cpp
)

target_compile_options(task_set_test_sans_exceptions PUBLIC -fno-exceptions)

target_link_libraries(
    task_set_test_sans_exceptions
    PUBLIC ${GTEST_LIBRARIES}
    PUBLIC gtest_main
    PUBLIC task_set_no_exceptions
)

if (NOT WIN32)
target_link_libraries(
    task_set_test_sans_exceptions
	PUBLIC pthread
)
endif (NOT WIN32)

#T---------- greedy_for_test
add_executable(
    greedy_for_test
    ./tests/greedy_for_test.cpp
)

target_link_libraries(
    greedy_for_test
    PUBLIC ${GTEST_LIBRARIES}
    PUBLIC gtest_main
    PUBLIC parallel_for
)

if (NOT WIN32)
target_link_libraries(
    greedy_for_test
	PUBLIC pthread
)
endif (NOT WIN32)

#T---------- chunked_for_test
add_executable(
    chunked_for_test
    ./tests/chunked_for_test.cpp
)

target_link_libraries(
    chunked_for_test
    PUBLIC ${GTEST_LIBRARIES}
    PUBLIC gtest_main
    PUBLIC parallel_for
)

if (NOT WIN32)
target_link_libraries(
    chunked_for_test
	PUBLIC pthread
)
endif (NOT WIN32)

#T---------- resource_pool_test
add_executable(
    resource_pool_test
    ./tests/resource_pool_test.cpp
)

target_link_libraries(
    resource_pool_test
    PUBLIC ${GTEST_LIBRARIES}
    PUBLIC gtest_main
    PUBLIC resource_pool
    PUBLIC thread_pool
)

if (NOT WIN32)
target_link_libraries(
    resource_pool_test
	PUBLIC pthread
)
endif (NOT WIN32)

#T---------- small_buffer_allocator_test
add_executable(
    small_buffer_allocator_test
    ./tests/small_buffer_allocator_test.cpp
)

target_link_libraries(
    small_buffer_allocator_test
    PUBLIC ${GTEST_LIBRARIES}
    PUBLIC gtest_main
    PUBLIC small_buffer_allocator
)

if (NOT WIN32)
target_link_libraries(
    small_buffer_allocator_test
	PUBLIC pthread
)
endif (NOT WIN32)


#T---------- once_function_test
add_executable(
    once_function_test
    ./tests/once_function_test.cpp
)

target_link_libraries(
    once_function_test
    PUBLIC ${GTEST_LIBRARIES}
    PUBLIC gtest_main
    PUBLIC once_function
)

if (NOT WIN32)
target_link_libraries(
    once_function_test
	PUBLIC pthread
)
endif (NOT WIN32)

#T---------- future_test
add_executable(
    future_test
    ./tests/future_test.cpp
)

target_link_libraries(
    future_test
    PUBLIC future
    PUBLIC ${GTEST_LIBRARIES}
    PUBLIC gtest_main
)

if (NOT WIN32)
target_link_libraries(
    future_test
	PUBLIC pthread
)
endif (NOT WIN32)

#T---------- future_test_sans_exceptions
add_executable(
    future_test_sans_exceptions
    ./tests/future_test.cpp
)

target_compile_options(future_test_sans_exceptions PUBLIC -fno-exceptions)

target_link_libraries(
    future_test_sans_exceptions
    PUBLIC ${GTEST_LIBRARIES}
    PUBLIC gtest_main
    PUBLIC future
    PUBLIC task_set_no_exceptions
)

if (NOT WIN32)
target_link_libraries(
    future_test_sans_exceptions
	PUBLIC pthread
)
endif (NOT WIN32)

############# Benchmarks
#B---------- thread_benchmark_common
add_library(
    thread_benchmark_common STATIC 
    ./benchmarks/thread_benchmark_common.h
    ./dispenso/empty.cpp
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
    ./benchmarks/nested_for_benchmark.cpp
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
    ./benchmarks/future_benchmark.cpp
)

target_link_libraries(
    future_benchmark
    PUBLIC thread_pool
    PUBLIC task_set
    PUBLIC thread_benchmark_common
    PUBLIC folly
    PUBLIC follybenchmark
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
    ./benchmarks/once_function_benchmark.cpp
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
    ./benchmarks/small_buffer_benchmark.cpp
)

target_link_libraries(
    small_buffer_benchmark
    PUBLIC small_buffer_allocator
    PUBLIC thread_benchmark_common
)


#B---------- simple_pool_benchmark
add_executable(
    simple_pool_benchmark
    ./benchmarks/simple_pool_benchmark.cpp
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
    ./benchmarks/trivial_compute_benchmark.cpp
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
    ./benchmarks/simple_for_benchmark.cpp
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
    ./benchmarks/simple_for_benchmark.cpp
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

