cmake_minimum_required(VERSION 3.12)

if (WIN32)
set(glog_DIR "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/packages/glog_x64-windows-static/share/glog")
set(glog_LIBRARY "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/packages/glog_x64-windows-static/include/glog/lib")
set(glog_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/packages/glog_x64-windows-static/include/glog/include")
set(ZLIB_LIBRARY  "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/packages/zlib_x64-windows-static/lib")
set(ZLIB_INCLUDE_DIR  "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/packages/zlib_x64-windows-static/include")
set(folly_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/include")

set(libevent_DIR "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/packages/libevent_x64-windows-static/share/libevent")

set(GLOG_LINK_LIBRARIES "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/lib/glog.lib")
set(FOLLY_LINK_LIBRARIES "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/lib/folly.lib")
set(GFLAGS_LINK_LIBRARIES "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/lib/gflags_static.lib")
set(DOUBLE_CONV_LINK_LIBRARIES "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/lib/double-conversion.lib")
set(BOOST_THREAD_LINK_LIBRARIES "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/lib/boost_thread-vc140-mt.lib")
set(BOOST_CONTEXT_LINK_LIBRARIES "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/lib/boost_context-vc140-mt.lib")
set(EVENT_CORE_LINK_LIBRARIES "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/lib/event_core.lib")
set(EVENT_LINK_LIBRARIES "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/lib/event.lib")
set(EVENT_EXTRA_LINK_LIBRARIES "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/lib/event_extra.lib")

set(GLOG_LINK_LIBRARIES_DBG "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/debug/lib/glog.lib")
set(FOLLY_LINK_LIBRARIES_DBG "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/debug/lib/folly.lib")
set(GFLAGS_LINK_LIBRARIES_DBG "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/debug/lib/gflags_static_debug.lib")
set(DOUBLE_CONV_LINK_LIBRARIES_DBG "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/debug/lib/double-conversion.lib")
set(BOOST_THREAD_LINK_LIBRARIES_DBG "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/debug/lib/boost_thread-vc140-mt-gd.lib")
set(BOOST_CONTEXT_LINK_LIBRARIES_DBG "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/debug/lib/boost_context-vc140-mt-gd.lib")
set(EVENT_CORE_LINK_LIBRARIES_DBG "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/debug/lib/event_core.lib")
set(EVENT_LINK_LIBRARIES_DBG "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/debug/lib/event.lib")
set(EVENT_EXTRA_LINK_LIBRARIES_DBG "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/debug/lib/event_extra.lib")
endif (WIN32)

# Locate google benchmark
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_GTESTS OFF)
add_subdirectory(../../thirdparty/benchmark benchmarkDir)
#set(benchmark_DIR "${CMAKE_SOURCE_DIR}/../thirdparty/benchmark/")

# Locate OpenMP
if (NOT WIN32)
    find_package(OpenMP)
endif (NOT WIN32)

# Locate TBB
add_subdirectory(../../thirdparty/tbb tbbDir)
#set(tbb_DIR "${CMAKE_SOURCE_DIR}/../thirdparty/tbb/")
#set(ZLIB_LIBRARY  "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/packages/zlib_x64-windows-static/lib")
#set(ZLIB_INCLUDE_DIR  "${CMAKE_SOURCE_DIR}/../thirdparty/tbb/include")



message(STATUS "+++++++++++++++++++++${glog_INCLUDE_DIR}")

# Locate folly
if (NOT WIN32)
    find_package(folly CONFIG REQUIRED)
endif (NOT WIN32)
	include_directories(${glog_INCLUDE_DIR})
    #find_package(folly CONFIG REQUIRED)
	include_directories(${folly_INCLUDE_DIR})
	
include_directories(${FOLLY_INCLUDE_DIR})

# Locate gflags
if (NOT WIN32)
    find_package(gflags REQUIRED)
endif (NOT WIN32)


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

find_package(libevent CONFIG REQUIRED)

target_link_libraries(
    future_benchmark
    PUBLIC thread_pool
    PUBLIC task_set
    PUBLIC thread_benchmark_common
    PUBLIC concurrentqueue
)

if (WIN32)
target_link_libraries(
    future_benchmark
	PUBLIC ws2_32
    PUBLIC synchronization
	PUBLIC libevent::core libevent::extra
	
    PUBLIC optimized ${EVENT_CORE_LINK_LIBRARIES}
    PUBLIC optimized ${EVENT_LINK_LIBRARIES}
    PUBLIC optimized ${EVENT_EXTRA_LINK_LIBRARIES}
	PUBLIC optimized ${BOOST_CONTEXT_LINK_LIBRARIES}
	PUBLIC optimized ${BOOST_THREAD_LINK_LIBRARIES}
    PUBLIC optimized ${Boost_LIBRARIES}
	PUBLIC optimized ${DOUBLE_CONV_LINK_LIBRARIES}
    PUBLIC optimized ${GLOG_LINK_LIBRARIES}
	PUBLIC optimized ${FOLLY_LIBRARIES}
	PUBLIC optimized  ${FOLLY_LINK_LIBRARIES}
    PUBLIC optimized ${GFLAGS_LINK_LIBRARIES}
	
    PUBLIC debug ${EVENT_CORE_LINK_LIBRARIES_DBG}
    PUBLIC debug ${EVENT_LINK_LIBRARIES_DBG}
    PUBLIC debug ${EVENT_EXTRA_LINK_LIBRARIES_DBG}
	PUBLIC debug ${BOOST_CONTEXT_LINK_LIBRARIES_DBG}
	PUBLIC debug ${BOOST_THREAD_LINK_LIBRARIES_DBG}
    PUBLIC debug ${Boost_LIBRARIES_DBG}
	PUBLIC debug ${DOUBLE_CONV_LINK_LIBRARIES_DBG}
    PUBLIC debug ${GLOG_LINK_LIBRARIES_DBG}
	PUBLIC debug ${FOLLY_LIBRARIES_DBG}
	PUBLIC debug ${FOLLY_LINK_LIBRARIES_DBG}
    PUBLIC debug ${GFLAGS_LINK_LIBRARIES_DBG}
)
else (WIN32)
target_link_libraries(
    future_benchmark
    PUBLIC folly	
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

