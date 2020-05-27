cmake_minimum_required(VERSION 3.12)

project(Dispenso VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(./)

############# External libraries
add_library(
    concurrentqueue
    ./dispenso/ext/moodycamel/concurrentqueue.h
)

set_target_properties(concurrentqueue PROPERTIES LINKER_LANGUAGE CXX)


add_library(
    lightweightsemaphore
    ./dispenso/ext/moodycamel/lightweightsemaphore.h
)

set_target_properties(lightweightsemaphore PROPERTIES LINKER_LANGUAGE CXX)


add_library(
    blockingconcurrentqueue
    ./dispenso/ext/moodycamel/blockingconcurrentqueue.h
)

target_link_libraries(
    blockingconcurrentqueue
    PUBLIC concurrentqueue
    PUBLIC lightweightsemaphore
)

set_target_properties(blockingconcurrentqueue PROPERTIES LINKER_LANGUAGE CXX)


############# Libraries
#----------- platform
add_library(
    platform
    ./dispenso/platform.h
)

set_target_properties(platform PROPERTIES LINKER_LANGUAGE CXX)


#----------- completion_event
add_library(
    completion_event
    ./dispenso/platform.h
    ./dispenso/completion_event.h
    ./dispenso/detail/completion_event_impl.h
)

set_target_properties(completion_event PROPERTIES LINKER_LANGUAGE CXX)

if (WIN32)
    add_library( synchronization SHARED IMPORTED )
    set_target_properties( synchronization PROPERTIES IMPORTED_LOCATION ../third-party/toolchains/synchronization.lib )

    target_link_libraries(completion_event synchronization)
endif (WIN32)


#----------- small_buffer_allocator
add_library(
    small_buffer_allocator
    ./dispenso/small_buffer_allocator.h
)

set_target_properties(small_buffer_allocator PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(
    small_buffer_allocator
    PUBLIC concurrentqueue
    PUBLIC platform
)


#----------- once_function
add_library(
    once_function
    ./dispenso/once_function.h
    ./dispenso/detail/once_callable_impl.h
)

set_target_properties(once_function PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(
    once_function
    PUBLIC small_buffer_allocator
)

############# Detail
#----------- per_pool_per_thread_info
add_library(
    per_pool_per_thread_info
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
    thread_pool
    ./dispenso/thread_pool.h
    ./dispenso/thread_pool.cpp
)

target_include_directories(thread_pool PUBLIC ./)
target_include_directories(thread_pool PUBLIC ./dispenso/ext/moodycamel/)

target_link_libraries(
    thread_pool
    PUBLIC concurrentqueue
    PUBLIC once_function
    PUBLIC per_pool_per_thread_info
    PUBLIC platform
)


#----------- task_set
add_library(
    task_set
    ./dispenso/task_set.h
    ./dispenso/task_set.cpp
)

target_link_libraries(
    task_set
    PUBLIC thread_pool
)

#----------- parallel_for
add_library(
    parallel_for
    ./dispenso/parallel_for.h
)

set_target_properties(parallel_for PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(
    parallel_for
    PUBLIC per_pool_per_thread_info
    PUBLIC task_set
)

#----------- future
add_library(
    future
    ./dispenso/future.h
    ./dispenso/detail/future_impl.h
    ./dispenso/detail/future_impl2.h
)

set_target_properties(future PROPERTIES LINKER_LANGUAGE CXX)


target_link_libraries(
    future
    PUBLIC completion_event
    PUBLIC once_function
    PUBLIC task_set
)



