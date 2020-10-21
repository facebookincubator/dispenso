cmake_minimum_required(VERSION 3.12)

############# Libraries
#----------- platform
add_library(
    platform STATIC 
    ./platform.h
    ./empty.cpp
)

set_target_properties(platform PROPERTIES LINKER_LANGUAGE CXX)


#----------- completion_event
add_library(
    completion_event STATIC 
    ./platform.h
    ./completion_event.h
    ./detail/completion_event_impl.h
    ./empty.cpp
)

set_target_properties(completion_event PROPERTIES LINKER_LANGUAGE CXX)
if (WIN32)
    target_link_libraries(completion_event PUBLIC synchronization)
endif (WIN32)


#----------- small_buffer_allocator
add_library(
    small_buffer_allocator STATIC 
    ./small_buffer_allocator.h
    ./empty.cpp
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
    ./once_function.h
    ./detail/once_callable_impl.h
    ./empty.cpp
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
    ./detail/per_thread_info.h
    ./detail/per_thread_info.cpp
)

target_include_directories(per_pool_per_thread_info PUBLIC ../)

target_link_libraries(
    per_pool_per_thread_info
    PUBLIC concurrentqueue
    PUBLIC platform
)


############# Libraries (cont.)
#----------- thread_pool
add_library(
    thread_pool STATIC 
    ./thread_pool.h
    ./thread_pool.cpp
)

target_include_directories(thread_pool PUBLIC ../)

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
    ./task_set.h
    ./task_set.cpp
)

target_link_libraries(
    task_set
    PUBLIC thread_pool
)

#----------- parallel_for
add_library(
    parallel_for STATIC 
    ./parallel_for.h
    ./empty.cpp
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
    ./future.h
    ./detail/future_impl.h
    ./detail/future_impl2.h
    ./empty.cpp
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
    ./resource_pool.h
    ./empty.cpp
)

set_target_properties(resource_pool PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(
    resource_pool
    PUBLIC blockingconcurrentqueue
    PUBLIC platform
)

