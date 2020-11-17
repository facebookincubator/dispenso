cmake_minimum_required(VERSION 3.12)


# Locate GTest
if (NOT WIN32)
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})
else (NOT WIN32)
    add_subdirectory(../../thirdparty/googletest googletestDir)
    include_directories(../../thirdparty/googletest/googlemock/include)
endif (NOT WIN32)


############# Tests
#----------- task_set_no_exceptions
add_library(
    task_set_no_exceptions STATIC 
    ../dispenso/task_set.cpp
    ../dispenso/task_set.h
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
    ./thread_pool_test.cpp
)

target_link_libraries(
    thread_pool_test
    PUBLIC ${GTEST_BOTH_LIBRARIES}
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
    ./task_set_test.cpp
)

target_link_libraries(
    task_set_test
    PUBLIC ${GTEST_BOTH_LIBRARIES}
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
    ./task_set_test.cpp
)

target_compile_options(task_set_test_sans_exceptions PUBLIC -fno-exceptions)

target_link_libraries(
    task_set_test_sans_exceptions
    PUBLIC ${GTEST_BOTH_LIBRARIES}
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
    ./greedy_for_test.cpp
)

target_link_libraries(
    greedy_for_test
    PUBLIC ${GTEST_BOTH_LIBRARIES}
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
    ./chunked_for_test.cpp
)

target_link_libraries(
    chunked_for_test
    PUBLIC ${GTEST_BOTH_LIBRARIES}
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
    ./resource_pool_test.cpp
)

target_link_libraries(
    resource_pool_test
    PUBLIC ${GTEST_BOTH_LIBRARIES}
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
    ./small_buffer_allocator_test.cpp
)

target_link_libraries(
    small_buffer_allocator_test
    PUBLIC ${GTEST_BOTH_LIBRARIES}
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
    ./once_function_test.cpp
)

target_link_libraries(
    once_function_test
    PUBLIC ${GTEST_BOTH_LIBRARIES}
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
    ./future_test.cpp
)

target_link_libraries(
    future_test
    PUBLIC future
    PUBLIC ${GTEST_BOTH_LIBRARIES}
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
    ./future_test.cpp
)

target_compile_options(future_test_sans_exceptions PUBLIC -fno-exceptions)

target_link_libraries(
    future_test_sans_exceptions
    PUBLIC ${GTEST_BOTH_LIBRARIES}
    PUBLIC future
    PUBLIC task_set_no_exceptions
)

if (NOT WIN32)
target_link_libraries(
    future_test_sans_exceptions
	PUBLIC pthread
)
endif (NOT WIN32)

