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
add_subdirectory(dispenso)

if(BUILD_TESTS)
    ############# Tests
    add_subdirectory(tests)
endif (BUILD_TESTS)

if(BUILD_BENCHMARKS)
    ############# Benchmarks
    add_subdirectory(benchmarks)
endif (BUILD_BENCHMARKS)

