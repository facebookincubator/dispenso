cmake_minimum_required(VERSION 3.12)
project(Dispenso VERSION 0.1.0)

#TODO: Fix windows build

if (WIN32)
#set(CMAKE_TOOLCHAIN_FILE "../thirdparty/vcpkg/scripts/buildsystems/vcpkg.cmake")
add_compile_definitions(NOMINMAX)
endif (WIN32)

#normally compiles with 14 but MSVC requeired 17
if (WIN32)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../thirdparty/vcpkg/installed/x64-windows-static/share/ ${CMAKE_PREFIX_PATH}")
else (WIN32)
set(CMAKE_CXX_STANDARD 14)
endif (WIN32)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

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
include_directories(./dispenso/third-party/moodycamel/)

############# Libraries
add_subdirectory(dispenso)

set(DISPENSO_BUILD_TESTS OFF CACHE BOOL "Should tests be built?")
set(DISPENSO_BUILD_BENCHMARKS OFF CACHE BOOL "Should benchmarks be built?")

if(DISPENSO_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif (DISPENSO_BUILD_TESTS)

if(DISPENSO_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif (DISPENSO_BUILD_BENCHMARKS)
