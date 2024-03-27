# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.12)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cpp)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS *.h)
message("SOURCES:  ${SOURCES}")

if(DISPENSO_SHARED_LIB)
  add_compile_definitions(DISPENSO_SHARED_LIB DISPENSO_LIB_EXPORT)
  add_library(dispenso SHARED ${SOURCES} ${HEADERS})

  target_compile_options(dispenso PRIVATE
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fvisibility=hidden>
  )
else()
  add_library(dispenso STATIC ${SOURCES} ${HEADERS})
endif()

target_compile_options(dispenso PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /WX>
  $<$<CXX_COMPILER_ID:GNU>: -Wno-stringop-overflow>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Wconversion -Wno-sign-conversion -Werror>
)

if(WIN32)
  target_compile_definitions(dispenso PUBLIC NOMINMAX)
endif()

target_include_directories(dispenso
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/third-party/moodycamel>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/dispenso/third-party/moodycamel>
)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
target_link_libraries(dispenso PUBLIC Threads::Threads)

check_cxx_source_compiles("
#include <atomic>
#include <stdint.h>
std::atomic<int8_t> a(0);
std::atomic<int16_t> b(0);
std::atomic<int32_t> c(0);
std::atomic<int64_t> d(0);
int main() {
  ++a;
  ++b;
  ++c;
  return ++d;
}
" DISPENSO_HAS_ATOMIC_WITHOUT_LIB)

if (NOT DISPENSO_HAS_ATOMIC_WITHOUT_LIB)
  target_link_libraries(dispenso PUBLIC atomic)
endif()

if(WIN32)
  target_link_libraries(dispenso PUBLIC Synchronization Winmm)
endif()

if (NOT DISPENSO_STANDALONE)
  return()
endif()

## Install library ##

set_target_properties(dispenso
    PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

install(TARGETS dispenso
  EXPORT ${PROJECT_NAME}_Exports
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    NAMELINK_SKIP
  # on Windows put the dlls into bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  # ... and the import lib into the devel package
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT ${PROJECT_NAME}_Exports
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
  NAMESPACE Dispenso::
)

install(TARGETS dispenso
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    NAMELINK_ONLY
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

## Install headers ##

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
    PATTERN *.h
)

## Generate and install CMake target exports ##

include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
INSTALL_DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
)

write_basic_package_version_file(
  "${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
)
