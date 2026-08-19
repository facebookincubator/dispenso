# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.12)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cpp)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS *.h)

if(NOT DISPENSO_BUILD_FAST_MATH)
  list(FILTER SOURCES EXCLUDE REGEX "fast_math/")
  list(FILTER HEADERS EXCLUDE REGEX "fast_math/")
endif()

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
  # /wd4316: OnceFunction is intentionally cache-line (64B) aligned (see
  # once_function.h); MSVC's make_shared/make_unique control-block allocator does
  # not propagate that over-alignment. Benign here (the alignment is a
  # false-sharing optimization, not a correctness requirement), so silence the
  # warning rather than under-align or drop the optimization.
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4316>
  # GCC false positive: stringop-overflow through deeply inlined atomics.
  # Seen with GCC 13 on Linux and GCC 14 on macOS ARM64.
  $<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overflow>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Wconversion -Wno-sign-conversion>
  # Fatal only when DISPENSO_WERROR is on; see the option in the top-level
  # CMakeLists for why distributors need to be able to turn this off.
  $<$<AND:$<BOOL:${DISPENSO_WERROR}>,$<CXX_COMPILER_ID:MSVC>>:/WX>
  $<$<AND:$<BOOL:${DISPENSO_WERROR}>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>
)

if(WIN32)
  target_compile_definitions(dispenso PUBLIC NOMINMAX)
endif()

target_include_directories(dispenso
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(DISPENSO_USE_SYSTEM_CONCURRENTQUEUE)
  # Deliberately unversioned. A floor belongs here in principle --
  # moodycamel::ConcurrentQueue is a by-value member of ThreadPool, so its
  # layout is part of dispenso's ABI, and a system copy that disagrees with
  # third-party/moodycamel is an ODR problem. It cannot be expressed: upstream
  # has said project(concurrentqueue VERSION 1.0.0) at every release including
  # v1.0.5, so the config version file always reports 1.0.0 -- a number matching
  # no tag -- and any version argument here rejects a correct installation. The
  # header carries no version macro either. 1.6.1 shipped such a floor and could
  # not configure against vcpkg's concurrentqueue at all.
  find_package(concurrentqueue CONFIG REQUIRED)
  target_link_libraries(dispenso PUBLIC concurrentqueue::concurrentqueue)
else()
  # SYSTEM so the compiler does not diagnose third-party code we do not
  # maintain. Without it, any warning dispenso enables is also applied to the
  # bundled headers, and the only way to keep the build clean is to patch them
  # -- which then has to be re-applied on every update, forever, for warnings
  # that say nothing about dispenso.
  #
  # Only the build tree needs the extra directory: the bundled headers install
  # to <prefix>/include/moodycamel, which the
  # $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> above already covers.
  # Consumers therefore see them as ordinary headers; if that ever matters,
  # they can add -isystem themselves.
  target_include_directories(dispenso SYSTEM
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/third-party>
  )
endif()

# Export the C++ standard requirement so downstream consumers compile with at
# least the same standard dispenso was built with (default: C++14).
target_compile_features(dispenso PUBLIC cxx_std_${CMAKE_CXX_STANDARD})

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

# Build list of directory patterns to exclude from header installation.
set(_dispenso_install_excludes "")
if(NOT DISPENSO_BUILD_FAST_MATH)
  list(APPEND _dispenso_install_excludes PATTERN "fast_math" EXCLUDE)
endif()
# The vendored headers are never installed under include/dispenso/; see the
# separate install below.
list(APPEND _dispenso_install_excludes PATTERN "third-party" EXCLUDE)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
    PATTERN *.h
    ${_dispenso_install_excludes}
)

if(NOT DISPENSO_USE_SYSTEM_CONCURRENTQUEUE)
  # Install the bundled copy at <prefix>/include/moodycamel. Public dispenso
  # headers include <moodycamel/concurrentqueue.h>, so this is what lets a
  # single -I<prefix>/include resolve them: an installed dispenso stays usable
  # from build systems that do not consume the exported CMake target.
  #
  # This is not the standalone package's layout -- vcpkg installs it under
  # <prefix>/include/concurrentqueue/moodycamel, and consumers reach it through
  # the include directory on concurrentqueue::concurrentqueue rather than the
  # prefix root. Only the bundled path needs to work without a CMake target.
  install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/third-party/moodycamel
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
      PATTERN *.h
      PATTERN LICENSE.md
  )
endif()

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
