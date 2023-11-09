# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.12)


include(FetchContent)

if(CMAKE_CXX_STANDARD LESS 17)
  message(STATUS, "Using older taskflow due to c++14 support")
FetchContent_Declare(
  taskflow
  GIT_REPOSITORY https://github.com/taskflow/taskflow.git
  GIT_TAG        v2.7.0
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  )
else()
  message(STATUS, "Using up-to-date taskflow")
FetchContent_Declare(
  taskflow
  GIT_REPOSITORY https://github.com/taskflow/taskflow.git
  GIT_TAG        v3.6.0
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  )
endif()
FetchContent_GetProperties(taskflow)
if(NOT taskflow_POPULATED)
  FetchContent_Populate(taskflow)
endif()

FetchContent_MakeAvailable(taskflow)

add_library(taskflow INTERFACE)
target_include_directories(taskflow INTERFACE ${taskflow_SOURCE_DIR})

find_package(benchmark REQUIRED)
if (NOT WIN32)
  find_package(OpenMP)
endif (NOT WIN32)
find_package(TBB)
find_package(folly)

if (WIN32)
  set (REQUIRED_LIBS dispenso benchmark::benchmark benchmark::benchmark_main taskflow)
else (WIN32)
  set (REQUIRED_LIBS dispenso benchmark::benchmark benchmark::benchmark_main pthread taskflow)
endif (WIN32)

if (TBB_FOUND)
  set (OPTIONAL_LIBS ${OPTIONAL_LIBS} tbb)
else (TBB_FOUND)
  add_compile_definitions(BENCHMARK_WITHOUT_TBB)
endif (TBB_FOUND)

if (OpenMP_CXX_FOUND)
  set (OPTIONAL_LIBS ${OPTIONAL_LIBS} OpenMP::OpenMP_CXX)
endif (OpenMP_CXX_FOUND)

if (FOLLY_LIBRARIES AND NOT ${BENCHMARK_WITHOUT_FOLLY})
  find_package(gflags)
  set (OPTIONAL_LIBS ${OPTIONAL_LIBS} ${FOLLY_LIBRARIES})
else (FOLLY_LIBRARIES AND NOT ${BENCHMARK_WITHOUT_FOLLY})
  add_compile_definitions(BENCHMARK_WITHOUT_FOLLY)
endif (FOLLY_LIBRARIES AND NOT ${BENCHMARK_WITHOUT_FOLLY})

file(GLOB BENCHMARK_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/benchmarks/*.cpp)

foreach(BENCHMARK_FILE ${BENCHMARK_FILES})
  set(BENCHMARK_NAME)
  get_filename_component(BENCHMARK_NAME ${BENCHMARK_FILE} NAME_WE)
  add_executable(${BENCHMARK_NAME} ${BENCHMARK_FILE})
  target_link_libraries(${BENCHMARK_NAME} ${REQUIRED_LIBS} ${OPTIONAL_LIBS})
endforeach()

