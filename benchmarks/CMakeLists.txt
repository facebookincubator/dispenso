cmake_minimum_required(VERSION 3.12)

find_package(benchmark REQUIRED)
if (NOT WIN32)
  find_package(OpenMP)
endif (NOT WIN32)
find_package(TBB)
find_package(folly)

if (WIN32)
  set (REQUIRED_LIBS dispenso benchmark::benchmark benchmark::benchmark_main)
else (WIN32)
  set (REQUIRED_LIBS dispenso benchmark::benchmark benchmark::benchmark_main pthread)
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

