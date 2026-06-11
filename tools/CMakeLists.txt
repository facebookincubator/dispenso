# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Development/tuning tools — not user-facing benchmarks.
# wake_cost_bench: platform syscall cost measurement for wake tuning.
# mandelbrot_instrument: scheduler quality analysis (chunk distribution, affinity).

if (WIN32)
  set(_TOOL_THREAD_LIB)
else()
  set(_TOOL_THREAD_LIB pthread)
endif()

add_executable(wake_cost_bench wake_cost_bench.cpp)
target_compile_features(wake_cost_bench PRIVATE cxx_std_14)
target_link_libraries(wake_cost_bench dispenso ${_TOOL_THREAD_LIB})

# mandelbrot_instrument requires TBB. Check for imported targets directly
# rather than TBB_FOUND, since find_package(TBB) runs in benchmarks/CMakeLists.txt
# (a sibling subdirectory whose variables don't propagate here).
if(TARGET TBB::tbb)
  set(_TOOL_TBB_LIB TBB::tbb)
elseif(TARGET tbb)
  set(_TOOL_TBB_LIB tbb)
endif()

if(_TOOL_TBB_LIB)
  add_executable(mandelbrot_instrument mandelbrot_instrument.cpp)
  target_compile_features(mandelbrot_instrument PRIVATE cxx_std_20)
  target_include_directories(mandelbrot_instrument PRIVATE ${PROJECT_SOURCE_DIR}/benchmarks)
  target_link_libraries(mandelbrot_instrument dispenso ${_TOOL_TBB_LIB} ${_TOOL_THREAD_LIB})
else()
  message(STATUS "TBB not found — skipping mandelbrot_instrument (requires TBB)")
endif()
