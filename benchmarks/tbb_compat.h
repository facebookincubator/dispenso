/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * TBB compatibility header for supporting both legacy TBB and oneTBB (2021+).
 *
 * oneTBB 2021+ removed task_scheduler_init in favor of global_control and task_arena.
 * oneTBB also changed pipeline.h to parallel_pipeline.h and renamed filter types.
 * This header provides a unified interface that works with both versions.
 */

#if !defined(BENCHMARK_WITHOUT_TBB)

// Include version header to detect TBB version
#if __has_include(<tbb/version.h>)
#include <tbb/version.h>
#elif __has_include(<tbb/tbb_stddef.h>)
#include <tbb/tbb_stddef.h>
#endif

// Detect oneTBB (2021+) vs legacy TBB
#if defined(TBB_VERSION_MAJOR) && TBB_VERSION_MAJOR >= 2021
#define TBB_USE_ONETBB 1
#else
#define TBB_USE_ONETBB 0
#endif

#if TBB_USE_ONETBB
// oneTBB 2021+ uses global_control instead of task_scheduler_init
#include <tbb/global_control.h>
#include <tbb/parallel_pipeline.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

namespace tbb_compat {

// RAII wrapper for thread count control, compatible with old task_scheduler_init API
class task_scheduler_init {
 public:
  explicit task_scheduler_init(int num_threads)
      : control_(tbb::global_control::max_allowed_parallelism, num_threads) {}

 private:
  tbb::global_control control_;
};

// Pipeline filter mode compatibility
// In oneTBB, tbb::filter::serial -> tbb::filter_mode::serial_in_order
// In oneTBB, tbb::filter::parallel -> tbb::filter_mode::parallel
namespace filter {
constexpr auto serial = tbb::filter_mode::serial_in_order;
constexpr auto parallel = tbb::filter_mode::parallel;
} // namespace filter

} // namespace tbb_compat

#else
// Legacy TBB (< 2021)
#include <tbb/pipeline.h>
#include <tbb/task_group.h>
#include <tbb/task_scheduler_init.h>

namespace tbb_compat {

using task_scheduler_init = tbb::task_scheduler_init;

// Pipeline filter mode compatibility - just alias the legacy types
namespace filter {
constexpr auto serial = tbb::filter::serial;
constexpr auto parallel = tbb::filter::parallel;
} // namespace filter

} // namespace tbb_compat

#endif // TBB_USE_ONETBB

#endif // !BENCHMARK_WITHOUT_TBB
