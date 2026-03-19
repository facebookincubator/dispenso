/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Basic thread pool benchmarks comparing dispenso, TBB, and Folly.
 * Tests simple task scheduling throughput.
 */

#include <dispenso/task_set.h>

#if !defined(BENCHMARK_WITHOUT_TBB)
#include <tbb/task_group.h>
#include "tbb_compat.h"
#endif // !BENCHMARK_WITHOUT_TBB

#if !defined(BENCHMARK_WITHOUT_FOLLY)
#include <folly/VirtualExecutor.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#endif // !BENCHMARK_WITHOUT_FOLLY

#include "thread_benchmark_common.h"

static constexpr int kSmallSize = 1000;
static constexpr int kMediumSize = 10000;
static constexpr int kLargeSize = 1000000;

struct alignas(64) Work {
  size_t count = 0;

  void operator+=(size_t o) {
    count += o;
  }
};

Work g_work[1025];
std::atomic<int> g_tCounter{0};
inline int testTid() {
  static DISPENSO_THREAD_LOCAL int t = -1;
  if (t < 0) {
    t = g_tCounter.fetch_add(1, std::memory_order_acq_rel);
  }
  return t;
}

inline Work& work() {
  static DISPENSO_THREAD_LOCAL Work* w = nullptr;

  if (!w) {
    if (testTid() == 0) {
      w = g_work + 1024;
    } else {
      w = g_work + (testTid() & 1023);
    }
  }
  return *w;
}

void BM_dispenso(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const int num_elements = state.range(1);
  dispenso::ThreadPool pool(num_threads);

  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);
    for (int i = 0; i < num_elements; ++i) {
      tasks.schedule([i]() { work() += i; });
    }
  }
}

void BM_dispenso_bulk(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const int num_elements = state.range(1);
  dispenso::ThreadPool pool(num_threads);

  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);
    tasks.scheduleBulk(
        static_cast<size_t>(num_elements), [](size_t i) { return [i]() { work() += i; }; });
  }
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);
  tbb_compat::task_scheduler_init initsched(num_threads);

  for (auto UNUSED_VAR : state) {
    tbb::task_group g;
    for (int i = 0; i < num_elements; ++i) {
      g.run([i]() { work() += i; });
    }
    g.wait();
  }
}
#endif // !BENCHMARK_WITHOUT_TBB

#if !defined(BENCHMARK_WITHOUT_FOLLY)
void BM_folly(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);
  folly::CPUThreadPoolExecutor follyExec(num_threads);

  for (auto UNUSED_VAR : state) {
    folly::VirtualExecutor tasks(&follyExec);
    for (int i = 0; i < num_elements; ++i) {
      tasks.add([i]() { work() += i; });
    }
  }
}
#endif // !BENCHMARK_WITHOUT_FOLLY

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int j : {kSmallSize, kMediumSize, kLargeSize}) {
    for (int s : pow2HalfStepThreads()) {
      b->Args({s, j});
    }
  }
}

#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb)->Apply(CustomArguments)->Unit(benchmark::kMicrosecond)->UseRealTime();
#endif // !BENCHMARK_WITHOUT_TBB

#if !defined(BENCHMARK_WITHOUT_FOLLY)
BENCHMARK(BM_folly)->Apply(CustomArguments)->Unit(benchmark::kMicrosecond)->UseRealTime();
#endif // !BENCHMARK_WITHOUT_FOLLY

BENCHMARK(BM_dispenso)->Apply(CustomArguments)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_dispenso_bulk)->Apply(CustomArguments)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK_MAIN();
