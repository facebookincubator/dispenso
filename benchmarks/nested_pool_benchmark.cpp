/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmarks for nested/hierarchical task scheduling patterns.
 * Tests scenarios where tasks spawn additional child tasks.
 */

#include <cmath>

#include <dispenso/task_set.h>

#if !defined(BENCHMARK_WITHOUT_TBB)
#include <tbb/task_group.h>
#include "tbb_compat.h"
#endif // !BENCHMARK_WITHOUT_TBB

#if !defined(BENCHMARK_WITHOUT_FOLLY)
#include <folly/VirtualExecutor.h>
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Collect.h>
#include <folly/coro/Coroutine.h>
#include <folly/coro/Task.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#endif // !BENCHMARK_WITHOUT_FOLLY

#include "thread_benchmark_common.h"

static constexpr int kSmallSize = 1000;
static constexpr int kMediumSize = 10000;
static constexpr int kLargeSize = 300000;

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
    dispenso::TaskSet outerTasks(pool);
    for (int i = 0; i < num_elements; ++i) {
      outerTasks.schedule([&pool, num_elements]() {
        int num = std::sqrt(num_elements);
        dispenso::TaskSet tasks(pool);
        for (int j = 0; j < num; ++j) {
          tasks.schedule([j]() { work() += j; });
        }
      });
    }
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
      g.run([num_elements]() {
        int num = std::sqrt(num_elements);
        tbb::task_group g2;
        for (int j = 0; j < num; ++j) {
          g2.run([j]() { work() += j; });
        }
        g2.wait();
      });
    }
    g.wait();
  }
}
#endif // !BENCHMARK_WITHOUT_TBB

#if !defined(BENCHMARK_WITHOUT_FOLLY)
void BM_folly(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);

  if (num_elements > 10000) {
    state.SkipWithError("We run out of memory here with too many elements");
  }

  folly::CPUThreadPoolExecutor follyExec(num_threads);
  for (auto UNUSED_VAR : state) {
    folly::coro::blockingWait([&]() -> folly::coro::Task<void> {
      std::vector<folly::coro::Task<void>> tasks;
      for (int i = 0; i < num_elements; ++i) {
        tasks.push_back(folly::coro::co_invoke([num_elements]() -> folly::coro::Task<void> {
          co_await folly::coro::co_reschedule_on_current_executor;
          std::vector<folly::coro::Task<void>> tasks2;
          int num = std::sqrt(num_elements);
          for (int j = 0; j < num; ++j) {
            tasks2.push_back(folly::coro::co_invoke([j]() -> folly::coro::Task<void> {
              co_await folly::coro::co_reschedule_on_current_executor;
              work() += j;
            }));
          }
          co_await folly::coro::collectAllRange(std::move(tasks2));
        }));
      }
      co_await folly::coro::collectAllRange(std::move(tasks)).scheduleOn(&follyExec);
    }());
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

BENCHMARK_MAIN();
